import json
import shutil
import struct
import time
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Union, Any, Tuple

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import sqlglot
from sqlglot import exp

from ..http_client import HTTPRangeClient, RemoteFileObject

from .meta import RowGroupMeta
from .io import FusedFileObject
from .utils import get_url_hash, get_query_hash, atomic_write, extract_partition_column
from .logger import flow_log, logger

# Constants for magic numbers and boundaries
PARQUET_FOOTER_SIZE = 8  # Last 8 bytes of parquet file contain metadata length and magic
PARQUET_MAGIC = b'PAR1'
MIN_PARQUET_FILE_SIZE = 100  # Minimum valid parquet file size in bytes
DEFAULT_PARTITION_CANDIDATES = ["timestamp", "date", "id", "key", "uuid", "pk"]
LARGE_VALUE_SET_THRESHOLD_PERCENT = 0.2  # 20% of row groups
LARGE_VALUE_SET_THRESHOLD_MIN = 10  # Absolute minimum for large value set
FULL_FILE_CHUNK_SIZE = 10 * 1024 * 1024  # 10MB chunks for full file download
BUFFER_MULTIPLIER = 1.1  # Buffer size multiplier for partial fetches

class SmartParquetReader:
    """Read and query Parquet files from HTTP sources with intelligent caching."""
    
    def __init__(self, url: str, partition_col: Optional[str] = None, cache_dir: Union[str, Path] = "./smart_cache", cache_ttl: int = 600, debug: bool = False):
        if not url:
            raise ValueError("url cannot be empty")
        if not isinstance(url, str):
            raise TypeError(f"url must be a string, got {type(url).__name__}")
        if cache_ttl < 0:
            raise ValueError(f"cache_ttl must be non-negative, got {cache_ttl}")
        
        self.url = url
        self.partition_col = partition_col
        self.root_cache_dir = Path(cache_dir).resolve()
        self.cache_ttl = cache_ttl  # TTL in seconds (default 10 minutes)
        self.debug = debug  # Enable/disable flow logs
        self.client = HTTPRangeClient(url)
        
        # Internal State
        self.row_groups: List[RowGroupMeta] = []
        self._footer_bytes: Optional[bytes] = None
        self._file_size = 0
        self._is_initialized = False
        self._detected_col: str = ""
        
        # Cache paths
        self.cache_path: Optional[Path] = None
        self.etag_path: Optional[Path] = None
        self.footer_path: Optional[Path] = None
        self.meta_path: Optional[Path] = None
        self.query_cache_dir: Optional[Path] = None
        self.sql_result_cache_dir: Optional[Path] = None
        self.timestamp_path: Optional[Path] = None

    def initialize(self):
        if self._is_initialized:
            return

        if self.debug:
            flow_log.start_timer()
        
        # 1. Setup Cache Paths (lightweight)
        self._setup_cache_paths()
        
        if self._try_load_from_cache():
            self._is_initialized = True
            return
        
        # 3. Fetch Remote Metadata
        try:
            self._file_size = self.client.get_file_size()
            if self._file_size == 0:
                raise ValueError("Remote file size is 0 bytes")
            if self._file_size < MIN_PARQUET_FILE_SIZE:
                raise ValueError(f"File too small: {self._file_size} bytes")
            remote_etag = self._fetch_remote_etag() or f"size_{self._file_size}"
        except Exception as e:
            if self.debug:
                flow_log.node(f"Connection Failed", {"Error": str(e)}, style="error")
            raise

        local_etag = self.etag_path.read_text().strip() if self.etag_path.exists() else None
        cache_valid = (local_etag == remote_etag) and self.footer_path.exists() and self.meta_path.exists()

        if cache_valid:
            try:
                self._load_metadata()
                if self.debug:
                    flow_log.node("Initialize (Cache HIT)", {"URL": self.url}, style="success")
            except ValueError as e:
                if self.debug:
                    flow_log.node("Cache Invalid", {"Error": str(e)}, style="warning")
                cache_valid = False
        
        if not cache_valid:
            if local_etag:
                self._purge_cache()
            self._warmup_cache(remote_etag)

        self._is_initialized = True

    def _try_load_from_cache(self) -> bool:
        """Attempt to load metadata from cache if available and valid."""
        if not (self._is_cache_within_ttl() and self.etag_path.exists() and 
                self.footer_path.exists() and self.meta_path.exists()):
            return False
        
        try:
            with open(self.meta_path, "r") as f:
                metadata = json.load(f)
                self._file_size = metadata.get('file_size', 0) or self.client.get_file_size()
            
            self._load_metadata()
            if self.debug:
                flow_log.node("Initialize (Cache HIT - TTL)", {"URL": self.url}, style="success")
            return True
        except (ValueError, IOError, json.JSONDecodeError):
            if self.debug:
                flow_log.node("Cache Load Failed", style="warning")
            return False

    def _is_cache_within_ttl(self) -> bool:
        """Check if cache timestamp is within TTL window."""
        if not self.timestamp_path or not self.timestamp_path.exists():
            return False
        try:
            timestamp = float(self.timestamp_path.read_text().strip())
            age = time.time() - timestamp
            return age < self.cache_ttl
        except Exception:
            return False

    def _fetch_remote_etag(self) -> Optional[str]:
        """Fetch remote ETag from HTTP headers.
        
        Attempts to reuse headers from last request to avoid extra HEAD call.
        Falls back to a new HEAD request if necessary.
        
        Returns:
            ETag string or None if unable to fetch
        """
        if self.client._last_headers:
            tag = self.client._last_headers.get("ETag", "") or self.client._last_headers.get("etag", "")
            if tag:
                return tag.strip('"')
        
        # Fallback: make a new HEAD request
        try:
            resp = self.client._session.head(self.url, allow_redirects=True)
            tag = resp.headers.get("ETag", "") or resp.headers.get("etag", "")
            return tag.strip('"')
        except Exception:
            return None

    def _purge_cache(self):
        """Remove all cached files and directories for this URL.
        
        Attempts to delete footer, metadata, etag, timestamp, and query result caches.
        Warning messages are logged for any files that fail to delete.
        """
        try:
            if self.footer_path.exists(): self.footer_path.unlink()
            if self.meta_path.exists(): self.meta_path.unlink()
            if self.etag_path.exists(): self.etag_path.unlink()
            if self.timestamp_path and self.timestamp_path.exists(): self.timestamp_path.unlink()
            if self.query_cache_dir.exists(): shutil.rmtree(self.query_cache_dir)
            if self.sql_result_cache_dir.exists(): shutil.rmtree(self.sql_result_cache_dir)
        except Exception as e:
            logger.warning(f"Purge error: {e}")

    def _load_metadata(self):
        """Load and validate cached parquet metadata."""
        try:
            with open(self.footer_path, "rb") as f:
                self._footer_bytes = f.read()
            
            if len(self._footer_bytes) < PARQUET_FOOTER_SIZE:
                raise ValueError(f"Corrupted footer: expected >= {PARQUET_FOOTER_SIZE} bytes, got {len(self._footer_bytes)}")
            
            with open(self.meta_path, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    groups_data = data
                    self._detected_col = self.partition_col or "legacy_unknown"
                else:
                    groups_data = data.get('groups', [])
                    self._detected_col = data.get('column', 'unknown')

                if not groups_data:
                    raise ValueError("Metadata contains no row groups")
                
                self.row_groups = [RowGroupMeta(**d) for d in groups_data]
        except (json.JSONDecodeError, ValueError, KeyError, TypeError) as e:
            if self.debug:
                flow_log.node("Metadata Error", {"Error": str(e)}, style="error")
            raise ValueError(f"Corrupted metadata: {str(e)}") from e
        except IOError as e:
            raise ValueError(f"Failed to read metadata files: {str(e)}") from e

    def _warmup_cache(self, etag: str):
        """Download and parse Parquet footer from remote source to initialize cache."""
        if not etag:
            raise ValueError("etag cannot be empty")
        
        t0 = time.monotonic()
        
        # A. Validate file size
        if self._file_size < MIN_PARQUET_FILE_SIZE:
            raise ValueError(f"Invalid Parquet file: size is {self._file_size} bytes (minimum {MIN_PARQUET_FILE_SIZE} required)")
        
        # A. Download Raw Footer
        try:
            tail_bytes = self.client.read_range(self._file_size - PARQUET_FOOTER_SIZE, self._file_size - 1)
        except Exception as e:
            raise ValueError(f"Failed to read parquet footer: {str(e)}") from e
        
        if len(tail_bytes) < PARQUET_FOOTER_SIZE:
            raise ValueError(f"Failed to read parquet footer: expected {PARQUET_FOOTER_SIZE} bytes, got {len(tail_bytes)}")
        if tail_bytes[4:] != PARQUET_MAGIC:
            raise ValueError(f"Invalid Parquet Magic: file does not end with {PARQUET_MAGIC.decode('ascii')}")
            
        meta_len = struct.unpack("<I", tail_bytes[:4])[0]
        if meta_len < 0 or meta_len > self._file_size - PARQUET_FOOTER_SIZE:
            raise ValueError(f"Invalid Parquet metadata length: {meta_len}")
            
        total_footer_len = meta_len + PARQUET_FOOTER_SIZE
        
        footer_start = self._file_size - total_footer_len
        if footer_start < 0:
            raise ValueError(f"Invalid footer offset: {footer_start}")
            
        try:
            self._footer_bytes = self.client.read_range(footer_start, self._file_size - 1)
        except Exception as e:
            raise ValueError(f"Failed to read complete footer: {str(e)}") from e
        
        if len(self._footer_bytes) != total_footer_len:
            raise ValueError(f"Failed to read complete footer: expected {total_footer_len} bytes, got {len(self._footer_bytes)}")
        
        try:
            atomic_write(self.footer_path, self._footer_bytes)
        except Exception as e:
            raise ValueError(f"Failed to write footer cache: {str(e)}") from e
            
        # B. Parse Stats 
        try:
            with RemoteFileObject(self.client, buffer_size=max(64*1024, total_footer_len), close_client=False, file_size=self._file_size) as f:
                pf = pq.ParquetFile(f)
                
                # Resolve Partition Column
                if self.partition_col:
                    if self.partition_col not in pf.schema.names:
                        raise ValueError(f"Partition column '{self.partition_col}' not found in schema: {pf.schema.names}")
                    self._detected_col = self.partition_col
                else:
                    try:
                        self._detected_col = next(name for name in DEFAULT_PARTITION_CANDIDATES if name in pf.schema.names)
                    except StopIteration:
                        self._detected_col = pf.schema.names[0]

                col_idx = pf.schema.names.index(self._detected_col)
                
                results = []
                for i in range(pf.num_row_groups):
                    try:
                        rg = pf.metadata.row_group(i)
                        stats = rg.column(col_idx).statistics
                        
                        if stats and stats.has_min_max:
                            min_v = stats.min
                            max_v = stats.max
                            if isinstance(min_v, bytes): min_v = min_v.decode('utf-8', errors='replace')
                            if isinstance(max_v, bytes): max_v = max_v.decode('utf-8', errors='replace')

                            size = sum(rg.column(c).total_compressed_size for c in range(rg.num_columns))
                            results.append(RowGroupMeta(i, min_v, max_v, size))
                    except Exception as e:
                        logger.warning(f"Failed to extract statistics for row group {i}: {e}")
                        continue
                
                self.row_groups = results
                if not results:
                    raise ValueError("No row groups with statistics found in parquet file")
                
                save_data = {
                    "column": self._detected_col,
                    "file_size": self._file_size,
                    "groups": [asdict(r) for r in results]
                }
                try:
                    atomic_write(self.meta_path, json.dumps(save_data))
                except Exception as e:
                    raise ValueError(f"Failed to write metadata cache: {str(e)}") from e
        except Exception as e:
            if isinstance(e, ValueError):
                raise
            raise ValueError(f"Failed to parse parquet metadata: {str(e)}") from e

        try:
            atomic_write(self.etag_path, etag)
            atomic_write(self.timestamp_path, str(time.time()))
        except Exception as e:
            raise ValueError(f"Failed to write cache metadata: {str(e)}") from e
        
        duration = time.monotonic() - t0
        if self.debug:
            flow_log.node("Warmup Complete", {
                "Duration": f"{duration:.2f}s",
                "Key Column": self._detected_col,
                "Row Groups": len(results)
            }, style="success")

    def _extract_filters(self, sql: str) -> Tuple[str, List[Any]]:
        """Parse SQL to extract filter values for the partition column.
        
        Extracts WHERE clause conditions to identify row groups that need to be fetched.
        Returns (column_name, [values]).
        
        Args:
            sql: SQL query string with {url} placeholder
            
        Returns:
            Tuple of (column_name, list_of_filter_values). Returns empty list if no filters found.
        """
        try:
            parsed = sqlglot.parse_one(sql)
        except Exception:
            return self._detected_col, []

        where = parsed.find(exp.Where)
        if not where:
            return self._detected_col, []

        found_values = []
        col_lower = self._detected_col.lower()

        for eq_expr in where.find_all(exp.EQ):
            col = eq_expr.find(exp.Column)
            lit = eq_expr.find(exp.Literal)
            if col and lit and col.name.lower() == col_lower:
                found_values.append(lit.this)

        for in_expr in where.find_all(exp.In):
            col = in_expr.this
            if isinstance(col, exp.Column) and col.name.lower() == col_lower:
                for arg in in_expr.args.get("expressions", []):
                    if isinstance(arg, exp.Literal):
                        found_values.append(arg.this)

        return self._detected_col, list(set(found_values))

    def _value_in_range(self, value: Any, min_val: Any, max_val: Any) -> bool:
        """Check if value falls within min and max range, with type coercion."""
        try:
            return min_val <= value <= max_val
        except TypeError:
            try:
                return str(min_val) <= str(value) <= str(max_val)
            except Exception:
                return False

    def _try_load_arrow_cache(self, cache_path: Path, timestamp_path: Optional[Path] = None) -> Optional[pa.Table]:
        """Attempt to load Arrow table from cache file and return if within TTL."""
        if not cache_path.exists():
            return None
        
        # Check timestamp if provided
        if timestamp_path and timestamp_path.exists():
            try:
                cache_timestamp = float(timestamp_path.read_text().strip())
                age = time.time() - cache_timestamp
                if age >= self.cache_ttl:
                    return None  # Cache expired
            except Exception:
                return None  # Timestamp read failed
        
        # Try to load the table
        try:
            return pa.ipc.open_file(cache_path).read_all()
        except Exception as e:
            logger.warning(f"Failed to read cache {cache_path}: {e}")
            return None

    def _save_arrow_cache_safe(self, cache_path: Path, table: pa.Table, timestamp_path: Optional[Path] = None) -> bool:
        """Safely save Arrow table to cache with optional timestamp."""
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._save_arrow_cache(cache_path, table)
            if timestamp_path:
                atomic_write(timestamp_path, str(time.time()))
            return True
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
            return False

    def _extract_projected_columns(self, sql: str) -> Optional[List[str]]:
        """Extract column names from SELECT clause for column pushdown optimization.
        
        Analyzes the SELECT clause to determine which columns will be used,
        allowing us to skip loading unused columns from disk.
        
        Args:
            sql: SQL query string with {url} placeholder
            
        Returns:
            List of column names, or None if SELECT * or extraction fails
        """
        try:
            parsed = sqlglot.parse_one(sql)
            select = parsed.find(exp.Select)
            
            if not select:
                return None
            
            # Check for SELECT *
            for expr in select.expressions:
                if isinstance(expr, exp.Star):
                    return None
            
            # Extract column names
            columns = []
            for expr in select.expressions:
                if isinstance(expr, exp.Column):
                    columns.append(expr.name)
                elif isinstance(expr, exp.Alias):
                    # For aliased columns, use the original column name
                    if isinstance(expr.this, exp.Column):
                        columns.append(expr.this.name)
                elif isinstance(expr, exp.Literal):
                    # Skip literal values (e.g., constants in SELECT)
                    pass
                else:
                    # For complex expressions, try to extract column names
                    for col in expr.find_all(exp.Column):
                        if col.name not in columns:
                            columns.append(col.name)
            
            return columns if columns else None
        except Exception:
            return None

    def _setup_cache_paths(self):
        """Set up cache directory structure without loading metadata.
        
        Creates necessary cache directories for storing parquet footer,
        metadata, query results, and SQL result caches.
        
        This is a lightweight operation that doesn't require file I/O.
        """
        url_hash = get_url_hash(self.url)
        self.cache_path = self.root_cache_dir / url_hash
        self.cache_path.mkdir(parents=True, exist_ok=True)
        
        self.etag_path = self.cache_path / "etag.txt"
        self.footer_path = self.cache_path / "footer.bin"
        self.meta_path = self.cache_path / "metadata.json"
        self.timestamp_path = self.cache_path / "timestamp.txt"
        self.query_cache_dir = self.cache_path / "queries"
        self.query_cache_dir.mkdir(exist_ok=True)
        self.sql_result_cache_dir = self.cache_path / "sql_results"
        self.sql_result_cache_dir.mkdir(exist_ok=True)

    def _auto_detect_partition_column(self, sql: str) -> None:
        """Automatically detect partition column from SQL WHERE clause if not provided.
        
        Only detects if partition_col is not already set. Uses extract_partition_column
        to parse the SQL and identify the partition column from the WHERE clause.
        
        Args:
            sql: SQL query string to parse
        """
        if not self.partition_col:
            detected = extract_partition_column(sql)
            if detected:
                self.partition_col = detected
                if self.debug:
                    logger.debug(f"Auto-detected partition column from SQL: {detected}")

    def execute_sql(self, sql: str, cache_result: bool = False) -> pa.Table:
        """Execute SQL query with automatic predicate pushdown and result caching.
        
        This method performs the following optimizations:
        1. Auto-detects partition column from SQL WHERE clause if not provided
        2. Checks SQL result cache first (instant return if available and within TTL)
        3. Extracts filters from WHERE clause for row group pruning
        4. Extracts column names from SELECT clause for column pushdown
        5. Downloads only required row groups
        6. Caches results for subsequent queries
        
        Args:
            sql: SQL query string. Use {url} placeholder for the parquet file URL.
                 For example: "SELECT * FROM {url} WHERE key = 'VALUE'"
            cache_result: If True, cache the final SQL result for subsequent calls
            
        Returns:
            PyArrow Table with query results
            
        Raises:
            ValueError: If SQL parsing fails or data is invalid
            Exception: Other exceptions from underlying libraries (duckdb, pyarrow)
        """
        # 0. Set up cache paths (lightweight, no metadata loading)
        self._setup_cache_paths()
        
        # 0b. Auto-detect partition column from SQL if not provided
        self._auto_detect_partition_column(sql)
        
        # 1. Check if final SQL result is cached and within TTL (instant return, zero DuckDB!)
        sql_hash = get_query_hash(sql, None)
        sql_cache_path = self.sql_result_cache_dir / f"{sql_hash}.arrow"
        sql_cache_timestamp_path = self.sql_result_cache_dir / f"{sql_hash}.timestamp"
        
        cached_result = self._try_load_arrow_cache(sql_cache_path, sql_cache_timestamp_path)
        if cached_result is not None:
            if self.debug:
                age = time.time() - float(sql_cache_timestamp_path.read_text().strip())
                flow_log.node("SQL Result Cache HIT", {"Path": sql_cache_path.name, "Age": f"{age:.2f}s"})
            return cached_result
        
        # 2. Cache miss - load metadata and proceed
        self.initialize()
        
        # 3. Pre-process SQL for Parsing
        parseable_sql = sql.replace("{url}", "source_table")
        
        # 3. Parse SQL for filters and projected columns
        col_name, filter_values = self._extract_filters(parseable_sql)
        projected_columns = self._extract_projected_columns(parseable_sql)
        
        # 4. Smart Fetch
        if filter_values:
            if self.debug:
                flow_log.node("Smart Optimization", {
                    "Inferred Filter": f"{col_name} IN {filter_values}",
                    "Projected Columns": projected_columns or "all"
                })
            arrow_data = self._smart_fetch(filter_values, cache_result=cache_result, columns=projected_columns)
        else:
            if self.debug:
                flow_log.node("Full Scan Warning", {
                    "Reason": f"No simple '{self._detected_col}' filter found.",
                    "Projected Columns": projected_columns or "all"
                }, style="warning")
            arrow_data = self._smart_fetch(None, cache_result=False, columns=projected_columns)

        if len(arrow_data) == 0:
            return pa.Table.from_pylist([])

        # 5. DuckDB Execution
        con = duckdb.connect()
        try:
            con.register('source_view', arrow_data)
            final_sql = sql.replace("{url}", "source_view")
            
            if self.debug:
                flow_log.node("Executing SQL", {"Query": final_sql.replace('\n', ' ')[:50]+"..."})
            result = con.execute(final_sql).arrow()
            if isinstance(result, pa.RecordBatchReader):
                result = result.read_all()
            
            # Cache the final SQL result for instant retrieval on subsequent calls
            if self._save_arrow_cache_safe(sql_cache_path, result, sql_cache_timestamp_path):
                if self.debug:
                    flow_log.node("SQL Result Cached", {"Path": sql_cache_path.name})
            
            if self.debug:
                flow_log.end("Success", {"Rows": len(result)})
            return result
        except Exception as e:
            if self.debug:
                flow_log.node("SQL Error", {"Message": str(e)}, style="error")
            raise
        finally:
            con.close()


    def _smart_fetch(self, values: Optional[List[Any]], cache_result: bool, columns: Optional[List[str]] = None) -> pa.Table:
        """Internal method to fetch row groups based on values.
        
        Args:
            values: List of partition values to filter by, or None for full scan
            cache_result: Whether to cache the result for future queries
            columns: List of column names to read (None = all columns)
            
        Returns:
            PyArrow Table with filtered rows
        """
        if self.debug:
            flow_log.start_timer()
        
        # Calculate Query Hash and check cache
        query_hash_key = str(sorted(values)) if values else "FULL_SCAN"
        
        # Check Cache
        if cache_result:
            query_hash = get_query_hash(query_hash_key, None)
            cache_path = self.query_cache_dir / f"{query_hash}.arrow"
            cached_table = self._try_load_arrow_cache(cache_path)
            if cached_table is not None:
                if self.debug:
                    flow_log.node("Cache HIT", {"Path": cache_path.name})
                return cached_table

        # Filter Row Groups
        if values:
            # For large value sets, do full scan to avoid expensive filtering
            large_value_threshold = max(LARGE_VALUE_SET_THRESHOLD_MIN, int(len(self.row_groups) * LARGE_VALUE_SET_THRESHOLD_PERCENT))
            if len(values) > large_value_threshold:
                if self.debug:
                    flow_log.node("Large Value Set", {
                        "Values": len(values),
                        "Row Groups": len(self.row_groups),
                        "Approach": "Full scan"
                    })
                return self._fetch_full_file(columns)
            
            target_indices = []
            target_sizes = []
            for rg in self.row_groups:
                for v in values:
                    if self._value_in_range(v, rg.min_val, rg.max_val):
                        target_indices.append(rg.index)
                        target_sizes.append(rg.compressed_size)
                        break
        else:
            target_indices = [rg.index for rg in self.row_groups]
            target_sizes = [rg.compressed_size for rg in self.row_groups]

        if not target_indices:
            return pa.Table.from_pylist([])

        # For full scans or all row groups, download entire file
        if values is None or len(target_indices) == len(self.row_groups):
            if self.debug:
                flow_log.node("Full File Download", {
                    "Reason": "Full scan or all row groups needed",
                    "Size": f"{self._file_size / 1024 / 1024:.2f} MB"
                })
            return self._fetch_full_file(columns)
        
        # Partial fetch using FusedFileObject
        max_chunk = max(target_sizes) if target_sizes else FULL_FILE_CHUNK_SIZE
        buffer_size = int(max_chunk * BUFFER_MULTIPLIER)
        
        try:
            with FusedFileObject(self.client, self._footer_bytes, self._file_size, buffer_size) as f:
                pf = pq.ParquetFile(f)
                # Push down column selection to read_row_groups for efficiency
                if columns:
                    table = pf.read_row_groups(target_indices, columns=columns)
                else:
                    table = pf.read_row_groups(target_indices)
                
                # Memory Filter
                if values:
                    filter_col = self._detected_col if self._detected_col in table.column_names else table.column_names[0]
                    mask = pc.is_in(table[filter_col], value_set=pa.array(values))
                    table = table.filter(mask)
        except Exception as e:
            if self.debug:
                flow_log.node("Partial Fetch Failed - Falling back to full download", {
                    "Error": str(e)[:50],
                    "Row Groups": len(target_indices)
                }, style="warning")
            else:
                logger.warning(f"Partial fetch failed (falling back to full download): {e}")
            # Fallback to full file download on any error
            return self._fetch_full_file(columns)
        
        # Save Cache
        if cache_result and values and len(table) > 0:
            query_hash = get_query_hash(query_hash_key, None)
            cache_path = self.query_cache_dir / f"{query_hash}.arrow"
            self._save_arrow_cache_safe(cache_path, table)

        return table

    def _fetch_full_file(self, columns: Optional[List[str]] = None) -> pa.Table:
        """Download and parse the entire parquet file by streaming it to memory.
        
        Args:
            columns: List of column names to read (None = all columns)
            
        Returns:
            PyArrow Table with all rows from the file
        """
        import io
        
        file_data = io.BytesIO()
        chunk_size = FULL_FILE_CHUNK_SIZE
        downloaded = 0
        
        try:
            if self.debug:
                flow_log.node("Downloading Full File", {
                    "Size": f"{self._file_size / 1024 / 1024:.2f} MB",
                    "Chunk Size": f"{chunk_size / 1024 / 1024:.1f} MB"
                })
            
            while downloaded < self._file_size:
                end = min(downloaded + chunk_size, self._file_size)
                try:
                    chunk = self.client.read_range(downloaded, end - 1)
                except Exception as e:
                    raise ValueError(f"Failed to download chunk at offset {downloaded}: {str(e)}") from e
                
                if not chunk:
                    # No more data
                    break
                file_data.write(chunk)
                downloaded += len(chunk)
            
            # Verify download
            total_downloaded = file_data.tell()
            if total_downloaded == 0:
                raise ValueError(f"Failed to download file: 0 bytes received (expected {self._file_size})")
            
            file_data.seek(0)
            
            # Verify parquet magic
            magic = file_data.read(4)
            if magic != PARQUET_MAGIC:
                file_data.seek(0)
                first_bytes = file_data.read(100)
                raise ValueError(f"Invalid Parquet magic: {first_bytes[:10]}, expected {PARQUET_MAGIC}")
            
            file_data.seek(0)
            
            # Parse parquet from BytesIO
            try:
                pf = pq.ParquetFile(file_data)
                if columns:
                    table = pf.read(columns=columns)
                else:
                    table = pf.read()
            except Exception as e:
                raise ValueError(f"Failed to parse parquet file: {str(e)}") from e
                    
            if self.debug:
                flow_log.node("File Downloaded", {
                    "Downloaded": f"{total_downloaded / 1024 / 1024:.2f} MB",
                    "Rows": len(table)
                })
        except Exception as e:
            if isinstance(e, ValueError):
                raise
            if self.debug:
                flow_log.node("Full File Download Failed", {
                    "Error": str(e),
                    "File Size": f"{self._file_size} bytes",
                    "Downloaded": f"{file_data.tell()} bytes"
                }, style="error")
            raise ValueError(f"Failed to download full parquet file: {str(e)}") from e
        
        return table

    def _save_arrow_cache(self, cache_path: Path, table: pa.Table):
        """Atomically save PyArrow table to Arrow IPC format."""
        temp_path = cache_path.with_suffix(".tmp")
        with pa.OSFile(str(temp_path), 'wb') as sink:
            with pa.ipc.new_file(sink, table.schema) as writer:
                writer.write_table(table)
        Path(temp_path).replace(cache_path)

    def query(self, value: Any, columns: Optional[List[str]] = None) -> pa.Table:
        """Fetch rows for a specific partition column value."""
        self.initialize()
        return self._smart_fetch([value], cache_result=True, columns=columns)

    def close(self):
        """Close the HTTP client connection."""
        self.client.close()

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures client is closed."""
        self.close()
