import hashlib
import logging
import re
from pathlib import Path
from typing import Union, List, Optional
import sqlglot
from sqlglot import exp

logger = logging.getLogger("SmartParquet.Utils")

def get_url_hash(url: str) -> str:
    """Generate SHA256 hash of a URL for use as a directory name.
    
    Creates a unique, stable identifier for caching data associated with a URL.
    
    Args:
        url: The HTTP(S) URL to hash
    Returns:
        Hex string of SHA256 hash
    """
    return hashlib.sha256(url.encode()).hexdigest()

def get_query_hash(key: str, columns: Optional[List[str]]) -> str:
    """Generate unique hash for a query configuration.
    
    Combines key and column list to create a unique cache key.
    
    Args:
        key: The partition column value being queried
        columns: List of columns to fetch (or None)
    Returns:
        Hex string of SHA256 hash
    """
    key = f"{key}|{str(columns)}"
    return hashlib.sha256(key.encode()).hexdigest()

def extract_partition_column(sql: str) -> Optional[str]:
    """Extract partition column from WHERE clause for SmartParquetReader optimization.
    
    This function parses SQL queries to identify the column used for filtering,
    allowing SmartParquetReader to skip unnecessary row groups during initialization.
    
    Examples:
        'WHERE uuid = value' -> 'uuid'
        'WHERE related_keys = value' -> 'related_keys'
        'WHERE id IN (...)' -> 'id'
        'WHERE table.id = 1 AND status = 2' -> 'id'

    Args:
        sql: SQL query string
        
    Returns:
        First detected partition column if found, None otherwise
    """
    try:
        parsed = sqlglot.parse_one(sql)
        where = parsed.find(exp.Where)

        if not where:
            return None

        # Look for equality or IN expressions
        for condition in where.find_all((exp.EQ, exp.In)):
            column = None

            # EQ: column = value
            if isinstance(condition, exp.EQ):
                if isinstance(condition.left, exp.Column):
                    column = condition.left

            # IN: column IN (...)
            elif isinstance(condition, exp.In):
                if isinstance(condition.this, exp.Column):
                    column = condition.this

            if column:
                return column.name  # returns unqualified column name

        return None

    except Exception:
        return None

def atomic_write(path: Path, content: Union[str, bytes]):
    """Write content to a file atomically using temp file and rename.
    
    Args:
        path: Target file path
        content: Content to write (str or bytes)
    Raises:
        Exception: If write or rename fails (temp file cleanup attempted)
    """
    temp_path = path.with_suffix(".tmp")
    mode = "wb" if isinstance(content, bytes) else "w"
    
    try:
        with open(temp_path, mode) as f:
            f.write(content)
        temp_path.replace(path)
    except Exception as e:
        logger.error(f"Failed to write atomic file {path}: {e}")
        if temp_path.exists(): 
            temp_path.unlink()
        raise

def extract_sql_url(sql: str) -> Optional[str]:
    """Extract HTTP/HTTPS URL from SQL FROM clause.
    
    Handles various formatting:
    - FROM 'url'
    - FROM  'url' (multiple spaces)
    - FROM\\n'url' (newlines)
    - FROM\\n        'url' (newlines with indentation)
    
    Args:
        sql: SQL query string
        
    Returns:
        URL if found, None otherwise
    """
    # Match FROM followed by optional whitespace (including newlines) and a quote
    match = re.search(r"FROM\s+'([^']+)'", sql, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1)
    return None
