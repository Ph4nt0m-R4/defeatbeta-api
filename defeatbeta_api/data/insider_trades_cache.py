"""
File-based cache for parsed SEC Form 4 insider-trading transactions.

The cache lives alongside the httpcachefs data under::

    httpcachefs/cache/<version>/insider_trades/<TICKER>/
        data.parquet        # Cached transactions DataFrame
        metadata.json       # { "accessions_parsed": [...], "cached_at": "..." }

On repeat calls the cache identifies which Form 4 accession numbers have
already been parsed and fetches only the new ones from SEC, then merges.
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional, Set

import pandas as pd

from defeatbeta_api.utils.util import validate_httpfs_cache_directory

logger = logging.getLogger(__name__)


class InsiderTradesCache:
    """Per-ticker disk cache for parsed SEC Form 4 transactions.

    Parameters
    ----------
    ticker : str
        Stock symbol (upper-cased).
    cache_ttl : int
        Seconds before the cache is considered stale and a full re-parse
        is forced.  ``0`` disables TTL expiry (cache is invalidated only
        when new accession numbers appear).  Default is ``86400`` (24 h).
    """

    def __init__(self, ticker: str, cache_ttl: int = 86_400):
        self.ticker = ticker.upper()
        self.cache_ttl = cache_ttl

        # Build cache directory inside the existing httpcachefs tree
        root = Path(validate_httpfs_cache_directory())
        self.cache_dir = root / "insider_trades" / self.ticker
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.data_path = self.cache_dir / "data.parquet"
        self.meta_path = self.cache_dir / "metadata.json"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_cached_accessions(self) -> Set[str]:
        """Return the set of accession numbers already parsed and cached."""
        meta = self._read_metadata()
        if meta is None:
            return set()
        return set(meta.get("accessions_parsed", []))

    def is_within_ttl(self) -> bool:
        """Return ``True`` if the cache was written within *cache_ttl* seconds."""
        meta = self._read_metadata()
        if meta is None:
            return False
        cached_at = meta.get("cached_at")
        if cached_at is None:
            return False
        try:
            age = time.time() - float(cached_at)
            return age < self.cache_ttl
        except (ValueError, TypeError):
            return False

    def load(self) -> Optional[pd.DataFrame]:
        """Load the cached DataFrame, or ``None`` if no cache exists."""
        if not self.data_path.exists():
            return None
        try:
            return pd.read_parquet(self.data_path)
        except Exception:
            logger.warning("Corrupted insider-trades cache for %s — ignoring", self.ticker)
            return None

    def save(self, df: pd.DataFrame, accessions_parsed: Set[str]) -> None:
        """Persist *df* and the set of parsed accession numbers to disk."""
        try:
            df.to_parquet(self.data_path, index=False)
            meta = {
                "accessions_parsed": sorted(accessions_parsed),
                "cached_at": str(time.time()),
            }
            self.meta_path.write_text(json.dumps(meta, indent=2))
        except Exception:
            logger.exception("Failed to write insider-trades cache for %s", self.ticker)

    def invalidate(self) -> None:
        """Remove all cached data for this ticker."""
        try:
            if self.data_path.exists():
                self.data_path.unlink()
            if self.meta_path.exists():
                self.meta_path.unlink()
        except Exception:
            logger.warning("Failed to invalidate insider-trades cache for %s", self.ticker)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _read_metadata(self) -> Optional[dict]:
        if not self.meta_path.exists():
            return None
        try:
            return json.loads(self.meta_path.read_text())
        except (json.JSONDecodeError, IOError):
            return None
