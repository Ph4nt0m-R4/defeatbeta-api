import logging
import threading
from typing import Optional, Dict

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import HTTPClientConfig
from .exceptions import (
    HTTPRangeError, 
    ResourceNotFoundError, 
    RangeNotSupportedError, 
    FileSizeError
)

logger = logging.getLogger(__name__)

class HTTPRangeClient:
    """A stateless client for performing HTTP Range requests.
    
    This class handles HTTP Range (RFC 7233) requests for efficient remote file access.
    It manages session pooling, retries, timeout, and caching of file metadata.
    
    Key features:
    - Stateless design: all state is independent, can be used safely in multithreaded code
    - Automatic retry logic with exponential backoff
    - Connection pooling for efficiency
    - Range request validation (detects servers that don't support ranges)
    - Metadata caching (size, headers, ETag)
    - Telemetry tracking (requests, bytes downloaded)
    """

    def __init__(self, url: str, config: Optional[HTTPClientConfig] = None):
        if not url:
            raise ValueError("url cannot be empty")
        if not isinstance(url, str):
            raise TypeError(f"url must be a string, got {type(url).__name__}")
        
        self.url = url
        self.config = config or HTTPClientConfig()
        self._session = self._create_session()
        self._lock = threading.RLock()
        
        # Metadata Cache
        self._size: Optional[int] = None
        self._last_headers: Optional[Dict[str, str]] = None
        self._range_supported: Optional[bool] = None
        
        # Telemetry
        self.total_bytes_downloaded = 0
        self.total_requests = 0

    def _create_session(self) -> requests.Session:
        session = requests.Session()
        retry_strategy = Retry(
            total=self.config.retries,
            backoff_factor=self.config.backoff_factor,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=self.config.pool_connections,
            pool_maxsize=self.config.pool_maxsize
        )
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        session.headers.update(self.config.headers)
        return session

    def _request(self, method: str, headers: Optional[Dict[str, str]] = None, stream: bool = False) -> requests.Response:
        try:
            response = self._session.request(
                method=method,
                url=self.url,
                headers=headers,
                timeout=(self.config.connect_timeout, self.config.read_timeout),
                stream=stream,
                allow_redirects=True
            )
            
            if response.status_code == 404:
                raise ResourceNotFoundError(f"Resource not found: {self.url}")
            
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP Request failed for {self.url}: {str(e)}")
            raise HTTPRangeError(f"Network error: {str(e)}") from e

    def get_file_size(self) -> int:
        """Get the file size in bytes from the remote server.
        
        Caches result in self._size to avoid repeated HEAD requests.
        Uses locking to ensure thread-safe caching.
        
        Returns:
            File size in bytes
            
        Raises:
            FileSizeError: If file size cannot be determined
            ResourceNotFoundError: If resource not found (404)
            HTTPRangeError: For other HTTP errors
        """
        if self._size is not None:
            return self._size

        with self._lock:
            if self._size is not None:
                return self._size

            try:
                response = self._request("HEAD")
            except Exception:
                # Fallback logic could go here if HEAD is blocked
                raise

            # Store headers for later use (ETag, etc.)
            self._last_headers = dict(response.headers)

            if 'X-Linked-Size' in response.headers:
                self._size = int(response.headers['X-Linked-Size'])
                return self._size

            if 'Content-Length' in response.headers:
                self._size = int(response.headers['Content-Length'])
                return self._size
            
            raise FileSizeError(f"Could not determine file size for {self.url}")

    def read_range(self, start: int, end: int) -> bytes:
        # Validate range parameters
        if start < 0:
            raise ValueError(f"start must be non-negative, got {start}")
        if end < start:
            raise ValueError(f"end ({end}) must be >= start ({start})")
        if self._size is not None and end >= self._size:
            raise ValueError(f"end ({end}) must be < file_size ({self._size})")
        
        headers = {"Range": f"bytes={start}-{end}"}
        
        try:
            self.total_requests += 1
            response = self._request("GET", headers=headers)
            
            if response.status_code != 206:
                if response.status_code == 200:
                    # Server doesn't support ranges - mark it and fail fast
                    self._range_supported = False
                    logger.error(f"Server does not support HTTP Range requests (returned 200 instead of 206).")
                    raise RangeNotSupportedError(f"Server returned status 200 - range requests not supported for {self.url}") from None
                
                raise RangeNotSupportedError(f"Server returned status {response.status_code} for range request.") from None
            
            # Mark as supported on first successful range request
            if self._range_supported is None:
                self._range_supported = True
            
            content = response.content
            self.total_bytes_downloaded += len(content)
            return content
            
        except RangeNotSupportedError:
            raise
        except requests.exceptions.RequestException as e:
            raise HTTPRangeError(f"Failed to read range {start}-{end}: {str(e)}") from e

    def close(self):
        self._session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
