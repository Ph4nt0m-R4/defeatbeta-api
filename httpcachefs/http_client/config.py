from dataclasses import dataclass, field
from typing import Dict

@dataclass(frozen=True)
class HTTPClientConfig:
    """
    Configuration for the HTTP client.
    
    Attributes:
        connect_timeout: Connection timeout in seconds (default: 10.0)
        read_timeout: Read timeout in seconds (default: 60.0)
        retries: Number of retries for failed requests (default: 3)
        backoff_factor: Exponential backoff factor for retries (default: 0.5)
        pool_connections: Number of connection pools (default: 10)
        pool_maxsize: Maximum size of each pool (default: 10)
        headers: Custom HTTP headers to include in requests
    """
    connect_timeout: float = 10.0
    read_timeout: float = 60.0
    retries: int = 3
    backoff_factor: float = 0.5
    pool_connections: int = 10
    pool_maxsize: int = 10
    headers: Dict[str, str] = field(default_factory=lambda: {
        'User-Agent': 'HttpCacheFS-RangeClient/1.0',
        'Accept-Encoding': 'identity',
        'Connection': 'keep-alive'
    })
