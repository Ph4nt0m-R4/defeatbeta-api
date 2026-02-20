"""
Custom exceptions for the HTTP Range Client.
"""

class HTTPRangeError(Exception):
    """Base exception for all HTTP range request errors."""
    pass

class ResourceNotFoundError(HTTPRangeError):
    """Raised when the remote resource cannot be found (404)."""
    pass

class RangeNotSupportedError(HTTPRangeError):
    """Raised when the server does not support byte range requests."""
    pass

class FileSizeError(HTTPRangeError):
    """Raised when file size cannot be determined."""
    pass
