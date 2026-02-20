from io import IOBase, SEEK_SET, SEEK_CUR, SEEK_END, DEFAULT_BUFFER_SIZE
from typing import Optional
from .client import HTTPRangeClient
from .exceptions import HTTPRangeError

class RemoteFileObject(IOBase):
    """
    A file-like object interface for a remote URL using range requests.
    Supports read(), seek(), tell() and buffering.
    """

    def __init__(self, client: HTTPRangeClient, buffer_size: int = 8 * 1024 * 1024, close_client: bool = False, file_size: Optional[int] = None):
        self.client = client
        self.buffer_size = buffer_size
        self.close_client = close_client
        
        # Use provided file_size to avoid redundant HEAD request
        if file_size is not None:
            self._size = file_size
        else:
            self._size = client.get_file_size()
        self._pos = 0
        
        # Buffer
        self._buffer = b""
        self._buffer_start = -1
        self._buffer_end = -1

    def readable(self) -> bool: return True
    def seekable(self) -> bool: return True
    
    def __len__(self) -> int:
        """Return file size for compatibility with some libraries."""
        return self._size

    def seek(self, offset: int, whence: int = SEEK_SET) -> int:
        if whence == SEEK_SET:
            self._pos = offset
        elif whence == SEEK_CUR:
            self._pos += offset
        elif whence == SEEK_END:
            self._pos = self._size + offset
        else:
            raise ValueError(f"Invalid whence value: {whence}. Must be SEEK_SET, SEEK_CUR, or SEEK_END.")
        
        self._pos = max(0, min(self._pos, self._size))
        return self._pos

    def tell(self) -> int:
        return self._pos

    def read(self, size: int = -1) -> bytes:
        if size == -1:
            size = self._size - self._pos

        if size <= 0 or self._pos >= self._size:
            return b""

        target_end = min(self._pos + size, self._size)
        real_size = target_end - self._pos

        # Check Cache Hit
        if (self._pos >= self._buffer_start) and (target_end <= self._buffer_end):
            offset = self._pos - self._buffer_start
            data = self._buffer[offset : offset + real_size]
            self._pos += len(data)
            return data

        # Cache Miss
        fetch_size = max(real_size, self.buffer_size)
        fetch_end = min(self._pos + fetch_size, self._size)
        
        # Note: We subtract 1 because HTTP range is inclusive [start, end]
        # Validate range boundaries
        if self._pos < 0 or fetch_end - 1 >= self._size:
            raise ValueError(f"Invalid range: start={self._pos}, end={fetch_end - 1}, file_size={self._size}")
        
        try:
            self._buffer = self.client.read_range(self._pos, fetch_end - 1)
            self._buffer_start = self._pos
            self._buffer_end = self._buffer_start + len(self._buffer)
        except Exception as e:
            # If read fails, return empty to signal EOF rather than propagating error
            if self._buffer_start == -1:
                # First read failed - this is a real error
                raise HTTPRangeError(f"Failed to read from position {self._pos}: {str(e)}") from e
            return b""

        data = self._buffer[0 : real_size]
        self._pos += len(data)
        return data

    def close(self):
        if self.close_client:
            self.client.close()
        super().close()
