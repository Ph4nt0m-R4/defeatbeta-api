from httpcachefs.http_client import RemoteFileObject

class FusedFileObject(RemoteFileObject):
    """
    Hybrid file object:
    - Reads < footer_start served from Memory (Zero Latency)
    - Reads >= footer_start served from Network (Lazy Loading)
    Supports context manager protocol for automatic resource cleanup.
    """
    def __init__(self, client, footer_data: bytes, file_size: int, buffer_size: int, close_client: bool = False):
        if footer_data is None:
            raise ValueError("footer_data cannot be None")
        if not isinstance(footer_data, bytes):
            raise TypeError(f"footer_data must be bytes, got {type(footer_data).__name__}")
        if file_size < 0:
            raise ValueError(f"file_size must be non-negative, got {file_size}")
        if buffer_size <= 0:
            raise ValueError(f"buffer_size must be positive, got {buffer_size}")
        
        super().__init__(client, buffer_size, close_client=close_client, file_size=file_size)
        self.footer_data = footer_data
        self.footer_start = self._size - len(footer_data)
        
        # Validate footer consistency
        if self.footer_start < 0:
            raise ValueError(f"Invalid footer: footer_start={self.footer_start}, file_size={self._size}, footer_len={len(footer_data)}")

    def read(self, size: int = -1) -> bytes:
        # Check if read is fully within footer region (served from memory)
        if self._pos >= self.footer_start:
            offset = self._pos - self.footer_start
            if size == -1:
                data = self.footer_data[offset:]
            else:
                end_idx = min(offset + size, len(self.footer_data))
                data = self.footer_data[offset:end_idx]
            self._pos += len(data)
            return data
        
        # Otherwise, fallback to network request via parent class
        return super().read(size)
    
    def get_file_size(self) -> int:
        """Return the file size."""
        return self._size
