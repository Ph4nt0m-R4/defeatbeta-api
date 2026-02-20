from dataclasses import dataclass

@dataclass
class RowGroupMeta:
    """Metadata for a single Parquet Row Group.
    
    Stores statistics about a row group to enable pruning and optimization.
    The min_val and max_val are used to determine which row groups contain
    data matching filter predicates.
    
    Attributes:
        index: Index of the row group within the parquet file
        min_val: Minimum value of the partition column in this row group
        max_val: Maximum value of the partition column in this row group
        compressed_size: Total size of this row group in bytes (compressed)
    """
    index: int
    min_val: str
    max_val: str
    compressed_size: int
