"""Data backend abstractions for supporting multiple dataframe libraries."""

from .base import DataBackend
from .pandas_backend import PandasBackend

try:
    from .polars_backend import PolarsBackend
    POLARS_AVAILABLE = True
except ImportError:
    PolarsBackend = None
    POLARS_AVAILABLE = False

__all__ = [
    "DataBackend",
    "PandasBackend",
    "PolarsBackend",
    "POLARS_AVAILABLE",
    "get_backend",
]


def get_backend(data) -> DataBackend:
    """Automatically detect and return the appropriate backend for the given data.
    
    Args:
        data: Input data (pandas DataFrame, polars DataFrame, etc.)
        
    Returns:
        DataBackend: The appropriate backend instance for the data type
        
    Raises:
        TypeError: If the data type is not supported
    """
    # Check for pandas DataFrame
    try:
        import pandas as pd
        if isinstance(data, pd.DataFrame):
            return PandasBackend()
    except ImportError:
        pass
    
    # Check for polars DataFrame
    if POLARS_AVAILABLE:
        try:
            import polars as pl
            if isinstance(data, pl.DataFrame):
                return PolarsBackend()
        except ImportError:
            pass
    
    raise TypeError(
        f"Unsupported data type: {type(data)}. "
        "Supported types are: pandas.DataFrame"
        + (", polars.DataFrame" if POLARS_AVAILABLE else "")
    )
