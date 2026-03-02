"""Polars backend implementation for efficient and larger-than-memory datasets."""

from typing import Any, List, Optional, Tuple, Union

import numpy as np

try:
    import polars as pl
    import pandas as pd
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False
    pl = None
    pd = None

from .base import DataBackend


class PolarsBackend(DataBackend):
    """Backend implementation for Polars DataFrames.
    
    This backend enables:
    - Faster data processing compared to pandas
    - Better memory efficiency
    - Support for larger-than-memory datasets through lazy evaluation
    - Multi-threaded operations
    
    Polars supports both eager and lazy DataFrames:
    - pl.DataFrame: Eager evaluation (in-memory)
    - pl.LazyFrame: Lazy evaluation (optimized query execution, can handle larger-than-memory data)
    """
    
    def __init__(self):
        """Initialize Polars backend."""
        if not POLARS_AVAILABLE:
            raise ImportError(
                "Polars is not installed. Install it with: pip install polars"
            )
    
    @property
    def name(self) -> str:
        """Return the name of the backend."""
        return "polars"
    
    def supports_lazy_loading(self) -> bool:
        """Polars supports lazy loading through LazyFrames."""
        return True
    
    def get_shape(self, df: Union[pl.DataFrame, pl.LazyFrame]) -> Tuple[int, int]:
        """Get the shape of the dataframe."""
        if isinstance(df, pl.LazyFrame):
            # For LazyFrame, we need to collect first or use a different approach
            # to get row count. Here we collect, but for very large datasets,
            # you might want to use SQL-style count operations
            collected = df.select(pl.count()).collect()
            n_rows = collected.item()
            n_cols = len(df.columns)
        else:
            n_rows, n_cols = df.shape
        return n_rows, n_cols
    
    def get_columns(self, df: Union[pl.DataFrame, pl.LazyFrame]) -> List[str]:
        """Get column names from the dataframe."""
        return df.columns
    
    def select_columns(
        self, 
        df: Union[pl.DataFrame, pl.LazyFrame], 
        columns: List[str]
    ) -> Union[pl.DataFrame, pl.LazyFrame]:
        """Select specific columns from the dataframe."""
        return df.select(columns)
    
    def to_numpy(
        self,
        df: Union[pl.DataFrame, pl.LazyFrame],
        columns: Optional[List[str]] = None,
        dtype: Optional[type] = None
    ) -> np.ndarray:
        """Convert dataframe (or specific columns) to numpy array."""
        # If LazyFrame, collect it first
        if isinstance(df, pl.LazyFrame):
            df = df.collect()
        
        if columns is not None:
            data = df.select(columns)
        else:
            data = df
        
        result = data.to_numpy()
        if dtype is not None:
            result = result.astype(dtype)
        return result
    
    def copy(self, df: Union[pl.DataFrame, pl.LazyFrame]) -> Union[pl.DataFrame, pl.LazyFrame]:
        """Create a copy of the dataframe."""
        return df.clone()
    
    def get_index(self, df: Union[pl.DataFrame, pl.LazyFrame]) -> np.ndarray:
        """Get the index of the dataframe.
        
        Note: Polars doesn't have a dedicated index like pandas.
        We return a range index.
        """
        n_rows, _ = self.get_shape(df)
        return np.arange(n_rows)
    
    def set_column(
        self,
        df: Union[pl.DataFrame, pl.LazyFrame],
        column_name: str,
        values: Union[np.ndarray, Any]
    ) -> Union[pl.DataFrame, pl.LazyFrame]:
        """Set values for a specific column in the dataframe."""
        if isinstance(df, pl.LazyFrame):
            # For LazyFrame, collect, modify, and convert back
            df = df.collect()
            df = df.with_columns(pl.Series(column_name, values))
            return df.lazy()
        else:
            return df.with_columns(pl.Series(column_name, values))
    
    def drop_columns(
        self,
        df: Union[pl.DataFrame, pl.LazyFrame],
        columns: List[str],
        inplace: bool = False
    ) -> Union[pl.DataFrame, pl.LazyFrame]:
        """Drop columns from the dataframe."""
        # Polars operations are not in-place by default
        result = df.drop(columns)
        if inplace:
            # Note: Polars doesn't truly support inplace operations
            # We return None to match pandas behavior
            return None
        return result
    
    def get_unique_values(self, df: Union[pl.DataFrame, pl.LazyFrame], column: str) -> np.ndarray:
        """Get unique values for a column."""
        if isinstance(df, pl.LazyFrame):
            df = df.collect()
        return df[column].unique().to_numpy()
    
    def to_pandas(self, df: Union[pl.DataFrame, pl.LazyFrame]) -> pd.DataFrame:
        """Convert the dataframe to pandas DataFrame."""
        if isinstance(df, pl.LazyFrame):
            df = df.collect()
        return df.to_pandas()
    
    def is_null(self, df: Union[pl.DataFrame, pl.LazyFrame], column: str) -> np.ndarray:
        """Check for null values in a column."""
        if isinstance(df, pl.LazyFrame):
            df = df.collect()
        return df[column].is_null().to_numpy()
    
    def apply_transform(
        self,
        df: Union[pl.DataFrame, pl.LazyFrame],
        columns: List[str],
        transform_func: Any
    ) -> Union[pl.DataFrame, pl.LazyFrame]:
        """Apply a transformation function to specific columns.
        
        Note: For sklearn transformers, we need to convert to pandas temporarily.
        """
        was_lazy = isinstance(df, pl.LazyFrame)
        
        if was_lazy:
            df = df.collect()
        
        # Convert to pandas for transformation
        pandas_df = df.to_pandas()
        
        # Apply transformation
        if hasattr(transform_func, 'transform'):
            pandas_df[columns] = transform_func.transform(pandas_df[columns])
        elif callable(transform_func):
            pandas_df[columns] = transform_func(pandas_df[columns])
        else:
            raise ValueError("transform_func must be a sklearn transformer or callable")
        
        # Convert back to polars
        result = pl.from_pandas(pandas_df)
        
        if was_lazy:
            return result.lazy()
        return result
    
    def concat_dataframes(
        self,
        dfs: List[Union[pl.DataFrame, pl.LazyFrame]],
        axis: int = 0
    ) -> Union[pl.DataFrame, pl.LazyFrame]:
        """Concatenate multiple dataframes."""
        # Check if any are LazyFrames
        any_lazy = any(isinstance(df, pl.LazyFrame) for df in dfs)
        
        if any_lazy:
            # Convert all to LazyFrame
            lazy_dfs = [df if isinstance(df, pl.LazyFrame) else df.lazy() for df in dfs]
            if axis == 0:
                return pl.concat(lazy_dfs)
            else:
                # Horizontal concatenation
                return pl.concat(lazy_dfs, how="horizontal")
        else:
            if axis == 0:
                return pl.concat(dfs)
            else:
                return pl.concat(dfs, how="horizontal")
    
    def sample_rows(
        self,
        df: Union[pl.DataFrame, pl.LazyFrame],
        n: int,
        random_state: Optional[int] = None
    ) -> Union[pl.DataFrame, pl.LazyFrame]:
        """Sample n rows from the dataframe."""
        n_rows, _ = self.get_shape(df)
        n = min(n, n_rows)
        
        if isinstance(df, pl.LazyFrame):
            # For LazyFrame, sample after collecting
            df_collected = df.collect()
            sampled = df_collected.sample(n=n, seed=random_state)
            return sampled.lazy()
        else:
            return df.sample(n=n, seed=random_state)
    
    def head(self, df: Union[pl.DataFrame, pl.LazyFrame], n: int = 5) -> Union[pl.DataFrame, pl.LazyFrame]:
        """Get the first n rows of the dataframe."""
        return df.head(n)
