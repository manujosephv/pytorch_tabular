"""Pandas backend implementation."""

from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .base import DataBackend


class PandasBackend(DataBackend):
    """Backend implementation for pandas DataFrames.
    
    This backend provides compatibility with the existing pandas-based
    TabularDatamodule implementation.
    """
    
    @property
    def name(self) -> str:
        """Return the name of the backend."""
        return "pandas"
    
    def supports_lazy_loading(self) -> bool:
        """Pandas does not support lazy loading."""
        return False
    
    def get_shape(self, df: pd.DataFrame) -> Tuple[int, int]:
        """Get the shape of the dataframe."""
        return df.shape
    
    def get_columns(self, df: pd.DataFrame) -> List[str]:
        """Get column names from the dataframe."""
        return df.columns.tolist()
    
    def select_columns(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Select specific columns from the dataframe."""
        return df[columns]
    
    def to_numpy(
        self, 
        df: pd.DataFrame, 
        columns: Optional[List[str]] = None, 
        dtype: Optional[type] = None
    ) -> np.ndarray:
        """Convert dataframe (or specific columns) to numpy array."""
        if columns is not None:
            data = df[columns]
        else:
            data = df
        
        if dtype is not None:
            return data.values.astype(dtype)
        return data.values
    
    def copy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create a copy of the dataframe."""
        return df.copy()
    
    def get_index(self, df: pd.DataFrame) -> pd.Index:
        """Get the index of the dataframe."""
        return df.index
    
    def set_column(
        self, 
        df: pd.DataFrame, 
        column_name: str, 
        values: Union[np.ndarray, Any]
    ) -> pd.DataFrame:
        """Set values for a specific column in the dataframe."""
        df[column_name] = values
        return df
    
    def drop_columns(
        self, 
        df: pd.DataFrame, 
        columns: List[str], 
        inplace: bool = False
    ) -> Optional[pd.DataFrame]:
        """Drop columns from the dataframe."""
        return df.drop(columns=columns, inplace=inplace)
    
    def get_unique_values(self, df: pd.DataFrame, column: str) -> np.ndarray:
        """Get unique values for a column."""
        return df[column].unique()
    
    def to_pandas(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert the dataframe to pandas DataFrame (already pandas)."""
        return df
    
    def is_null(self, df: pd.DataFrame, column: str) -> np.ndarray:
        """Check for null values in a column."""
        return df[column].isna().values
    
    def apply_transform(
        self, 
        df: pd.DataFrame, 
        columns: List[str], 
        transform_func: Any
    ) -> pd.DataFrame:
        """Apply a transformation function to specific columns."""
        # For sklearn transformers
        if hasattr(transform_func, 'transform'):
            df[columns] = transform_func.transform(df[columns])
        # For callable functions
        elif callable(transform_func):
            df[columns] = transform_func(df[columns])
        else:
            raise ValueError("transform_func must be a sklearn transformer or callable")
        return df
    
    def concat_dataframes(self, dfs: List[pd.DataFrame], axis: int = 0) -> pd.DataFrame:
        """Concatenate multiple dataframes."""
        return pd.concat(dfs, axis=axis)
    
    def sample_rows(
        self, 
        df: pd.DataFrame, 
        n: int, 
        random_state: Optional[int] = None
    ) -> pd.DataFrame:
        """Sample n rows from the dataframe."""
        return df.sample(n=min(n, len(df)), random_state=random_state)
    
    def head(self, df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
        """Get the first n rows of the dataframe."""
        return df.head(n)
