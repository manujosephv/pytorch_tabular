"""Abstract base class for data backends."""

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple, Union

import numpy as np


class DataBackend(ABC):
    """Abstract base class for data backends supporting different dataframe libraries.
    
    This class defines the interface that all data backends must implement to support
    various dataframe libraries (pandas, polars, spark, etc.) in PyTorch Tabular.
    """
    
    @abstractmethod
    def get_shape(self, df: Any) -> Tuple[int, int]:
        """Get the shape of the dataframe.
        
        Args:
            df: Input dataframe
            
        Returns:
            Tuple[int, int]: (n_rows, n_columns)
        """
        pass
    
    @abstractmethod
    def get_columns(self, df: Any) -> List[str]:
        """Get column names from the dataframe.
        
        Args:
            df: Input dataframe
            
        Returns:
            List[str]: List of column names
        """
        pass
    
    @abstractmethod
    def select_columns(self, df: Any, columns: List[str]) -> Any:
        """Select specific columns from the dataframe.
        
        Args:
            df: Input dataframe
            columns: List of column names to select
            
        Returns:
            Dataframe with selected columns
        """
        pass
    
    @abstractmethod
    def to_numpy(self, df: Any, columns: Optional[List[str]] = None, dtype: Optional[type] = None) -> np.ndarray:
        """Convert dataframe (or specific columns) to numpy array.
        
        Args:
            df: Input dataframe
            columns: Optional list of columns to convert. If None, converts all columns.
            dtype: Optional numpy dtype for the output array
            
        Returns:
            np.ndarray: Numpy array representation of the data
        """
        pass
    
    @abstractmethod
    def copy(self, df: Any) -> Any:
        """Create a copy of the dataframe.
        
        Args:
            df: Input dataframe
            
        Returns:
            Copied dataframe
        """
        pass
    
    @abstractmethod
    def get_index(self, df: Any) -> Any:
        """Get the index of the dataframe.
        
        Args:
            df: Input dataframe
            
        Returns:
            Index of the dataframe
        """
        pass
    
    @abstractmethod
    def set_column(self, df: Any, column_name: str, values: Union[np.ndarray, Any]) -> Any:
        """Set values for a specific column in the dataframe.
        
        Args:
            df: Input dataframe
            column_name: Name of the column to set
            values: Values to set (can be numpy array or native type)
            
        Returns:
            Dataframe with updated column
        """
        pass
    
    @abstractmethod
    def drop_columns(self, df: Any, columns: List[str], inplace: bool = False) -> Any:
        """Drop columns from the dataframe.
        
        Args:
            df: Input dataframe
            columns: List of column names to drop
            inplace: Whether to modify the dataframe in place
            
        Returns:
            Dataframe with columns dropped (or None if inplace=True)
        """
        pass
    
    @abstractmethod
    def get_unique_values(self, df: Any, column: str) -> np.ndarray:
        """Get unique values for a column.
        
        Args:
            df: Input dataframe
            column: Column name
            
        Returns:
            np.ndarray: Array of unique values
        """
        pass
    
    @abstractmethod
    def to_pandas(self, df: Any) -> Any:
        """Convert the dataframe to pandas DataFrame.
        
        Args:
            df: Input dataframe
            
        Returns:
            pandas.DataFrame: Pandas representation of the data
        """
        pass
    
    @abstractmethod
    def is_null(self, df: Any, column: str) -> np.ndarray:
        """Check for null values in a column.
        
        Args:
            df: Input dataframe
            column: Column name
            
        Returns:
            np.ndarray: Boolean array indicating null values
        """
        pass
    
    @abstractmethod
    def apply_transform(self, df: Any, columns: List[str], transform_func: Any) -> Any:
        """Apply a transformation function to specific columns.
        
        Args:
            df: Input dataframe
            columns: List of column names to transform
            transform_func: sklearn transformer or callable
            
        Returns:
            Dataframe with transformed columns
        """
        pass
    
    @abstractmethod
    def concat_dataframes(self, dfs: List[Any], axis: int = 0) -> Any:
        """Concatenate multiple dataframes.
        
        Args:
            dfs: List of dataframes to concatenate
            axis: Axis along which to concatenate (0=rows, 1=columns)
            
        Returns:
            Concatenated dataframe
        """
        pass
    
    @abstractmethod
    def sample_rows(self, df: Any, n: int, random_state: Optional[int] = None) -> Any:
        """Sample n rows from the dataframe.
        
        Args:
            df: Input dataframe
            n: Number of rows to sample
            random_state: Random seed for reproducibility
            
        Returns:
            Sampled dataframe
        """
        pass
    
    @abstractmethod
    def head(self, df: Any, n: int = 5) -> Any:
        """Get the first n rows of the dataframe.
        
        Args:
            df: Input dataframe
            n: Number of rows to return
            
        Returns:
            Dataframe with first n rows
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the backend.
        
        Returns:
            str: Backend name (e.g., 'pandas', 'polars', 'spark')
        """
        pass
    
    @abstractmethod
    def supports_lazy_loading(self) -> bool:
        """Check if the backend supports lazy loading / out-of-core computation.
        
        Returns:
            bool: True if backend supports lazy loading
        """
        pass
