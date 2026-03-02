"""Enhanced TabularDatamodule with multi-backend support."""

from pathlib import Path
from typing import Any, Optional, Tuple, Union

import numpy as np
from omegaconf import DictConfig
from sklearn.base import TransformerMixin
import torch

from .tabular_datamodule import TabularDatamodule as BaseTabularDatamodule
from .data_backends import get_backend, DataBackend, PandasBackend

# Import for type hints
try:
    import pandas as pd
    import polars as pl
    DataFrame = Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame]
except ImportError:
    import pandas as pd
    DataFrame = pd.DataFrame


class TabularDatamoduleV2(BaseTabularDatamodule):
    """Enhanced TabularDatamodule with support for multiple data backends.
    
    This class extends the original TabularDatamodule to support:
    - Pandas DataFrames (backward compatible)
    - Polars DataFrames (faster, more memory efficient)
    - Polars LazyFrames (larger-than-memory datasets)
    
    The backend is automatically detected based on the input data type.
    
    Example:
        >>> import polars as pl
        >>> # Using Polars for better performance
        >>> train_df = pl.read_csv("large_file.csv")
        >>> datamodule = TabularDatamoduleV2(train=train_df, config=config)
        >>> 
        >>> # Using Polars LazyFrame for larger-than-memory data
        >>> train_lazy = pl.scan_csv("huge_file.csv")
        >>> datamodule = TabularDatamoduleV2(train=train_lazy, config=config)
    """
    
    def __init__(
        self,
        train: DataFrame,
        config: DictConfig,
        validation: DataFrame = None,
        target_transform: Optional[Union[TransformerMixin, Tuple]] = None,
        train_sampler: Optional[torch.utils.data.Sampler] = None,
        seed: Optional[int] = 42,
        cache_data: str = "memory",
        copy_data: bool = True,
        verbose: bool = True,
        backend: Optional[DataBackend] = None,
    ):
        """Initialize TabularDatamoduleV2 with multi-backend support.
        
        Args:
            train (DataFrame): Training data (pandas or polars DataFrame)
            config (DictConfig): Configuration object
            validation (DataFrame, optional): Validation data. Defaults to None.
            target_transform (Optional[Union[TransformerMixin, Tuple]], optional): 
                Target transformation. Defaults to None.
            train_sampler (Optional[torch.utils.data.Sampler], optional): 
                Custom sampler for training. Defaults to None.
            seed (Optional[int], optional): Random seed. Defaults to 42.
            cache_data (str, optional): Cache mode. Defaults to "memory".
            copy_data (bool, optional): Whether to copy data. Defaults to True.
            verbose (bool, optional): Verbosity. Defaults to True.
            backend (Optional[DataBackend], optional): 
                Explicitly specify backend. If None, auto-detects from data type. Defaults to None.
        """
        # Detect or use provided backend
        if backend is None:
            self.backend = get_backend(train)
        else:
            self.backend = backend
        
        if verbose:
            print(f"Using {self.backend.name} backend for data processing")
        
        # Convert to pandas for the base class if not already pandas
        # This maintains backward compatibility with the existing implementation
        if self.backend.name != "pandas":
            # For non-pandas backends, convert to pandas for now
            # In a full implementation, we'd refactor all operations to use backend
            self._original_train = train
            self._original_validation = validation
            train = self.backend.to_pandas(train)
            if validation is not None:
                validation = self.backend.to_pandas(validation)
        
        # Call parent constructor
        super().__init__(
            train=train,
            config=config,
            validation=validation,
            target_transform=target_transform,
            train_sampler=train_sampler,
            seed=seed,
            cache_data=cache_data,
            copy_data=copy_data,
            verbose=verbose,
        )
        
        # Store original data references for potential streaming operations
        if self.backend.name != "pandas":
            self.supports_streaming = self.backend.supports_lazy_loading()
        else:
            self.supports_streaming = False
    
    def sample_for_transform_fit(self, df: Any, sample_size: int = 10000) -> Any:
        """Sample data for fitting transformers when dealing with large datasets.
        
        This is particularly useful for larger-than-memory datasets where we
        need to fit transformers (like StandardScaler) on a representative sample.
        
        Args:
            df: Input dataframe
            sample_size: Number of rows to sample
            
        Returns:
            Sampled dataframe
        """
        n_rows, _ = self.backend.get_shape(df)
        if n_rows <= sample_size:
            return df
        return self.backend.sample_rows(df, n=sample_size, random_state=self.seed)


# Backward compatibility alias
TabularDatamoduleMultiBackend = TabularDatamoduleV2
