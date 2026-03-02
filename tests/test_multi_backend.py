"""Tests for data backends and multi-backend support."""

import numpy as np
import pytest


class TestPandasBackend:
    """Test PandasBackend functionality."""
    
    def test_backend_basic_operations(self):
        """Test basic backend operations with pandas."""
        import pandas as pd
        from pytorch_tabular.data_backends import PandasBackend, get_backend
        
        # Create test data
        df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": [10, 20, 30, 40, 50],
            "c": ["x", "y", "z", "x", "y"],
        })
        
        # Auto-detect backend
        backend = get_backend(df)
        assert backend.name == "pandas"
        assert isinstance(backend, PandasBackend)
        
        # Test shape
        shape = backend.get_shape(df)
        assert shape == (5, 3)
        
        # Test columns
        cols = backend.get_columns(df)
        assert cols == ["a", "b", "c"]
        
        # Test select columns
        selected = backend.select_columns(df, ["a", "b"])
        assert list(selected.columns) == ["a", "b"]
        
        # Test to_numpy
        arr = backend.to_numpy(df, columns=["a"])
        assert arr.shape == (5, 1)
        assert np.array_equal(arr.flatten(), [1, 2, 3, 4, 5])
        
        # Test copy
        copied = backend.copy(df)
        assert copied is not df
        assert copied.equals(df)
        
        # Test get_index
        idx = backend.get_index(df)
        assert len(idx) == 5
        
        # Test unique values
        uniques = backend.get_unique_values(df, "c")
        assert set(uniques) == {"x", "y", "z"}
        
        # Test head
        head = backend.head(df, n=3)
        assert len(head) == 3
        
    def test_backend_transforms(self):
        """Test transform operations with pandas backend."""
        import pandas as pd
        from sklearn.preprocessing import StandardScaler
        from pytorch_tabular.data_backends import PandasBackend
        
        df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": [10, 20, 30, 40, 50],
        })
        
        backend = PandasBackend()
        
        # Test apply_transform with sklearn transformer
        scaler = StandardScaler()
        scaler.fit(df[["a"]])
        
        transformed = backend.apply_transform(df.copy(), ["a"], scaler)
        assert "a" in transformed.columns
        assert not np.array_equal(transformed["a"].values, df["a"].values)
        
    def test_backend_capabilities(self):
        """Test backend capability flags."""
        from pytorch_tabular.data_backends import PandasBackend
        
        backend = PandasBackend()
        assert backend.supports_lazy_loading() is False


class TestPolarsBackend:
    """Test PolarsBackend functionality."""
    
    def test_polars_backend_eager(self):
        """Test Polars backend with eager DataFrame."""
        pytest.importorskip("polars")
        
        import polars as pl
        from pytorch_tabular.data_backends import PolarsBackend, get_backend
        
        # Create test data
        df = pl.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": [10, 20, 30, 40, 50],
            "c": ["x", "y", "z", "x", "y"],
        })
        
        # Auto-detect backend
        backend = get_backend(df)
        assert backend.name == "polars"
        assert isinstance(backend, PolarsBackend)
        
        # Test shape
        shape = backend.get_shape(df)
        assert shape == (5, 3)
        
        # Test columns
        cols = backend.get_columns(df)
        assert cols == ["a", "b", "c"]
        
        # Test to_numpy
        arr = backend.to_numpy(df, columns=["a"])
        assert arr.shape == (5, 1)
        assert np.array_equal(arr.flatten(), [1, 2, 3, 4, 5])
        
        # Test copy
        copied = backend.copy(df)
        assert copied is not df
        
        # Test unique values
        uniques = backend.get_unique_values(df, "c")
        assert set(uniques) == {"x", "y", "z"}
        
    def test_polars_backend_lazy(self):
        """Test Polars backend with LazyFrame."""
        pytest.importorskip("polars")
        
        import polars as pl
        from pytorch_tabular.data_backends import PolarsBackend
        
        # Create lazy dataframe
        df = pl.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": [10, 20, 30, 40, 50],
        }).lazy()
        
        backend = PolarsBackend()
        
        # Test columns (works on lazy)
        cols = backend.get_columns(df)
        assert cols == ["a", "b"]
        
        # Test head (works on lazy)
        head = backend.head(df, n=3)
        assert isinstance(head, pl.LazyFrame)
        
    def test_polars_backend_capabilities(self):
        """Test Polars backend capability flags."""
        pytest.importorskip("polars")
        
        from pytorch_tabular.data_backends import PolarsBackend
        
        backend = PolarsBackend()
        assert backend.supports_lazy_loading() is True


class TestTabularDatamoduleV2:
    """Test TabularDatamoduleV2 with multiple backends."""
    
    def test_datamodule_v2_pandas(self):
        """Test TabularDatamoduleV2 with pandas DataFrame."""
        import pandas as pd
        from pytorch_tabular import TabularDatamoduleV2
        from pytorch_tabular.config import DataConfig
        from omegaconf import OmegaConf
        
        # Create test data
        train_df = pd.DataFrame({
            "cont1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "cont2": [10.0, 20.0, 30.0, 40.0, 50.0],
            "cat1": ["a", "b", "a", "b", "a"],
            "target": [0, 1, 0, 1, 0],
        })
        
        test_df = train_df.copy()
        
        # Create config
        data_config = {
            "target": ["target"],
            "continuous_cols": ["cont1", "cont2"],
            "categorical_cols": ["cat1"],
            "batch_size": 2,
            "task": "classification",
        }
        config = OmegaConf.create(data_config)
        
        # Create datamodule
        datamodule = TabularDatamoduleV2(
            train=train_df,
            validation=test_df,
            config=config,
            verbose=False,
        )
        
        assert datamodule.backend.name == "pandas"
        assert datamodule.supports_streaming is False
        
    def test_datamodule_v2_polars_eager(self):
        """Test TabularDatamoduleV2 with Polars DataFrame."""
        pytest.importorskip("polars")
        
        import polars as pl
        from pytorch_tabular import TabularDatamoduleV2
        from omegaconf import OmegaConf
        
        # Create test data
        train_df = pl.DataFrame({
            "cont1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "cont2": [10.0, 20.0, 30.0, 40.0, 50.0],
            "cat1": ["a", "b", "a", "b", "a"],
            "target": [0, 1, 0, 1, 0],
        })
        
        test_df = train_df.clone()
        
        # Create config
        data_config = {
            "target": ["target"],
            "continuous_cols": ["cont1", "cont2"],
            "categorical_cols": ["cat1"],
            "batch_size": 2,
            "task": "classification",
        }
        config = OmegaConf.create(data_config)
        
        # Create datamodule
        datamodule = TabularDatamoduleV2(
            train=train_df,
            validation=test_df,
            config=config,
            verbose=False,
        )
        
        assert datamodule.backend.name == "polars"
        assert datamodule.supports_streaming is True
        
    def test_sampling_for_transform_fit(self):
        """Test sampling utility for large datasets."""
        pytest.importorskip("polars")
        
        import polars as pl
        from pytorch_tabular import TabularDatamoduleV2
        from omegaconf import OmegaConf
        
        # Create larger test data
        n = 1000
        train_df = pl.DataFrame({
            "cont1": list(range(n)),
            "cont2": list(range(n, 2*n)),
            "target": [0, 1] * (n // 2),
        })
        
        config = OmegaConf.create({
            "target": ["target"],
            "continuous_cols": ["cont1", "cont2"],
            "categorical_cols": [],
            "batch_size": 32,
            "task": "classification",
        })
        
        datamodule = TabularDatamoduleV2(
            train=train_df,
            config=config,
            verbose=False,
        )
        
        # Test sampling
        sample = datamodule.sample_for_transform_fit(train_df, sample_size=100)
        assert isinstance(sample, pl.DataFrame)
        assert len(sample) == 100
