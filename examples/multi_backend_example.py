"""
Example: Using Multi-Backend Support for Larger-Than-Memory Datasets

This example demonstrates how to use PyTorch Tabular with Polars DataFrames
for better performance and support for larger-than-memory datasets.

Requirements:
    pip install pytorch_tabular[polars]

Author: arnavk23
Date: 2026
"""

import pytorch_tabular as pt
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
from pytorch_tabular.models import CategoryEmbeddingModelConfig

# ============================================================================
# Example 1: Using Polars DataFrame (Eager Mode)
# ============================================================================
# Polars is faster than pandas for many operations and uses less memory

try:
    import polars as pl
    
    print("Example 1: Using Polars DataFrame (Eager Mode)")
    print("=" * 60)
    
    # Read data with Polars - faster than pandas
    train_df = pl.read_csv("train.csv")
    test_df = pl.read_csv("test.csv")
    
    print(f"Using {type(train_df)} for training")
    print(f"Data shape: {train_df.shape}")
    
    # Configure the model
    data_config = DataConfig(
        target=["target"],
        continuous_cols=["col1", "col2", "col3"],
        categorical_cols=["cat1", "cat2"],
    )
    
    model_config = CategoryEmbeddingModelConfig(
        task="classification",
        layers="128-64-32",
    )
    
    trainer_config = TrainerConfig(
        max_epochs=10,
        batch_size=1024,
    )
    
    optimizer_config = OptimizerConfig()
    
    # Use TabularDatamoduleV2 for multi-backend support
    # The backend is automatically detected from the data type
    model = pt.TabularModel(
        data_config=data_config,
        model_config=model_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config,
    )
    
    # Use TabularDatamoduleV2 directly for more control
    from pytorch_tabular import TabularDatamoduleV2
    
    datamodule = TabularDatamoduleV2(
        train=train_df,
        validation=test_df,
        config=model.config,
        verbose=True,
    )
    
    print(f"Backend: {datamodule.backend.name}")
    print(f"Supports streaming: {datamodule.supports_streaming}")
    print()

except ImportError:
    print("Polars not installed. Install with: pip install polars")
    print()


# ============================================================================
# Example 2: Using Polars LazyFrame (Lazy Mode)
# ============================================================================
# LazyFrame enables working with larger-than-memory datasets through
# lazy evaluation and query optimization

try:
    import polars as pl
    
    print("Example 2: Using Polars LazyFrame (Lazy Mode)")
    print("=" * 60)
    
    # Scan CSV files without loading them into memory
    # This is crucial for very large files that don't fit in RAM
    train_lazy = pl.scan_csv("huge_train.csv")
    test_lazy = pl.scan_csv("huge_test.csv")
    
    print(f"Using {type(train_lazy)} for training")
    print("Data is not loaded into memory yet (lazy evaluation)")
    
    # Configure the model (same as before)
    data_config = DataConfig(
        target=["target"],
        continuous_cols=["col1", "col2", "col3"],
        categorical_cols=["cat1", "cat2"],
    )
    
    # For very large datasets, you might want to sample data for fitting
    # transformers (like StandardScaler) instead of using the entire dataset
    from pytorch_tabular import TabularDatamoduleV2
    
    datamodule = TabularDatamoduleV2(
        train=train_lazy,
        validation=test_lazy,
        config=data_config,
        verbose=True,
    )
    
    # The datamodule will handle the lazy evaluation and convert to pandas
    # for the actual training. In future versions, this will be fully lazy.
    print(f"Backend: {datamodule.backend.name}")
    print(f"Supports streaming: {datamodule.supports_streaming}")
    
    # Sample data for transformer fitting (useful for huge datasets)
    # This reduces memory usage when fitting scalers and encoders
    sample_size = 100000  # Use 100k rows for fitting transformers
    sampled_data = datamodule.sample_for_transform_fit(
        train_lazy,
        sample_size=sample_size
    )
    print(f"Sampled {sample_size} rows for transformer fitting")
    print()

except ImportError:
    print("Polars not installed. Install with: pip install polars")
    print()


# ============================================================================
# Example 3: Comparing Performance (Pandas vs Polars)
# ============================================================================

try:
    import polars as pl
    import pandas as pd
    import time
    
    print("Example 3: Performance Comparison")
    print("=" * 60)
    
    # Generate sample data
    n_rows = 1_000_000
    print(f"Generating {n_rows:,} rows of sample data...")
    
    # Pandas
    start = time.time()
    pandas_df = pd.DataFrame({
        "col1": range(n_rows),
        "col2": range(n_rows, 2 * n_rows),
        "cat1": ["A", "B", "C"] * (n_rows // 3 + 1),
        "target": [0, 1] * (n_rows // 2),
    })[:n_rows]
    pandas_time = time.time() - start
    print(f"Pandas creation time: {pandas_time:.2f}s")
    
    # Polars
    start = time.time()
    polars_df = pl.DataFrame({
        "col1": range(n_rows),
        "col2": range(n_rows, 2 * n_rows),
        "cat1": ["A", "B", "C"] * (n_rows // 3 + 1),
        "target": [0, 1] * (n_rows // 2),
    })[:n_rows]
    polars_time = time.time() - start
    print(f"Polars creation time: {polars_time:.2f}s")
    print(f"Speedup: {pandas_time / polars_time:.2f}x faster")
    
    # Memory usage
    print(f"\nPandas memory usage: {pandas_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"Polars estimated memory: {polars_df.estimated_size() / 1024**2:.2f} MB")
    print()

except ImportError as e:
    print(f"Required packages not installed: {e}")
    print()


# ============================================================================
# Example 4: Using Backend Manually
# ============================================================================

print("Example 4: Manual Backend Selection")
print("=" * 60)

try:
    import polars as pl
    from pytorch_tabular.data_backends import get_backend, PandasBackend, PolarsBackend
    
    # Create sample data
    polars_df = pl.DataFrame({
        "a": [1, 2, 3],
        "b": [4, 5, 6],
    })
    
    # Auto-detect backend
    backend = get_backend(polars_df)
    print(f"Auto-detected backend: {backend.name}")
    
    # Use backend operations
    shape = backend.get_shape(polars_df)
    print(f"Shape: {shape}")
    
    cols = backend.get_columns(polars_df)
    print(f"Columns: {cols}")
    
    # Convert to numpy
    numpy_array = backend.to_numpy(polars_df, columns=["a"])
    print(f"Numpy array shape: {numpy_array.shape}")
    
    # Check capabilities
    print(f"Supports lazy loading: {backend.supports_lazy_loading()}")
    print()
    
except ImportError:
    print("Polars not installed. Install with: pip install polars")
    print()


# ============================================================================
# Best Practices and Tips
# ============================================================================

print("Best Practices for Larger-Than-Memory Datasets")
print("=" * 60)
print("""
1. Use Polars LazyFrame for datasets that don't fit in memory:
   train_lazy = pl.scan_csv("huge_file.csv")

2. Sample data for fitting transformers to reduce memory usage:
   datamodule.sample_for_transform_fit(df, sample_size=100000)

3. Use appropriate batch sizes:
   - Larger batch sizes for better GPU utilization
   - Smaller batch sizes if running out of memory

4. Consider using streaming for very large datasets:
   - Process data in chunks
   - Use lazy evaluation when possible

5. Monitor memory usage and adjust accordingly:
   - Use Polars for better memory efficiency
   - Enable disk caching for intermediate results

6. For extremely large datasets (>1TB), consider:
   - Using distributed computing frameworks (Spark)
   - Processing data in batches
   - Using a data warehouse or database backend
""")
