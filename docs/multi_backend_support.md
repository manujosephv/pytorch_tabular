# Multi-Backend Support for Larger-Than-Memory Datasets

This document describes the new multi-backend data support feature that enables PyTorch Tabular to work with larger-than-memory datasets using Polars and other data frameworks.

## Overview

PyTorch Tabular now supports multiple data backends beyond pandas:
- **Pandas** (default, backward compatible)
- **Polars** (faster, more memory efficient)
- **Polars LazyFrame** (lazy evaluation for larger-than-memory datasets)

This addresses [Issue #402](https://github.com/pytorch-tabular/pytorch_tabular/issues/402) by providing the architectural foundation for supporting various dataframe libraries and enabling work with datasets larger than available RAM.

## Installation

Install with Polars support:
```bash
pip install pytorch_tabular[polars]
```

Or install Polars separately:
```bash
pip install polars>=0.20.0
```

## Quick Start

### Using Polars DataFrame (Eager Mode)

```python
import polars as pl
from pytorch_tabular import TabularDatamoduleV2
from pytorch_tabular.config import DataConfig

# Read data with Polars - faster than pandas
train_df = pl.read_csv("train.csv")
test_df = pl.read_csv("test.csv")

# Create data config
data_config = DataConfig(
    target=["target"],
    continuous_cols=["col1", "col2"],
    categorical_cols=["cat1", "cat2"],
)

# Create datamodule - backend is automatically detected
datamodule = TabularDatamoduleV2(
    train=train_df,
    validation=test_df,
    config=data_config,
)

print(f"Backend: {datamodule.backend.name}")  # Output: polars
```

### Using Polars LazyFrame (Lazy Mode for Large Datasets)

```python
import polars as pl
from pytorch_tabular import TabularDatamoduleV2

# Scan CSV without loading into memory
train_lazy = pl.scan_csv("huge_file.csv")
test_lazy = pl.scan_csv("test.csv")

# LazyFrame enables working with larger-than-memory data
datamodule = TabularDatamoduleV2(
    train=train_lazy,
    validation=test_lazy,
    config=data_config,
)

# Sample for transformer fitting (reduces memory usage)
sample = datamodule.sample_for_transform_fit(
    train_lazy,
    sample_size=100000  # Use 100k rows to fit scalers/encoders
)
```

## Architecture

### Backend Abstraction

The new architecture introduces a `DataBackend` abstract base class that defines a common interface for all dataframe libraries:

```python
from pytorch_tabular.data_backends import DataBackend, get_backend

# Auto-detect backend from data
backend = get_backend(my_dataframe)

# Use backend operations
shape = backend.get_shape(my_dataframe)
columns = backend.get_columns(my_dataframe)
numpy_array = backend.to_numpy(my_dataframe)
```

### Available Backends

1. **PandasBackend** - For pandas DataFrames
   - Fully backward compatible
   - No lazy loading support
   - Best for datasets that fit in memory

2. **PolarsBackend** - For Polars DataFrames/LazyFrames
   - Faster than pandas
   - Better memory efficiency
   - Supports lazy evaluation
   - Multi-threaded operations

### Key Components

```
src/pytorch_tabular/
├── data_backends/
│   ├── __init__.py          # Backend selection and imports
│   ├── base.py              # Abstract DataBackend interface
│   ├── pandas_backend.py    # Pandas implementation
│   └── polars_backend.py    # Polars implementation
├── tabular_datamodule.py    # Original datamodule (pandas)
└── tabular_datamodule_v2.py # Enhanced with multi-backend support
```

## Performance Benefits

### Memory Efficiency

Polars uses Apache Arrow memory format which is more memory efficient than pandas:
- **Contiguous memory layout** - Better cache locality
- **Column-oriented storage** - Efficient for analytical operations
- **Zero-copy reads** - Share memory with Arrow-compatible libraries

### Speed Improvements

Typical speedups observed with Polars:
- **CSV reading**: 2-5x faster than pandas
- **Filtering operations**: 3-10x faster
- **Aggregations**: 2-8x faster
- **String operations**: 5-15x faster

### Larger-Than-Memory Support

Using LazyFrame enables:
- **Query optimization** - Polars optimizes the query plan before execution
- **Streaming execution** - Process data in chunks
- **Predicate pushdown** - Filter data early to reduce memory usage
- **Column pruning** - Only load required columns

## Usage Patterns

### Pattern 1: Drop-in Replacement for Pandas

```python
import polars as pl

# Simply replace pandas DataFrame with Polars
train_df = pl.read_csv("train.csv")  # Instead of pd.read_csv

# Use TabularDatamoduleV2 (automatically detects backend)
from pytorch_tabular import TabularDatamoduleV2
datamodule = TabularDatamoduleV2(train=train_df, config=config)
```

### Pattern 2: Lazy Loading for Large Files

```python
import polars as pl

# Scan instead of read - no memory loading yet
train = pl.scan_csv("10GB_file.csv")
test = pl.scan_csv("1GB_file.csv")

# Operations are recorded but not executed
filtered = train.filter(pl.col("age") > 18)

# Execution happens when needed (inside datamodule)
datamodule = TabularDatamoduleV2(train=filtered, ...)
```

### Pattern 3: Sampling for Transform Fitting

```python
from pytorch_tabular import TabularDatamoduleV2

# For very large datasets, fit transformers on a sample
datamodule = TabularDatamoduleV2(train=huge_lazy_df, config=config)

# Sample for fitting StandardScaler, LabelEncoder, etc.
sample = datamodule.sample_for_transform_fit(
    huge_lazy_df,
    sample_size=100_000
)
# Transformers are fit on sample, then applied to full data
```

### Pattern 4: Explicit Backend Selection

```python
from pytorch_tabular.data_backends import PolarsBackend

# Explicitly specify backend (advanced use)
datamodule = TabularDatamoduleV2(
    train=train_df,
    config=config,
    backend=PolarsBackend(),
)
```

## Backward Compatibility

- The original `TabularDatamodule` remains unchanged and fully functional
- Existing code using pandas DataFrames continues to work without modification
- `TabularDatamoduleV2` is opt-in and requires no changes to existing pipelines
- Backend detection is automatic - no code changes needed to use Polars

## Limitations and Future Work

### Current Limitations

1. **Conversion to Pandas**: Currently, non-pandas backends are converted to pandas internally for compatibility with existing code. This maintains functionality but doesn't fully leverage lazy evaluation.

2. **Transform Fitting**: sklearn transformers require in-memory data, so LazyFrames are collected during transform fitting. Use sampling to mitigate memory issues.

3. **Limited Spark Support**: Spark backend is planned but not yet implemented.

### Future Enhancements

1. **Native Backend Operations**: Refactor TabularDatamodule to use backend operations throughout, eliminating pandas conversion.

2. **Streaming DataLoaders**: Implement true streaming data loading for larger-than-memory datasets.

3. **Spark Backend**: Add support for PySpark DataFrames for distributed computing.

4. **Dask Backend**: Add support for Dask DataFrames for parallel computing.

5. **Custom Backend API**: Allow users to implement custom backends for proprietary data systems.

## Examples

See `examples/multi_backend_example.py` for comprehensive examples including:
- Basic Polars DataFrame usage
- LazyFrame for larger-than-memory data
- Performance comparisons
- Manual backend selection
- Best practices

## API Reference

### TabularDatamoduleV2

```python
class TabularDatamoduleV2(BaseTabularDatamodule):
    """Enhanced TabularDatamodule with multi-backend support."""
    
    def __init__(
        self,
        train: DataFrame,
        config: DictConfig,
        validation: DataFrame = None,
        backend: Optional[DataBackend] = None,
        **kwargs
    ):
        """
        Args:
            train: Training data (pandas, polars, or lazy)
            config: Configuration object
            validation: Validation data
            backend: Explicitly specify backend (auto-detected if None)
            **kwargs: Additional arguments passed to base class
        """
```

### DataBackend

```python
class DataBackend(ABC):
    """Abstract interface for data backends."""
    
    @abstractmethod
    def get_shape(self, df) -> Tuple[int, int]: ...
    
    @abstractmethod
    def to_numpy(self, df, columns=None, dtype=None) -> np.ndarray: ...
    
    @abstractmethod
    def supports_lazy_loading(self) -> bool: ...
```

### Backend Selection

```python
def get_backend(data) -> DataBackend:
    """Auto-detect and return appropriate backend for data."""
```

## Testing

Run tests for multi-backend support:
```bash
# Install test dependencies
pip install pytest polars

# Run backend tests
pytest tests/test_backends.py

# Run integration tests
pytest tests/test_datamodule_v2.py
```

## Contributing

This is an initial implementation addressing Issue #402. Contributions are welcome!

Areas for contribution:
1. Additional backend implementations (Spark, Dask, Ray)
2. Performance optimizations
3. Native lazy evaluation throughout the pipeline
4. Documentation and examples
5. Integration tests

See `CONTRIBUTING.md` for guidelines.

## References

- [Issue #402](https://github.com/pytorch-tabular/pytorch_tabular/issues/402) - Original feature request
- [Polars Documentation](https://pola-rs.github.io/polars/) - Polars user guide
- [Apache Arrow](https://arrow.apache.org/) - Arrow memory format

## License

This feature is part of PyTorch Tabular and is licensed under the MIT License.
