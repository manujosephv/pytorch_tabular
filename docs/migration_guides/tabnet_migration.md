# TabNet Migration Guide

As of 2024, the `pytorch-tabnet` package, which is a soft dependency for the `TabNetModel` in PyTorch Tabular, has lapsed maintenance (last updated in 2023). While the model remains functional in current environments, we recommend users transition to more modern and actively maintained architectures to ensure long-term stability and performance.

## Why Migrate?
- **Maintenance Status**: `pytorch-tabnet` is no longer receiving security patches or performance updates.
- **Compatibility**: Future updates to PyTorch or PyTorch Lightning may break `pytorch-tabnet` internals.
- **Superior Alternatives**: Architectures like **GANDALF** and **FT-Transformer** often provide equal or superior performance on tabular benchmarks with better stability.

## Recommended Alternatives

### 1. GANDALF (Gated Adaptive Network for Deep Automated Learning of Features)
GANDALF is highly efficient and often outperforms TabNet on modern benchmarks.
- **When to use**: High performance, minimal hyperparameter tuning.
- **Config**:
```python
from pytorch_tabular.models import GatedAdditiveTreeEnsembleConfig
model_config = GatedAdditiveTreeEnsembleConfig(...)
```

### 2. FT-Transformer
A robust adaptation of the Transformer architecture for tabular data.
- **When to use**: When you need strong contextual representations of features.
- **Config**:
```python
from pytorch_tabular.models import FTTransformerConfig
model_config = FTTransformerConfig(...)
```

## Migration Steps
1. Update your `ModelConfig` to use one of the recommended alternatives.
2. If you have a pre-trained TabNet model, you will need to retrain a new model with the new architecture, as weights are not transferable between different model classes.

For more information, please refer to our [Issue #611](https://github.com/manujosephv/pytorch_tabular/issues/611).
