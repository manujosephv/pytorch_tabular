# TabNet Dependency Status and Migration Guide

## Important Notice

⚠️ **The `pytorch-tabnet` dependency is no longer actively maintained.**

- **Last Update**: July 2023 (v4.1.0)
- **Repository**: https://github.com/dreamquark-ai/tabnet
- **Status**: No updates for 2.5+ years
- **Issue**: https://github.com/pytorch-tabular/pytorch_tabular/issues/611

While TabNet continues to work in PyTorch Tabular, users should be aware of potential compatibility issues with newer versions of PyTorch, Python, or other dependencies.

## Current Status in PyTorch Tabular

- TabNet is available as an **optional model** (requires `pytorch-tabnet<4.2`)
- Included in the `extra` optional dependencies: `pip install pytorch_tabular[extra]`
- Uses soft dependency checks with graceful fallback
- Will display a `FutureWarning` when instantiated

## Recommended Alternatives

For new projects or production use, consider these actively maintained alternatives:

### 1. **FTTransformer** (Feature Tokenizer Transformer)
- **Best For**: General tabular data, high performance
- **Advantages**: State-of-the-art results, attention mechanism, interpretable
- **Paper**: "Revisiting Deep Learning Models for Tabular Data" (2021)

```python
from pytorch_tabular.models import FTTransformerConfig

model_config = FTTransformerConfig(
    task="classification",
    num_heads=8,
    num_attn_blocks=6,
    learning_rate=1e-4
)
```

### 2. **GANDALF** (Gated Adaptive Network for Deep Automated Learning of Features)
- **Best For**: Mixed data types, automated feature learning
- **Advantages**: Adaptive gating, handles complex interactions
- **Paper**: "GANDALF: Gated Adaptive Network for Deep Automated Learning of Features" (2023)

```python
from pytorch_tabular.models import GANDALFConfig

model_config = GANDALFConfig(
    task="classification",
    gflu_stages=6,
    learning_rate=1e-3
)
```

### 3. **TabTransformer**
- **Best For**: Datasets with many categorical features
- **Advantages**: Column-wise attention, good for categorical-heavy data
- **Paper**: "TabTransformer: Tabular Data Modeling Using Contextual Embeddings" (2020)

```python
from pytorch_tabular.models import TabTransformerConfig

model_config = TabTransformerConfig(
    task="classification",
    num_heads=8,
    num_attn_blocks=6,
    learning_rate=1e-4
)
```

### 4. **GATE** (Gated Additive Tree Ensemble)
- **Best For**: Interpretability, combining neural networks with trees
- **Advantages**: Built-in feature importance, tree-like interpretability

```python
from pytorch_tabular.models import GATEConfig

model_config = GATEConfig(
    task="classification",
    gflu_stages=6,
    learning_rate=1e-3
)
```

### 5. **CategoryEmbeddingModel** or **AutoInt**
- **Best For**: Simple baseline, fast training
- **Advantages**: Well-tested, reliable, efficient

```python
from pytorch_tabular.models import CategoryEmbeddingModelConfig

model_config = CategoryEmbeddingModelConfig(
    task="classification",
    layers="1024-512-256",
    learning_rate=1e-3
)
```

## Migration Guide

### From TabNet to FTTransformer

TabNet and FTTransformer both use attention mechanisms, making FTTransformer a natural replacement:

**Before (TabNet):**
```python
from pytorch_tabular.models import TabNetModelConfig

model_config = TabNetModelConfig(
    task="classification",
    n_d=64,
    n_a=64,
    n_steps=5,
    gamma=1.5,
    learning_rate=0.02
)
```

**After (FTTransformer):**
```python
from pytorch_tabular.models import FTTransformerConfig

model_config = FTTransformerConfig(
    task="classification",
    num_heads=8,                    # Replaces attention mechanism
    num_attn_blocks=6,               # Comparable to n_steps
    d_model=256,                     # Comparable to n_d/n_a
    learning_rate=1e-4              # Usually needs lower LR
)
```

### From TabNet to GANDALF

GANDALF provides similar gating mechanisms:

**Before (TabNet):**
```python
model_config = TabNetModelConfig(
    task="classification",
    n_d=64,
    n_a=64,
    n_steps=5,
    learning_rate=0.02
)
```

**After (GANDALF):**
```python
from pytorch_tabular.models import GANDALFConfig

model_config = GANDALFConfig(
    task="classification",
    gflu_stages=6,                   # Comparable to n_steps
    gflu_feature_init_sparsity=0.3,  # Feature selection
    learning_rate=1e-3
)
```

## Comparison Table

| Feature | TabNet | FTTransformer | GANDALF | TabTransformer |
|---------|--------|---------------|---------|----------------|
| **Maintenance** | ❌ Unmaintained | ✅ Active | ✅ Active | ✅ Active |
| **Attention** | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes |
| **Feature Selection** | ✅ Built-in | ⚠️ Implicit | ✅ Built-in | ⚠️ Implicit |
| **Categorical Support** | ✅ Good | ✅ Excellent | ✅ Excellent | ✅ Excellent |
| **Speed** | ⚠️ Moderate | ✅ Fast | ✅ Fast | ✅ Fast |
| **Interpretability** | ✅ High | ⚠️ Moderate | ✅ High | ⚠️ Moderate |
| **Memory Usage** | ✅ Efficient | ⚠️ Higher | ✅ Efficient | ⚠️ Higher |

## Performance Comparison

Based on common benchmarks:

- **FTTransformer**: Often achieves best overall accuracy
- **GANDALF**: Strong performance with good interpretability
- **TabNet**: Competitive but no longer improving
- **TabTransformer**: Excellent for categorical-heavy data

## If You Must Use TabNet

If you have existing TabNet models or need to continue using it:

### Installation

```bash
# Install with TabNet support
pip install pytorch_tabular[extra]

# Or install pytorch-tabnet separately
pip install pytorch-tabnet<4.2
```

### Known Issues

1. **PyTorch Compatibility**: May not work with PyTorch 2.3+
2. **NumPy Compatibility**: May have issues with NumPy 2.0+
3. **Python 3.12+**: Not tested with latest Python versions

### Workarounds

If you encounter issues:

```bash
# Pin to compatible versions
pip install pytorch-tabnet<4.2 torch<2.3 numpy<2.0
```

## Future Plans

The PyTorch Tabular team is evaluating options for TabNet:

1. **Vendor the implementation** - Copy and maintain TabNet code internally
2. **Fork and maintain** - Create an actively maintained fork
3. **Deprecate** - Phase out TabNet in favor of alternatives

See issue #611 for discussion: https://github.com/pytorch-tabular/pytorch_tabular/issues/611

## Getting Help

- **Documentation**: https://pytorch-tabular.readthedocs.io/
- **Issues**: https://github.com/pytorch-tabular/pytorch_tabular/issues
- **Discussions**: https://github.com/pytorch-tabular/pytorch_tabular/discussions

For TabNet-specific issues, note that the upstream repository is not maintained.

## Summary

✅ **Recommendation**: Use FTTransformer, GANDALF, or TabTransformer for new projects  
⚠️ **Current TabNet users**: Plan migration to actively maintained alternatives  
ℹ️ **Existing models**: Will continue to work but may face compatibility issues  

The PyTorch Tabular team is committed to providing excellent tabular deep learning models. While TabNet was pioneering, newer architectures offer better performance and maintainability.
