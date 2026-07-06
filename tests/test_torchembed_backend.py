"""Tests for the torchembed embedding backend in Embedding1dLayer.

These tests are deliberately self-contained: they exercise Embedding1dLayer
directly without needing a full TabularModel training run, so they pass even
in CI environments that have torchembed installed but no GPU.

Run with:
    pytest tests/test_torchembed_backend.py -v
"""

import pytest
import torch

from pytorch_tabular.models.common.layers.embeddings import Embedding1dLayer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CARDINALITIES = [50, 7, 120]  # vocab sizes for three categorical columns
EMBEDDING_DIMS = [(card, min(50, (card + 1) // 2)) for card in CARDINALITIES]
BATCH_SIZE = 16
CONTINUOUS_DIM = 4


def _make_batch(batch_size: int = BATCH_SIZE, n_cats: int = len(CARDINALITIES)):
    """Return a minimal input dict compatible with Embedding1dLayer.forward."""
    categorical = torch.stack(
        [torch.randint(0, CARDINALITIES[i], (batch_size,)) for i in range(n_cats)],
        dim=1,
    )
    continuous = torch.randn(batch_size, CONTINUOUS_DIM)
    return {"categorical": categorical, "continuous": continuous}


# ---------------------------------------------------------------------------
# Native backend (always runs)
# ---------------------------------------------------------------------------


def test_native_backend_output_shape():
    """Native backend concatenates per-column embeddings + continuous features."""
    layer = Embedding1dLayer(
        continuous_dim=CONTINUOUS_DIM,
        categorical_embedding_dims=EMBEDDING_DIMS,
        embedding_backend="native",
    )
    x = _make_batch()
    out = layer(x)
    expected_cat_dim = sum(dim for _, dim in EMBEDDING_DIMS)
    assert out.shape == (BATCH_SIZE, CONTINUOUS_DIM + expected_cat_dim)


def test_native_backend_output_dim_property():
    layer = Embedding1dLayer(
        continuous_dim=CONTINUOUS_DIM,
        categorical_embedding_dims=EMBEDDING_DIMS,
        embedding_backend="native",
    )
    expected = sum(dim for _, dim in EMBEDDING_DIMS)
    assert layer.output_dim == expected


def test_invalid_backend_raises():
    with pytest.raises(ValueError, match="Unknown embedding_backend"):
        Embedding1dLayer(
            continuous_dim=CONTINUOUS_DIM,
            categorical_embedding_dims=EMBEDDING_DIMS,
            embedding_backend="nonexistent",
        )


# ---------------------------------------------------------------------------
# torchembed backend (skipped when torchembed is not installed)
# ---------------------------------------------------------------------------

torchembed = pytest.importorskip(
    "torchembed",
    reason="torchembed not installed; skipping torchembed-backend tests. "
    "Install with: pip install torchembed",
)


def test_torchembed_backend_output_shape():
    """torchembed backend must produce the same 2-D output shape as the native backend."""
    layer = Embedding1dLayer(
        continuous_dim=CONTINUOUS_DIM,
        categorical_embedding_dims=EMBEDDING_DIMS,
        embedding_backend="torchembed",
    )
    x = _make_batch()
    out = layer(x)

    # output is 2-D: (batch, continuous_dim + cat_output_dim)
    assert out.ndim == 2
    assert out.shape[0] == BATCH_SIZE
    assert out.shape[1] == CONTINUOUS_DIM + layer.output_dim


def test_torchembed_backend_output_dim_property():
    """output_dim must match MultiCategoricalEmbedding.output_dim."""
    from torchembed.categorical import MultiCategoricalEmbedding

    layer = Embedding1dLayer(
        continuous_dim=CONTINUOUS_DIM,
        categorical_embedding_dims=EMBEDDING_DIMS,
        embedding_backend="torchembed",
    )
    ref = MultiCategoricalEmbedding(cardinalities=CARDINALITIES)
    assert layer.output_dim == ref.output_dim


def test_torchembed_backend_is_differentiable():
    """Gradients must flow through the torchembed embedding layer."""
    layer = Embedding1dLayer(
        continuous_dim=CONTINUOUS_DIM,
        categorical_embedding_dims=EMBEDDING_DIMS,
        embedding_backend="torchembed",
    )
    x = _make_batch()
    out = layer(x)
    loss = out.sum()
    loss.backward()
    # If we reach here without RuntimeError, gradients are flowing fine.


def test_torchembed_missing_import(monkeypatch):
    """A helpful ImportError is raised when torchembed is not importable."""
    import builtins

    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "torchembed.categorical":
            raise ImportError("mocked absence")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    with pytest.raises(ImportError, match="pip install torchembed"):
        Embedding1dLayer(
            continuous_dim=CONTINUOUS_DIM,
            categorical_embedding_dims=EMBEDDING_DIMS,
            embedding_backend="torchembed",
        )
