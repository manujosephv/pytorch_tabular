"""Top-level package for Pytorch Tabular."""

__author__ = """Manu Joseph"""
__email__ = "manujosephv@gmail.com"
__version__ = "1.1.0"

from . import models, ssl_models
from .categorical_encoders import CategoricalEmbeddingTransformer
from .feature_extractor import DeepFeatureExtractor
from .tabular_datamodule import TabularDatamodule
from .tabular_datamodule_v2 import TabularDatamoduleV2
from .tabular_model import TabularModel
from .tabular_model_sweep import MODEL_SWEEP_PRESETS, model_sweep
from .tabular_model_tuner import TabularModelTuner
from .utils import available_models, available_ssl_models, get_logger

# Import backend utilities for advanced users
try:
    from .data_backends import (
        DataBackend,
        PandasBackend,
        PolarsBackend,
        get_backend,
        POLARS_AVAILABLE,
    )
    _BACKENDS_AVAILABLE = True
except ImportError:
    _BACKENDS_AVAILABLE = False

logger = get_logger("pytorch_tabular")

__all__ = [
    "TabularModel",
    "TabularModelTuner",
    "TabularDatamodule",
    "TabularDatamoduleV2",
    "models",
    "ssl_models",
    "CategoricalEmbeddingTransformer",
    "DeepFeatureExtractor",
    "utils",
    "model_sweep",
    "available_models",
    "available_ssl_models",
    "model_sweep",
    "MODEL_SWEEP_PRESETS",
]

# Add backend classes to __all__ if available
if _BACKENDS_AVAILABLE:
    __all__.extend([
        "DataBackend",
        "PandasBackend",
        "PolarsBackend",
        "get_backend",
    ])

# fix Sphinx issues, see https://bit.ly/2K2eptM
for item in __all__:
    if hasattr(item, "__module__"):
        setattr(item, "__module__", __name__)
