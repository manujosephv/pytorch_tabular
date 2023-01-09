"""Top-level package for Pytorch Tabular."""

__author__ = """Manu Joseph"""
__email__ = "manujosephv@gmail.com"
__version__ = "0.7.0"

from . import models, ssl_models
from .tabular_datamodule import TabularDatamodule
from .tabular_model import TabularModel

__all__ = [
    "TabularModel",
    "TabularDatamodule",
    "models",
    "ssl_models",
]

# fix Sphinx issues, see https://bit.ly/2K2eptM
for item in __all__:
    if hasattr(item, "__module__"):
        setattr(item, "__module__", __name__)
