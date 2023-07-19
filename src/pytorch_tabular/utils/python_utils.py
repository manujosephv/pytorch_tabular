import textwrap
from pathlib import Path
from typing import Any, Callable, Dict, IO, Optional, Union

import numpy as np
import torch
from lightning_lite.utilities.cloud_io import get_filesystem

import pytorch_tabular as root_module

from .logger import get_logger

_PATH = Union[str, Path]
_DEVICE = Union[torch.device, str, int]
_MAP_LOCATION_TYPE = Optional[Union[_DEVICE, Callable[[_DEVICE], _DEVICE], Dict[_DEVICE, _DEVICE]]]


logger = get_logger(__name__)


def getattr_nested(_module_src, _model_name):
    module = root_module
    for m in _module_src.split("."):
        module = getattr(module, m)
    return getattr(module, _model_name)


def ifnone(arg, default_arg):
    return default_arg if arg is None else arg


def generate_doc_dataclass(dataclass, desc=None, width=100):
    if desc is not None:
        doc_str = f"{desc}\nArgs:"
    else:
        doc_str = "Args:"
    for key in dataclass.__dataclass_fields__.keys():
        if key.startswith("_"):  # Skipping private fields
            continue
        atr = dataclass.__dataclass_fields__[key]
        if atr.init:
            type = str(atr.type).replace("<class '", "").replace("'>", "").replace("typing.", "")
            help_str = atr.metadata.get("help", "")
            if "choices" in atr.metadata.keys():
                help_str += f'. Choices are: [{",".join(["`"+str(ch)+"`" for ch in atr.metadata["choices"]])}].'
            # help_str += f'. Defaults to {atr.default}'
            h_str = textwrap.fill(
                f"{key} ({type}): {help_str}",
                width=width,
                subsequent_indent="\t\t",
                initial_indent="\t",
            )
            h_str = f"\n{h_str}\n"
            doc_str += h_str
    return doc_str


# Copied over pytorch_lightning.utilities.cloud_io.load as it was deprecated
def pl_load(
    path_or_url: Union[IO, _PATH],
    map_location: _MAP_LOCATION_TYPE = None,
) -> Any:
    """Loads a checkpoint.

    Args:
        path_or_url: Path or URL of the checkpoint.
        map_location: a function, ``torch.device``, string or a dict specifying how to remap storage locations.
    """
    if not isinstance(path_or_url, (str, Path)):
        # any sort of BytesIO or similar
        return torch.load(path_or_url, map_location=map_location)
    if str(path_or_url).startswith("http"):
        return torch.hub.load_state_dict_from_url(
            str(path_or_url),
            map_location=map_location,  # type: ignore[arg-type] # upstream annotation is not correct
        )
    fs = get_filesystem(path_or_url)
    with fs.open(path_or_url, "rb") as f:
        return torch.load(f, map_location=map_location)


def check_numpy(x):
    """Makes sure x is a numpy array."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    assert isinstance(x, np.ndarray)
    return x
