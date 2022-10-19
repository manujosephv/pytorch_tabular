# Pytorch Tabular
# Author: Manu Joseph <manujoseph@gmail.com>
# For license information, see LICENSE.TXT
"""Mixture Density Head Config"""
from dataclasses import dataclass, field
from typing import Dict

from pytorch_tabular.config.config import ModelConfig


INCOMPATIBLE_BACKBONES = ["NodeConfig", "TabNetModelConfig", "MDNConfig"]


@dataclass
class MDNConfig(ModelConfig):
    """MDN configuration
    Args:
        backbone_config (ModelConfig): The config for defining the Mixed Density Network Head
        mdn_config (MixtureDensityHeadConfig): The config for defining the Mixed Density Network Head

    Raises:
        NotImplementedError: Raises an error if task is not in ['regression','classification']
    """

    backbone_config_class: str = field(
        default=None,
        metadata={
            "help": "The config class for defining the Backbone. The config class should be a valid module path from `models`. e.g. `FTTransformerConfig`"
        },
    )
    backbone_config_params: Dict = field(
        default=None,
        metadata={"help": "The dict of config parameters for defining the Backbone."},
    )
    head: str = field(default="MixtureDensityHead")
    head_config: Dict = field(
        default=None,
        metadata={"help": "The config for defining the Mixed Density Network Head"},
    )
    _module_src: str = field(default="models.mixture_density")
    _model_name: str = field(default="MDNModel")
    _config_name: str = field(default="MDNConfig")
    _probabilistic: bool = field(default=True)

    def __post_init__(self):
        assert (
            self.backbone_config_class not in INCOMPATIBLE_BACKBONES
        ), f"{self.backbone_config_class} is not a supported backbone for MDN head"
        return super().__post_init__()


# cls = CategoryEmbeddingModelConfig
# desc = "Configuration for Data."
# doc_str = f"{desc}\nArgs:"
# for key in cls.__dataclass_fields__.keys():
#     atr = cls.__dataclass_fields__[key]
#     if atr.init:
#         type = str(atr.type).replace("<class '","").replace("'>","").replace("typing.","")
#         help_str = atr.metadata.get("help","")
#         if "choices" in atr.metadata.keys():
#             help_str += f'Choices are: {" ".join([str(ch) for ch in atr.metadata["choices"]])}'
#         doc_str+=f'\n\t\t{key} ({type}): {help_str}'

# print(doc_str)
