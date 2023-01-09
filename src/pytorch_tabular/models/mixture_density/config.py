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
        backbone_config_class (str): The config class for defining the Backbone. The config class should be
                a valid module path from `models`. e.g. `FTTransformerConfig`

        backbone_config_params (Dict): The dict of config parameters for defining the Backbone.


        task (str): Specify whether the problem is regression or classification. `backbone` is a task which
                considers the model as a backbone to generate features. Mostly used internally for SSL and related
                tasks.. Choices are: [`regression`,`classification`,`backbone`].

        head (str):

        head_config (Dict): The config for defining the Mixed Density Network Head

        embedding_dims (Optional[List]): The dimensions of the embedding for each categorical column as a
                list of tuples (cardinality, embedding_dim). If left empty, will infer using the cardinality of
                the categorical column using the rule min(50, (x + 1) // 2)

        embedding_dropout (float): Dropout to be applied to the Categorical Embedding. Defaults to 0.1

        batch_norm_continuous_input (bool): If True, we will normalize the continuous layer by passing it
                through a BatchNorm layer.

        learning_rate (float): The learning rate of the model. Defaults to 1e-3.

        loss (Optional[str]): The loss function to be applied. By Default it is MSELoss for regression and
                CrossEntropyLoss for classification. Unless you are sure what you are doing, leave it at MSELoss
                or L1Loss for regression and CrossEntropyLoss for classification

        metrics (Optional[List[str]]): the list of metrics you need to track during training. The metrics
                should be one of the functional metrics implemented in ``torchmetrics``. By default, it is
                accuracy if classification and mean_squared_error for regression

        metrics_params (Optional[List]): The parameters to be passed to the metrics function

        target_range (Optional[List]): The range in which we should limit the output variable. Currently
                ignored for multi-target regression. Typically used for Regression problems. If left empty, will
                not apply any restrictions

        seed (int): The seed for reproducibility. Defaults to 42
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
    head: str = field(init=False, default="MixtureDensityHead")
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
        assert self.head == "MixtureDensityHead"
        return super().__post_init__()


if __name__ == "__main__":
    from pytorch_tabular.utils import generate_doc_dataclass

    print(generate_doc_dataclass(MDNConfig))
