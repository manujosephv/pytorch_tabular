# Pytorch Tabular
# Author: Manu Joseph <manujoseph@gmail.com>
# For license information, see LICENSE.TXT
"""Mixture Density Head Config"""
from dataclasses import MISSING, dataclass, field
from pytorch_tabular.models.category_embedding.config import CategoryEmbeddingModelConfig
from typing import List, Optional

from pytorch_tabular.config import ModelConfig, _validate_choices


@dataclass
class MixtureDensityHeadConfig():
    """MixtureDensityHead configuration
    Args:
        num_gaussian (int): Number of Gaussian Distributions in the mixture model. Defaults to 1
        n_samples (int): Number of samples to draw from the posterior to get prediction. Defaults to 100
        central_tendency (str): Which measure to use to get the point prediction. Choices are 'mean', 'median'. Defaults to `mean`

    """

    num_gaussian: int = field(
        default=1,
        metadata={
            "help": "Number of Gaussian Distributions in the mixture model. Defaults to 1",
        },
    )
    n_samples: int = field(
        default=100,
        metadata={
            "help": "Number of samples to draw from the posterior to get prediction. Defaults to 100",
        },
    )
    central_tendency: str = field(
        default="mean",
        metadata={
            "help": "Which measure to use to get the point prediction. Defaults to mean",
            "choices": ['mean', 'median']
        },
    )
    fast_training: bool = field(
        default=False,
        metadata={
            "help": "Turning onthis parameter does away with sampling during training which speeds up training, but also doesn't give you visibility on training metrics. Defaults to True",
        },
    )
    _module_src: str = field(default="mixture_density")
    _model_name: str = field(default="MixtureDensityHead")
    _config_name: str = field(default="MixtureDensityHeadConfig")

@dataclass
class CategoryEmbeddingMDNConfig(CategoryEmbeddingModelConfig):
    """CategoryEmbeddingMDN configuration
    Args:
        task (str): Specify whether the problem is regression of classification.Choices are: regression classification
        learning_rate (float): The learning rate of the model
        loss (Union[str, NoneType]): The loss function to be applied.
            By Default it is MSELoss for regression and CrossEntropyLoss for classification.
            Unless you are sure what you are doing, leave it at MSELoss or L1Loss for regression and CrossEntropyLoss for classification
        metrics (Union[List[str], NoneType]): the list of metrics you need to track during training.
            The metrics should be one of the metrics implemented in PyTorch Lightning.
            By default, it is Accuracy if classification and MeanSquaredLogError for regression
        metrics_params (Union[List, NoneType]): The parameters to be passed to the Metrics initialized
        target_range (Union[List, NoneType]): The range in which we should limit the output variable. Currently ignored for multi-target regression
            Typically used for Regression problems. If left empty, will not apply any restrictions

        layers (str): Hyphen-separated number of layers and units in the classification head. eg. 32-64-32.
        batch_norm_continuous_input (bool): If True, we will normalize the contiinuous layer by passing it through a BatchNorm layer
        activation (str): The activation type in the classification head.
            The default activation in PyTorch like ReLU, TanH, LeakyReLU, etc.
            https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity
        embedding_dims (Union[List[int], NoneType]): The dimensions of the embedding for each categorical column
            as a list of tuples (cardinality, embedding_dim). If left empty, will infer using the cardinality of the categorical column
            using the rule min(50, (x + 1) // 2)
        embedding_dropout (float): probability of an embedding element to be zeroed.
        dropout (float): probability of an classification element to be zeroed.
        use_batch_norm (bool): Flag to include a BatchNorm layer after each Linear Layer+DropOut
        initialization (str): Initialization scheme for the linear layers. Choices are: `kaiming` `xavier` `random`

    Raises:
        NotImplementedError: Raises an error if task is not in ['regression','classification']
    """

    mdn_config: MixtureDensityHeadConfig = field(
        default=None,
        metadata = {
            "help" : "The config for defining the MDN"
        }
    )
    _module_src: str = field(default="mixture_density")
    _model_name: str = field(default="CategoryEmbeddingMDN")
    _config_name: str = field(default="CategoryEmbeddingMDNConfig")
    _probabilistic: bool = field(default=True)
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
