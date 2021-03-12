# Pytorch Tabular
# Author: Manu Joseph <manujoseph@gmail.com>
# For license information, see LICENSE.TXT
"""AutomaticFeatureInteraction Config"""
from dataclasses import dataclass, field
from typing import List, Optional

from pytorch_tabular.config import ModelConfig, _validate_choices


@dataclass
class AutoIntConfig(ModelConfig):
    """AutomaticFeatureInteraction configuration
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

        attn_embed_dim (int): The number of hidden units in the Multi-Headed Attention layers. Defaults to 32
        num_heads (int): The number of heads in the Multi-Headed Attention layer. Defaults to 2
        num_attn_blocks (int): The number of layers of stacked Multi-Headed Attention layers. Defaults to 2
        attn_dropouts (float): Dropout between layers of Multi-Headed Attention Layers. Defaults to 0.0
        has_residuals (bool): Flag to have a residual connect from enbedded output to attention layer output.
            Defaults to True
        embedding_dim (int): The dimensions of the embedding for continuous and categorical columns. Defaults to 16
        embedding_dropout (float): probability of an embedding element to be zeroed. Defaults to 0.0
        deep_layers (bool): Flag to enable a deep MLP layer before the Multi-Headed Attention layer. Defaults to False
        layers (str): Hyphen-separated number of layers and units in the deep MLP. Defaults to 128-64-32
        activation (str): The activation type in the deep MLP. The default activaion in PyTorch like
            ReLU, TanH, LeakyReLU, etc. https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity.
            Defaults to ReLU
        dropout (float): probability of an classification element to be zeroed in the deep MLP. Defaults to 0.0
        use_batch_norm (bool): Flag to include a BatchNorm layer after each Linear Layer+DropOut. Defaults to False
        batch_norm_continuous_input (bool): If True, we will normalize the contiinuous layer by passing it through a BatchNorm layer. Defaults to False
        attention_pooling (bool): If True, will combine the attention outputs of each block for final prediction. Defaults to False
        initialization (str): Initialization scheme for the linear layers. Defaults to `kaiming`.
            Choices are: [`kaiming`,`xavier`,`random`].

    Raises:
        NotImplementedError: Raises an error if task is not in ['regression','classification']
    """

    attn_embed_dim: int = field(
        default=32,
        metadata={
            "help": "The number of hidden units in the Multi-Headed Attention layers. Defaults to 32"
        },
    )
    num_heads: int = field(
        default=2,
        metadata={
            "help": "The number of heads in the Multi-Headed Attention layer. Defaults to 2"
        },
    )
    num_attn_blocks: int = field(
        default=3,
        metadata={
            "help": "The number of layers of stacked Multi-Headed Attention layers. Defaults to 2"
        },
    )
    attn_dropouts: float = field(
        default=0.0,
        metadata={
            "help": "Dropout between layers of Multi-Headed Attention Layers. Defaults to 0.0"
        },
    )
    has_residuals: bool = field(
        default=True,
        metadata={
            "help": "Flag to have a residual connect from enbedded output to attention layer output. Defaults to True"
        },
    )
    embedding_dim: int = field(
        default=16,
        metadata={
            "help": "The dimensions of the embedding for continuous and categorical columns. Defaults to 16"
        },
    )
    embedding_dropout: float = field(
        default=0.0,
        metadata={
            "help": "probability of an embedding element to be zeroed. Defaults to 0.0"
        },
    )
    deep_layers: bool = field(
        default=False,
        metadata={
            "help": "Flag to enable a deep MLP layer before the Multi-Headed Attention layer. Defaults to False"
        },
    )
    layers: str = field(
        default="128-64-32",
        metadata={
            "help": "Hyphen-separated number of layers and units in the deep MLP. Defaults to 128-64-32"
        },
    )
    activation: str = field(
        default="ReLU",
        metadata={
            "help": "The activation type in the deep MLP. The default activaion in PyTorch like ReLU, TanH, LeakyReLU, etc. https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity. Defaults to ReLU"
        },
    )
    dropout: float = field(
        default=0.0,
        metadata={
            "help": "probability of an classification element to be zeroed in the deep MLP. Defaults to 0.0"
        },
    )
    use_batch_norm: bool = field(
        default=False,
        metadata={
            "help": "Flag to include a BatchNorm layer after each Linear Layer+DropOut. Defaults to False"
        },
    )
    batch_norm_continuous_input: bool = field(
        default=False,
        metadata={
            "help": "If True, we will normalize the continuous layer by passing it through a BatchNorm layer. Defaults to Fasle"
        },
    )
    attention_pooling: bool = field(
        default=False,
        metadata={
            "help": "If True, will combine the attention outputs of each block for final prediction. Defaults to False"
        },
    )
    initialization: str = field(
        default="kaiming",
        metadata={
            "help": "Initialization scheme for the linear layers. Defaults to `kaiming`",
            "choices": ["kaiming", "xavier", "random"],
        },
    )
    _module_src: str = field(default="autoint")
    _model_name: str = field(default="AutoIntModel")
    _config_name: str = field(default="AutoIntConfig")


# cls = AutoIntConfig
# desc = "Configuration for Data."
# doc_str = f"{desc}\nArgs:"
# for key in cls.__dataclass_fields__.keys():
#     atr = cls.__dataclass_fields__[key]
#     if atr.init:
#         type = str(atr.type).replace("<class '","").replace("'>","").replace("typing.","")
#         help_str = atr.metadata.get("help","")
#         if "choices" in atr.metadata.keys():
#             help_str += f'. Choices are: [{",".join(["`"+str(ch)+"`" for ch in atr.metadata["choices"]])}].'
#         # help_str += f'. Defaults to {atr.default}'
#         doc_str+=f'\n\t\t{key} ({type}): {help_str}'

# print(doc_str)
