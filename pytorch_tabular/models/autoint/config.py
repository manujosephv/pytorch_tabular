# Pytorch Tabular
# Author: Manu Joseph <manujoseph@gmail.com>
# For license information, see LICENSE.TXT
"""AutomaticFeatureInteraction Config"""
from dataclasses import dataclass, field
from typing import Optional

from pytorch_tabular.config import ModelConfig


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
        metadata={"help": "The number of hidden units in the Multi-Headed Attention layers. Defaults to 32"},
    )
    num_heads: int = field(
        default=2,
        metadata={"help": "The number of heads in the Multi-Headed Attention layer. Defaults to 2"},
    )
    num_attn_blocks: int = field(
        default=3,
        metadata={"help": "The number of layers of stacked Multi-Headed Attention layers. Defaults to 2"},
    )
    attn_dropouts: float = field(
        default=0.0,
        metadata={"help": "Dropout between layers of Multi-Headed Attention Layers. Defaults to 0.0"},
    )
    has_residuals: bool = field(
        default=True,
        metadata={
            "help": "Flag to have a residual connect from enbedded output to attention layer output. Defaults to True"
        },
    )
    embedding_dim: int = field(
        default=16,
        metadata={"help": "The dimensions of the embedding for continuous and categorical columns. Defaults to 16"},
    )
    embedding_initialization: Optional[str] = field(
        default="kaiming_uniform",
        metadata={
            "help": "Initialization scheme for the embedding layers. Defaults to `kaiming`",
            "choices": ["kaiming_uniform", "kaiming_normal"],
        },
    )
    embedding_bias: bool = field(
        default=True,
        metadata={"help": "Flag to turn on Embedding Bias. Defaults to True"},
    )
    share_embedding: bool = field(
        default=False,
        metadata={
            "help": "The flag turns on shared embeddings in the input embedding process. The key idea here is to have an embedding for the feature as a whole along with embeddings of each unique values of that column. For more details refer to Appendix A of the TabTransformer paper. Defaults to False"
        },
    )
    share_embedding_strategy: Optional[str] = field(
        default="fraction",
        metadata={
            "help": "There are two strategies in adding shared embeddings. 1. `add` - A separate embedding for the feature is added to the embedding of the unique values of the feature. 2. `fraction` - A fraction of the input embedding is reserved for the shared embedding of the feature. Defaults to fraction.",
            "choices": ["add", "fraction"],
        },
    )
    shared_embedding_fraction: float = field(
        default=0.25,
        metadata={
            "help": "Fraction of the input_embed_dim to be reserved by the shared embedding. Should be less than one. Defaults to 0.25"
        },
    )
    deep_layers: bool = field(
        default=False,
        metadata={"help": "Flag to enable a deep MLP layer before the Multi-Headed Attention layer. Defaults to False"},
    )
    layers: str = field(
        default="128-64-32",
        metadata={"help": "Hyphen-separated number of layers and units in the deep MLP. Defaults to 128-64-32"},
    )
    activation: str = field(
        default="ReLU",
        metadata={
            "help": "The activation type in the deep MLP. The default activaion in PyTorch like ReLU, TanH, LeakyReLU, etc. https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity. Defaults to ReLU"
        },
    )
    use_batch_norm: bool = field(
        default=False,
        metadata={
            "help": "Flag to include a BatchNorm layer after each Linear Layer+DropOut in the deep MLP. Defaults to False"
        },
    )
    initialization: str = field(
        default="kaiming",
        metadata={
            "help": "Initialization scheme for the linear layers in the deep MLP. Defaults to `kaiming`",
            "choices": ["kaiming", "xavier", "random"],
        },
    )
    dropout: float = field(
        default=0.0,
        metadata={"help": "probability of an element to be zeroed in the deep MLP. Defaults to 0.0"},
    )
    attention_pooling: bool = field(
        default=False,
        metadata={
            "help": "If True, will combine the attention outputs of each block for final prediction. Defaults to False"
        },
    )
    _module_src: str = field(default="models.autoint")
    _model_name: str = field(default="AutoIntModel")
    _backbone_name: str = field(default="AutoIntBackbone")
    _config_name: str = field(default="AutoIntConfig")
