# Pytorch Tabular
# Author: Manu Joseph <manujoseph@gmail.com>
# For license information, see LICENSE.TXT
"""AutomaticFeatureInteraction Config"""
import warnings
from dataclasses import dataclass, field
from typing import Optional

from pytorch_tabular.config import ModelConfig
from pytorch_tabular.utils import ifnone


@dataclass
class FTTransformerConfig(ModelConfig):
    """Tab Transformer configuration
    Args:
        task (str): Specify whether the problem is regression of classification.
            Choices are: [`regression`,`classification`].
        embedding_dims (Union[List[int], NoneType]): The dimensions of the embedding for
            each categorical column as a list of tuples (cardinality, embedding_dim).
            If left empty, will infer using the cardinality of the categorical column using
            the rule min(50, (x + 1) // 2)
        learning_rate (float): The learning rate of the model
        loss (Union[str, NoneType]): The loss function to be applied.
            By Default it is MSELoss for regression and CrossEntropyLoss for classification.
            Unless you are sure what you are doing, leave it at MSELoss or L1Loss for regression
            and CrossEntropyLoss for classification
        metrics (Union[List[str], NoneType]): the list of metrics you need to track during training.
            The metrics should be one of the functional metrics implemented in ``torchmetrics``.
            By default, it is accuracy if classification and mean_squared_error for regression
        metrics_params (Union[List, NoneType]): The parameters to be passed to the metrics function
        target_range (Union[List, NoneType]): The range in which we should limit the output variable.
            Currently ignored for multi-target regression. Typically used for Regression problems.
            If left empty, will not apply any restrictions

        input_embed_dim (int): The embedding dimension for the input categorical features.
            Defaults to 32
        embedding_dropout (float): Dropout to be applied to the Categorical Embedding.
            Defaults to 0.1
        share_embedding (bool): The flag turns on shared embeddings in the input embedding process.
            The key idea here is to have an embedding for the feature as a whole along with embeddings of
            each unique values of that column. For more details refer to Appendix A of the TabTransformer paper.
            Defaults to False
        share_embedding_strategy (Union[str, NoneType]): There are two strategies in adding shared embeddings.
            1. `add` - A separate embedding for the feature is added to the embedding of the unique values of the feature.
            2. `fraction` - A fraction of the input embedding is reserved for the shared embedding of the feature.
            Defaults to fraction.
            Choices are: [`add`,`fraction`].
        shared_embedding_fraction (float): Fraction of the input_embed_dim to be reserved by the shared embedding.
            Should be less than one. Defaults to 0.25
        num_heads (int): The number of heads in the Multi-Headed Attention layer.
            Defaults to 8
        num_attn_blocks (int): The number of layers of stacked Multi-Headed Attention layers.
            Defaults to 6
        transformer_head_dim (Union[int, NoneType]): The number of hidden units in the Multi-Headed Attention layers.
            Defaults to None and will be same as input_dim.
        attn_dropout (float): Dropout to be applied after Multi headed Attention.
            Defaults to 0.1
        add_norm_dropout (float): Dropout to be applied in the AddNorm Layer.
            Defaults to 0.1
        ff_dropout (float): Dropout to be applied in the Positionwise FeedForward Network.
            Defaults to 0.1
        ff_hidden_multiplier (int): Multiple by which the Positionwise FF layer scales the input.
            Defaults to 4
        transformer_activation (str): The activation type in the transformer feed forward layers.
            In addition to the default activation in PyTorch like ReLU, TanH, LeakyReLU, etc.
            https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity,
            GEGLU, ReGLU and SwiGLU are also implemented(https://arxiv.org/pdf/2002.05202.pdf).
            Defaults to GEGLU
        out_ff_layers (str): Hyphen-separated number of layers and units in the deep MLP.
            Defaults to 128-64-32
        out_ff_activation (str): The activation type in the deep MLP. The default activaion in PyTorch like ReLU, TanH, LeakyReLU, etc.
            https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity.
            Defaults to ReLU
        out_ff_dropout (float): Probability of an classification element to be zeroed in the deep MLP.
            Defaults to 0.0
        use_batch_norm (bool): Flag to include a BatchNorm layer after each Linear Layer+DropOut.
            Defaults to False
        batch_norm_continuous_input (bool): If True, we will normalize the continuous layer by passing it through a BatchNorm layer.
            Defaults to False
        out_ff_initialization (str): Initialization scheme for the linear layers.
            Defaults to `kaiming`.
            Choices are: [`kaiming`,`xavier`,`random`].

    Raises:
        NotImplementedError: Raises an error if task is not in ['regression','classification']
    """

    input_embed_dim: int = field(
        default=32,
        metadata={"help": "The embedding dimension for the input categorical features. Defaults to 32"},
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
    attn_feature_importance: bool = field(
        default=True,
        metadata={
            "help": "If you are facing memory issues, you can turn off feature importance which will not save the attention weights. Defaults to True"
        },
    )
    num_heads: int = field(
        default=8,
        metadata={"help": "The number of heads in the Multi-Headed Attention layer. Defaults to 8"},
    )
    num_attn_blocks: int = field(
        default=6,
        metadata={"help": "The number of layers of stacked Multi-Headed Attention layers. Defaults to 6"},
    )
    transformer_head_dim: Optional[int] = field(
        default=None,
        metadata={
            "help": "The number of hidden units in the Multi-Headed Attention layers. Defaults to None and will be same as input_dim."
        },
    )
    attn_dropout: float = field(
        default=0.1,
        metadata={"help": "Dropout to be applied after Multi headed Attention. Defaults to 0.1"},
    )
    add_norm_dropout: float = field(
        default=0.1,
        metadata={"help": "Dropout to be applied in the AddNorm Layer. Defaults to 0.1"},
    )
    ff_dropout: float = field(
        default=0.1,
        metadata={"help": "Dropout to be applied in the Positionwise FeedForward Network. Defaults to 0.1"},
    )
    ff_hidden_multiplier: int = field(
        default=4,
        metadata={"help": "Multiple by which the Positionwise FF layer scales the input. Defaults to 4"},
    )
    # TODO improve documentation
    transformer_activation: str = field(
        default="GEGLU",
        metadata={
            "help": "The activation type in the transformer feed forward layers. In addition to the default activation in PyTorch like ReLU, TanH, LeakyReLU, etc. https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity, GEGLU, ReGLU and SwiGLU are also implemented(https://arxiv.org/pdf/2002.05202.pdf). Defaults to GEGLU",
        },
    )
    out_ff_layers: Optional[str] = field(
        default=None,
        metadata={
            "help": "DEPRECATED: Hyphen-separated number of layers and units in the deep MLP. Defaults to 128-64-32"
        },
    )
    out_ff_activation: Optional[str] = field(
        default=None,
        metadata={
            "help": "DEPRECATED: The activation type in the deep MLP. The default activaion in PyTorch like ReLU, TanH, LeakyReLU, etc. https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity. Defaults to ReLU"
        },
    )
    out_ff_dropout: Optional[float] = field(
        default=None,
        metadata={
            "help": "DEPRECATED: probability of an classification element to be zeroed in the deep MLP. Defaults to 0.0"
        },
    )
    out_ff_initialization: Optional[str] = field(
        default=None,
        metadata={
            "help": "DEPRECATED: Initialization scheme for the linear layers. Defaults to `kaiming`",
            "choices": [None, "kaiming", "xavier", "random"],
        },
    )
    _module_src: str = field(default="models.ft_transformer")
    _model_name: str = field(default="FTTransformerModel")
    _backbone_name: str = field(default="FTTransformerBackbone")
    _config_name: str = field(default="FTTransformerConfig")

    def __post_init__(self):
        deprecated_args = [
            "out_ff_layers",
            "out_ff_activation",
            "out_ff_dropoout",
            "out_ff_initialization",
        ]
        if self.head_config != {"layers": ""}:  # If the user has passed a head_config
            warnings.warn(
                "Ignoring the deprecated arguments, `out_ff_layers`, `out_ff_activation`, `out_ff_dropoout`, and `out_ff_initialization` as head_config is passed."
            )
        else:
            if any([p is not None for p in deprecated_args]):
                warnings.warn(
                    "The `out_ff_layers`, `out_ff_activation`, `out_ff_dropoout`, and `out_ff_initialization` arguments are deprecated and will be removed next release. Please use head and head_config as an alternative.",
                    DeprecationWarning,
                )
                # TODO: Remove this once we deprecate the old config
                # Fill the head_config using deprecated parameters
                self.head_config = dict(
                    layers=ifnone(self.out_ff_layers, ""),
                    activation=ifnone(self.out_ff_activation, "ReLU"),
                    dropout=ifnone(self.out_ff_dropout, 0.0),
                    use_batch_norm=False,
                    initialization=ifnone(self.out_ff_initialization, "kaiming"),
                )

        return super().__post_init__()


# cls = TabTransformerConfig
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
