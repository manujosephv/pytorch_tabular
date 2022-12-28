# Pytorch Tabular
# Author: Manu Joseph <manujoseph@gmail.com>
# For license information, see LICENSE.TXT
"""AutomaticFeatureInteraction Config"""
from dataclasses import dataclass, field

from pytorch_tabular.config import ModelConfig


@dataclass
class GatedAdditiveTreeEnsembleConfig(ModelConfig):
    gflu_stages: int = field(
        default=6,
        metadata={"help": "Number of layers in the feature abstraction layer"},
    )

    gflu_dropout: float = field(
        default=0.0, metadata={"help": "Dropout rate for the feature abstraction layer"}
    )

    tree_depth: int = field(default=1, metadata={"help": "Depth of the tree. "})

    num_trees: int = field(
        default=1,
        metadata={"help": "Number of trees to use in the ensemble. Defaults to 10"},
    )

    binning_activation: str = field(
        default="entmoid",
        metadata={
            "help": "The binning function to use. Defaults to entmoid",
            "choices": ["entmoid", "sparsemoid", "sigmoid"],
        },
    )
    feature_mask_function: str = field(
        default="entmax",
        metadata={
            "help": "The feature mask function to use. Defaults to entmax",
            "choices": ["entmax", "sparsemax", "softmax"],
        },
    )

    tree_dropout: float = field(
        default=0.0,
        metadata={"help": "probability of dropout in tree binning transformation."},
    )
    batch_norm_continuous_input: bool = field(
        default=True,
        metadata={
            "help": "If True, we will normalize the contiinuous layer by passing it through a BatchNorm layer"
        },
    )
    embedding_dropout: float = field(
        default=0.0,
        metadata={"help": "Dropout for the categorical embedding layer."},
    )

    use_batch_norm: bool = field(
        default=False,
        metadata={
            "help": "Flag to include a BatchNorm layer after each Linear Layer+DropOut"
        },
    )
    initialization: str = field(
        default="kaiming",
        metadata={
            "help": "Initialization scheme for the linear layers",
            "choices": ["kaiming", "xavier", "random"],
        },
    )
    activation: str = field(
        default="ReLU",
        metadata={
            "help": "The activation type in the classification head. The default activaion in PyTorch like ReLU, TanH, LeakyReLU, etc. https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity"
        },
    )
    chain_trees: bool = field(
        default=False,
        metadata={
            "help": "If True, we will chain the trees together. Defaults to False"
        },
    )
    tree_wise_attention: bool = field(
        default=True,
        metadata={
            "help": "If True, we will use tree wise attention to combine trees. Defaults to True"
        },
    )
    tree_wise_attention_dropout: float = field(
        default=0.0,
        metadata={
            "help": "probability of dropout in the tree wise attention layer. Defaults to 0.0"
        },
    )
    share_head_weights: bool = field(
        default=True,
        metadata={
            "help": "If True, we will share the weights between the heads. Defaults to True"
        },
    )

    _module_src: str = field(default="models.gate")
    _model_name: str = field(default="GatedAdditiveTreeEnsembleModel")
    _backbone_name: str = field(default="GatedAdditiveTreesBackbone")
    _config_name: str = field(default="GatedAdditiveTreeEnsembleConfig")

    def __post_init__(self):
        # if self.chain_trees and self.lambda_feature_diversity > 0:
        #     raise ValueError("Cannot use lambda_feature_diversity with chain_trees")
        super().__post_init__()


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
