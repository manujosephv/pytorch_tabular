# Pytorch Tabular
# Author: Manu Joseph <manujoseph@gmail.com>
# For license information, see LICENSE.TXT
"""Category Embedding Model Config"""
from dataclasses import dataclass, field

from pytorch_tabular.config import ModelConfig


@dataclass
class CategoryEmbeddingModelConfig(ModelConfig):
    """CategoryEmbeddingModel configuration
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

    layers: str = field(
        default="128-64-32",
        metadata={
            "help": "Hyphen-separated number of layers and units in the classification head. eg. 32-64-32. Defaults to 128-64-32"
        },
    )
    activation: str = field(
        default="ReLU",
        metadata={
            "help": "The activation type in the classification head. The default activaion in PyTorch like ReLU, TanH, LeakyReLU, etc. https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity. Defaults to ReLU"
        },
    )
    use_batch_norm: bool = field(
        default=False,
        metadata={"help": "Flag to include a BatchNorm layer after each Linear Layer+DropOut. Defaults to False"},
    )
    initialization: str = field(
        default="kaiming",
        metadata={
            "help": "Initialization scheme for the linear layers. Defaults to `kaiming`",
            "choices": ["kaiming", "xavier", "random"],
        },
    )
    dropout: float = field(
        default=0.0,
        metadata={
            "help": "probability of an classification element to be zeroed. This is added to each linear layer. Defaults to 0.0"
        },
    )
    _module_src: str = field(default="models.category_embedding")
    _model_name: str = field(default="CategoryEmbeddingModel")
    _backbone_name: str = field(default="CategoryEmbeddingBackbone")
    _config_name: str = field(default="CategoryEmbeddingModelConfig")

    # def __post_init__(self):
    #     assert self.head in dir(heads.blocks), f"{self.head} is not a valid head"
    #     _head_callable = getattr(heads.blocks, self.head)
    #     ideal_head_config = _head_callable._config_template
    #     invalid_keys = set(self.head_config.keys()) - set(
    #         ideal_head_config.__dict__.keys()
    #     )
    #     assert (
    #         len(invalid_keys) == 0
    #     ), f"`head_config` has some invalid keys: {invalid_keys}"

    #     # if self.head_config == {"layers": ""}:  # Default Head Config provided
    #     #     pass
    #     #     # self.head = "LinearHead"
    #     #     # # TODO: Remove this once we deprecate the old config
    #     #     # # Fill the head_config using deprecated parameters
    #     #     # self.head_config = dict(
    #     #     #     layers=ifnone(self.layers, "128-64-32"),
    #     #     #     activation=ifnone(self.activation, "ReLU"),
    #     #     #     dropout=ifnone(self.dropout, 0.0),
    #     #     #     use_batch_norm=ifnone(self.use_batch_norm, False),
    #     #     #     initialization=ifnone(self.initialization, "kaiming"),
    #     #     # )
    #     # else:
    #     #     assert self.head in dir(heads.blocks), f"{self.head} is not a valid head"
    #     #     _head_callable = getattr(heads.blocks, self.head)
    #     #     ideal_head_config = _head_callable._config_template
    #     #     invalid_keys = set(self.head_config.keys()) - set(
    #     #         ideal_head_config.__dict__.keys()
    #     #     )
    #     #     assert (
    #     #         len(invalid_keys) == 0
    #     #     ), f"`head_config` has some invalid keys: {invalid_keys}"

    #     return super().__post_init__()


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
