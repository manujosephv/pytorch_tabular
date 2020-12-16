from pytorch_tabular.config import ModelConfig
from dataclasses import MISSING, dataclass, field
from typing import List, Optional, Tuple
import os
from omegaconf import OmegaConf


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

        layers (str): Hypher-separated number of layers and units in the classification head. eg. 32-64-32.
        batch_norm_continuous_input (bool): If True, we will normalize the contiinuous layer by passing it through a BatchNorm layer
        activation (str): The activation type in the classification head. 
        The default activation in PyTorch like ReLU, TanH, LeakyReLU, etc. 
        https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity
        embedding_dims (Union[List[int], NoneType]): The dimensions of the embedding for each categorical column 
        as a list of tuples (cardinality, embedding_dim). If left empty, will infer using the cardinality of the categorical column 
        using the rule min(50, (x + 1) // 2)
        embedding_dropout (float): probability of an embedding element to be zeroed.
        dropout (float): probability of an classification element to be zeroed.
        use_batch_norm (bool): Whether to use batch normalization in the classification head
        initialization (str): Initialization scheme for the linear layersChoices are: kaiming xavier random
        target_range (Union[List, NoneType]): The range in which we should limit the output variable. Currently ignored for multi-target regression
        Typically used for Regression problems. If left empty, will not apply any restrictions

    Raises:
        NotImplementedError: Raises an error if task is not in ['regression','classification']
    """

    layers: str = field(
        default="128-64-32",
        metadata={
            "help": "Hypher-separated number of layers and units in the classification head. eg. 32-64-32."
        },
    )
    batch_norm_continuous_input: bool = field(
        default=True,
        metadata={
            "help": "If True, we will normalize the contiinuous layer by passing it through a BatchNorm layer"
        },
    )
    activation: str = field(
        default="ReLU",
        metadata={
            "help": "The activation type in the classification head. The default activaion in PyTorch like ReLU, TanH, LeakyReLU, etc. https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity"
        },
    )
    embedding_dims: Optional[List[int]] = field(
        default=None,
        metadata={
            "help": "The dimensions of the embedding for each categorical column as a list of tuples (cardinality, embedding_dim). If left empty, will infer using the cardinality of the categorical column using the rule min(50, (x + 1) // 2)"
        },
    )
    embedding_dropout: float = field(
        default=0.5,
        metadata={"help": "probability of an embedding element to be zeroed."},
    )
    dropout: float = field(
        default=0.5,
        metadata={"help": "probability of an classification element to be zeroed."},
    )
    use_batch_norm: bool = field(
        default=False,
        metadata={
            "help": "Whether to use batch normalization in the classification head"
        },
    )
    initialization: str = field(
        default="kaiming",
        metadata={
            "help": "Initialization scheme for the linear layers",
            "choices": ["kaiming", "xavier", "random"],
        },
    )
    # TODO  Currently only works for single regression tasks
    target_range: Optional[List] = field(
        default=None,
        metadata={
            "help": "The range in which we should limit the output variable. Currently ignored for multi-target regression. Typically used for Regression problems. If left empty, will not apply any restrictions"
        },
    )
    _module_src: str = field(
        default="category_embedding"
    )
    _model_name: str = field(
        default="CategoryEmbeddingModel"
    )

    # def __post_init__(self):
    #     assert self._module_src == "category_embedding", "Do not change attributes starting with _"
    #     assert self._model_name == "CategoryEmbeddingModel", "Do not change attributes starting wtih _"


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