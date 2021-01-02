from dataclasses import dataclass, field
from typing import List, Optional

from pytorch_tabular.config import ModelConfig


@dataclass
class NodeConfig(ModelConfig):
    """Model configuration
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

        num_layers (int): Number of Oblivious Decision Tree Layers in the Dense Architecture
        num_trees (int): Number of Oblivious Decision Trees in each layer
        additional_tree_output_dim (int): The additional output dimensions which is only used to
            pass through different layers of the architectures. Only the first `output_dim` outputs will be used for prediction
        depth (int): The depth of the individual Oblivious Decision Trees
        choice_function (str): Generates a sparse probability distribution to be used as feature weights(aka, soft feature selection)
            Choices are: ['entmax15', 'sparsemax']
        bin_function (str): Generates a sparse probability distribution to be used as tree leaf weights
            Choices are: ['entmoid15', 'sparsemoid']
        max_features (Union[int, NoneType]): If not None, sets a max limit on the number of features to be carried forward from layer to layer in the Dense Architecture
        input_dropout (float): Dropout to be applied to the inputs between layers of the Dense Architecture
        initialize_response (str): Initializing the response variable in the Oblivious Decision Trees.
            By default, it is a standard normal distribution. Choices are: ['normal', 'uniform']
        initialize_selection_logits (str): Initializing the feature selector.
            By default is a uniform distribution across the features. Choices are: ['uniform', 'normal']
        threshold_init_beta (float):
            Used in the Data-aware initialization of thresholds where the threshold is initialized randomly
            (with a beta distribution) to feature values in the first batch.
            It initializes threshold to a q-th quantile of data points.
            where q ~ Beta(:threshold_init_beta:, :threshold_init_beta:)
            If this param is set to 1, initial thresholds will have the same distribution as data points
            If greater than 1 (e.g. 10), thresholds will be closer to median data value
            If less than 1 (e.g. 0.1), thresholds will approach min/max data values.
        threshold_init_cutoff (float):
            Used in the Data-aware initialization of scales(used in the scaling ODTs).
            It is initialized in such a way that all the samples in the first batch belong to the linear
            region of the entmoid/sparsemoid(bin-selectors) and thereby have non-zero gradients
            Threshold log-temperatures initializer, in (0, inf)
            By default(1.0), log-temperatures are initialized in such a way that all bin selectors
            end up in the linear region of sparse-sigmoid. The temperatures are then scaled by this parameter.
            Setting this value > 1.0 will result in some margin between data points and sparse-sigmoid cutoff value
            Setting this value < 1.0 will cause (1 - value) part of data points to end up in flat sparse-sigmoid region
            For instance, threshold_init_cutoff = 0.9 will set 10% points equal to 0.0 or 1.0
            Setting this value > 1.0 will result in a margin between data points and sparse-sigmoid cutoff value
            All points will be between (0.5 - 0.5 / threshold_init_cutoff) and (0.5 + 0.5 / threshold_init_cutoff)
        embed_categorical (bool): Flag to embed categorical columns using an Embedding Layer.
            If turned off, the categorical columns are encoded using LeaveOneOutEncoder
        embedding_dims (Union[List[int], NoneType]): The dimensions of the embedding for each categorical column as a
            list of tuples (cardinality, embedding_dim). If left empty, will infer using the cardinality of the categorical column
            using the rule min(50, (x + 1) // 2)
        embedding_dropout (float): probability of an embedding element to be zeroed.

    Raises:
        NotImplementedError: Raises an error if task is not in ['regression','classification']
    """

    num_layers: int = field(
        default=1,
        metadata={
            "help": "Number of Oblivious Decision Tree Layers in the Dense Architecture"
        },
    )
    num_trees: int = field(
        default=2048,
        metadata={"help": "Number of Oblivious Decision Trees in each layer"},
    )
    additional_tree_output_dim: int = field(
        default=3,
        metadata={
            "help": "The additional output dimensions which is only used to pass through different layers of the architectures. Only the first output_dim outputs will be used for prediction"
        },
    )
    depth: int = field(
        default=6,
        metadata={"help": "The depth of the individual Oblivious Decision Trees"},
    )
    choice_function: str = field(
        default="entmax15",
        metadata={
            "help": "Generates a sparse probability distribution to be used as feature weights(aka, soft feature selection)",
            "choices": ["entmax15", "sparsemax"],
        },
    )
    bin_function: str = field(
        default="entmoid15",
        metadata={
            "help": "Generates a sparse probability distribution to be used as tree leaf weights",
            "choices": ["entmoid15", "sparsemoid"],
        },
    )
    max_features: Optional[int] = field(
        default=None,
        metadata={
            "help": "If not None, sets a max limit on the number of features to be carried forward from layer to layer in the Dense Architecture"
        },
    )
    input_dropout: float = field(
        default=0.0,
        metadata={
            "help": "Dropout to be applied to the inputs between layers of the Dense Architecture"
        },
    )
    initialize_response: str = field(
        default="normal",
        metadata={
            "help": "Initializing the response variable in the Oblivious Decision Trees. By default, it is a standard normal distribution",
            "choices": ["normal", "uniform"],
        },
    )
    initialize_selection_logits: str = field(
        default="uniform",
        metadata={
            "help": "Initializing the feature selector. By default is a uniform distribution across the features",
            "choices": ["uniform", "normal"],
        },
    )
    threshold_init_beta: float = field(
        default=1.0,
        metadata={
            "help": """
                Used in the Data-aware initialization of thresholds where the threshold is initialized randomly
                (with a beta distribution) to feature values in the first batch.
                It initializes threshold to a q-th quantile of data points.
                where q ~ Beta(:threshold_init_beta:, :threshold_init_beta:)
                If this param is set to 1, initial thresholds will have the same distribution as data points
                If greater than 1 (e.g. 10), thresholds will be closer to median data value
                If less than 1 (e.g. 0.1), thresholds will approach min/max data values.
            """
        },
    )
    threshold_init_cutoff: float = field(
        default=1.0,
        metadata={
            "help": """
                Used in the Data-aware initialization of scales(used in the scaling ODTs).
                It is initialized in such a way that all the samples in the first batch belong to the linear
                region of the entmoid/sparsemoid(bin-selectors) and thereby have non-zero gradients
                Threshold log-temperatures initializer, in (0, inf)
                By default(1.0), log-temperatures are initialized in such a way that all bin selectors
                end up in the linear region of sparse-sigmoid. The temperatures are then scaled by this parameter.
                Setting this value > 1.0 will result in some margin between data points and sparse-sigmoid cutoff value
                Setting this value < 1.0 will cause (1 - value) part of data points to end up in flat sparse-sigmoid region
                For instance, threshold_init_cutoff = 0.9 will set 10% points equal to 0.0 or 1.0
                Setting this value > 1.0 will result in a margin between data points and sparse-sigmoid cutoff value
                All points will be between (0.5 - 0.5 / threshold_init_cutoff) and (0.5 + 0.5 / threshold_init_cutoff)
            """
        },
    )
    embed_categorical: bool = field(
        default=False,
        metadata={
            "help": "Flag to embed categorical columns using an Embedding Layer. If turned off, the categorical columns are encoded using LeaveOneOutEncoder"
        },
    )
    embedding_dims: Optional[List[int]] = field(
        default=None,
        metadata={
            "help": "The dimensions of the embedding for each categorical column as a list of tuples (cardinality, embedding_dim). If left empty, will infer using the cardinality of the categorical column using the rule min(50, (x + 1) // 2)"
        },
    )
    embedding_dropout: float = field(
        default=0.0,
        metadata={"help": "probability of an embedding element to be zeroed."},
    )
    _module_src: str = field(default="node")
    _model_name: str = field(default="NODEModel")
    _config_name: str = field(default="NodeConfig")

    # def __post_init__(self):
    #     self._model_name = "NODEModel"
    #     if self.embed_categorical:
    #         self._model_name = "CategoryEmbedding"+self._model_name
    # assert self._module_src == "category_embedding", "Do not change attributes starting with _"
    # assert self._model_name == "CategoryEmbeddingModel", "Do not change attributes starting wtih _"


# cls = NodeConfig
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
