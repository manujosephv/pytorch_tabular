import warnings
from dataclasses import dataclass, field
from typing import Optional

from pytorch_tabular.config import ModelConfig


@dataclass
class NodeConfig(ModelConfig):
    """Model configuration
    Args:
        num_layers (int): Number of Oblivious Decision Tree Layers in the Dense Architecture

        num_trees (int): Number of Oblivious Decision Trees in each layer

        additional_tree_output_dim (int): The additional output dimensions which is only used to pass
                through different layers of the architectures. Only the first output_dim outputs will be used for
                prediction

        depth (int): The depth of the individual Oblivious Decision Trees

        choice_function (str): Generates a sparse probability distribution to be used as feature
                weights(aka, soft feature selection). Choices are: [`entmax15`,`sparsemax`].

        bin_function (str): Generates a sparse probability distribution to be used as tree leaf weights.
                Choices are: [`entmoid15`,`sparsemoid`].

        max_features (Optional[int]): If not None, sets a max limit on the number of features to be carried
                forward from layer to layer in the Dense Architecture

        input_dropout (float): Dropout to be applied to the inputs between layers of the Dense Architecture

        initialize_response (str): Initializing the response variable in the Oblivious Decision Trees. By
                default, it is a standard normal distribution. Choices are: [`normal`,`uniform`].

        initialize_selection_logits (str): Initializing the feature selector. By default is a uniform
                distribution across the features. Choices are: [`uniform`,`normal`].

        threshold_init_beta (float):                  Used in the Data-aware initialization of thresholds
                where the threshold is initialized randomly                 (with a beta distribution) to feature
                values in the first batch.                 It initializes threshold to a q-th quantile of data
                points.                 where q ~ Beta(:threshold_init_beta:, :threshold_init_beta:)
                If this param is set to 1, initial thresholds will have the same distribution as data points
                If greater than 1 (e.g. 10), thresholds will be closer to median data value                 If
                less than 1 (e.g. 0.1), thresholds will approach min/max data values.

        threshold_init_cutoff (float):                  Used in the Data-aware initialization of
                scales(used in the scaling ODTs).                 It is initialized in such a way that all the
                samples in the first batch belong to the linear                 region of the
                entmoid/sparsemoid(bin-selectors) and thereby have non-zero gradients                 Threshold
                log-temperatures initializer, in (0, inf)                 By default(1.0), log-temperatures are
                initialized in such a way that all bin selectors                 end up in the linear region of
                sparse-sigmoid. The temperatures are then scaled by this parameter.                 Setting this
                value > 1.0 will result in some margin between data points and sparse-sigmoid cutoff value
                Setting this value < 1.0 will cause (1 - value) part of data points to end up in flat sparse-
                sigmoid region                 For instance, threshold_init_cutoff = 0.9 will set 10% points equal
                to 0.0 or 1.0                 Setting this value > 1.0 will result in a margin between data points
                and sparse-sigmoid cutoff value                 All points will be between (0.5 - 0.5 /
                threshold_init_cutoff) and (0.5 + 0.5 / threshold_init_cutoff)

        cat_embedding_dropout (float): DEPRECATED: Please use `embedding_dropout` instead. probability of
                an embedding element to be zeroed.

        embed_categorical (bool): Flag to embed categorical columns using an Embedding Layer. If turned
                off, the categorical columns are encoded using LeaveOneOutEncoder. This is DEPRECATED and will
                always be `True` from next release.


        task (str): Specify whether the problem is regression or classification. `backbone` is a task which
                considers the model as a backbone to generate features. Mostly used internally for SSL and related
                tasks.. Choices are: [`regression`,`classification`,`backbone`].

        head (Optional[str]): The head to be used for the model. Should be one of the heads defined in
                `pytorch_tabular.models.common.heads`. Defaults to  LinearHead. Choices are:
                [`None`,`LinearHead`,`MixtureDensityHead`].

        head_config (Optional[Dict]): The config as a dict which defines the head. If left empty, will be
                initialized as default linear head.

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

    num_layers: int = field(
        default=1,
        metadata={"help": "Number of Oblivious Decision Tree Layers in the Dense Architecture"},
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
        metadata={"help": "Dropout to be applied to the inputs between layers of the Dense Architecture"},
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
    cat_embedding_dropout: float = field(
        default=0.0,
        metadata={
            "help": "DEPRECATED: Please use `embedding_dropout` instead. probability of an embedding element to be zeroed."
        },
    )

    embed_categorical: bool = field(
        default=False,
        metadata={
            "help": "Flag to embed categorical columns using an Embedding Layer. If turned off, the categorical columns are encoded using LeaveOneOutEncoder. This is DEPRECATED and will always be `True` from next release."
        },
    )

    _module_src: str = field(default="models.node")
    _model_name: str = field(default="NODEModel")
    _backbone_name: str = field(default="NODEBackbone")
    _config_name: str = field(default="NodeConfig")

    def __post_init__(self):
        if not self.embed_categorical:
            # raise deprecation warning
            warnings.warn(
                "embed_categorical is set to False and will use LeaveOneOutEncoder to encode categorical features. This is deprecated and will be removed in future versions and categorical columns will be embedded by default."
            )
        if self.cat_embedding_dropout > 0:
            warnings.warn(
                "`cat_embedding_dropout` is deprecated and will be removed in the next release. Please use `embedding_dropout` instead"
            )
            self.embedding_dropout = self.cat_embedding_dropout
        super().__post_init__()


# if __name__ == "__main__":
#     from pytorch_tabular.utils import generate_doc_dataclass
#     print(generate_doc_dataclass(NodeConfig))
