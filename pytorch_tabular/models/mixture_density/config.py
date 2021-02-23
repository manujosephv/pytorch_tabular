# Pytorch Tabular
# Author: Manu Joseph <manujoseph@gmail.com>
# For license information, see LICENSE.TXT
"""Mixture Density Head Config"""
from dataclasses import MISSING, dataclass, field
from typing import List, Optional

from pytorch_tabular.config import ModelConfig, _validate_choices
from pytorch_tabular.models.category_embedding.config import (
    CategoryEmbeddingModelConfig,
)
from pytorch_tabular.models.node.config import NodeConfig


@dataclass
class MixtureDensityHeadConfig:
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
            "choices": ["mean", "median"],
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
        mdn_config (MixtureDensityHeadConfig): The config for defining the Mixed Density Network Head

    Raises:
        NotImplementedError: Raises an error if task is not in ['regression','classification']
    """

    mdn_config: MixtureDensityHeadConfig = field(
        default=None, metadata={"help": "The config for defining the Mixed Density Network Head"}
    )
    _module_src: str = field(default="mixture_density")
    _model_name: str = field(default="CategoryEmbeddingMDN")
    _config_name: str = field(default="CategoryEmbeddingMDNConfig")
    _probabilistic: bool = field(default=True)


@dataclass
class NODEMDNConfig(NodeConfig):
    """NODEMDN configuration
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
        mdn_config (MixtureDensityHeadConfig): The config for defining the Mixed Density Network Head

    Raises:
        NotImplementedError: Raises an error if task is not in ['regression','classification']
    """

    mdn_config: MixtureDensityHeadConfig = field(
        default=None, metadata={"help": "The config for defining the Mixed Density Network Head"}
    )
    _module_src: str = field(default="mixture_density")
    _model_name: str = field(default="NODEMDN")
    _config_name: str = field(default="NODEMDNConfig")
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
