# Pytorch Tabular
# Author: Manu Joseph <manujoseph@gmail.com>
# For license information, see LICENSE.TXT
"""AutomaticFeatureInteraction Config."""
from dataclasses import dataclass, field

from pytorch_tabular.config import ModelConfig


@dataclass
class GANDALFConfig(ModelConfig):
    """Gated Adaptive Network for Deep Automated Learning of Features (GANDALF) Config.

    Args:
        gflu_stages (int): Number of layers in the feature abstraction layer. Defaults to 6

        gflu_dropout (float): Dropout rate for the feature abstraction layer. Defaults to 0.0

        gflu_feature_init_sparsity (float): Only valid for t-softmax. The percentage of features
                to be selected in each GFLU stage. This is just initialized and during learning
                it may change. Defaults to 0.3

        learnable_sparsity (bool): Only valid for t-softmax. If True, the sparsity parameters
                will be learned. If False, the sparsity parameters will be fixed to the initial
                values specified in `gflu_feature_init_sparsity` and `tree_feature_init_sparsity`.
                Defaults to True

        task (str): Specify whether the problem is regression or classification. `backbone` is a task which
                considers the model as a backbone to generate features. Mostly used internally for SSL and related
                tasks. Choices are: [`regression`,`classification`,`backbone`].

        head (Optional[str]): The head to be used for the model. Should be one of the heads defined in
                `pytorch_tabular.models.common.heads`. Defaults to  LinearHead. Choices are:
                [`None`,`LinearHead`,`MixtureDensityHead`].

        head_config (Optional[Dict]): The config as a dict which defines the head. If left empty, will be
                initialized as default linear head.

        embedding_dims (Optional[List]): The dimensions of the embedding for each categorical column as a
                list of tuples (cardinality, embedding_dim). If left empty, will infer using the cardinality of
                the categorical column using the rule min(50, (x + 1) // 2)

        embedding_dropout (float): Dropout to be applied to the Categorical Embedding. Defaults to 0.0

        batch_norm_continuous_input (bool): If True, we will normalize the continuous layer by passing it
                through a BatchNorm layer.

        learning_rate (float): The learning rate of the model. Defaults to 1e-3.

        loss (Optional[str]): The loss function to be applied. By Default, it is MSELoss for regression and
                CrossEntropyLoss for classification. Unless you are sure what you are doing, leave it at MSELoss
                or L1Loss for regression and CrossEntropyLoss for classification

        metrics (Optional[List[str]]): the list of metrics you need to track during training. The metrics
                should be one of the functional metrics implemented in ``torchmetrics``. By default, it is
                accuracy if classification and mean_squared_error for regression

        metrics_params (Optional[List]): The parameters to be passed to the metrics function. `task` is forced to
                be `multiclass` because the multiclass version can handle binary as well and for simplicity we are
                only using `multiclass`.

        metrics_prob_input (Optional[List]): Is a mandatory parameter for classification metrics defined in the config.
            This defines whether the input to the metric function is the probability or the class. Length should be
            same as the number of metrics. Defaults to None.

        target_range (Optional[List]): The range in which we should limit the output variable. Currently
                ignored for multi-target regression. Typically used for Regression problems. If left empty, will
                not apply any restrictions

        seed (int): The seed for reproducibility. Defaults to 42
    """

    gflu_stages: int = field(
        default=6,
        metadata={"help": "Number of layers in the feature abstraction layer. Defaults to 6"},
    )

    gflu_dropout: float = field(
        default=0.0, metadata={"help": "Dropout rate for the feature abstraction layer. Defaults to 0.0"}
    )

    gflu_feature_init_sparsity: float = field(
        default=0.3,
        metadata={
            "help": "Only valid for t-softmax. The perecentge of features to be selected in "
            "each GFLU stage. This is just initialized and during learning it may change"
        },
    )
    learnable_sparsity: bool = field(
        default=True,
        metadata={
            "help": "Only valid for t-softmax. If True, the sparsity parameters will be learned."
            "If False, the sparsity parameters will be fixed to the initial values specified in "
            "`gflu_feature_init_sparsity` and `tree_feature_init_sparsity`"
        },
    )
    _module_src: str = field(default="models.gandalf")
    _model_name: str = field(default="GANDALFModel")
    _backbone_name: str = field(default="GANDALFBackbone")
    _config_name: str = field(default="GANDALFConfig")

    def __post_init__(self):
        assert self.gflu_stages > 0, "gflu_stages should be greater than 0"
        return super().__post_init__()


if __name__ == "__main__":
    from pytorch_tabular.utils import generate_doc_dataclass

    print(generate_doc_dataclass(GANDALFConfig, desc="GANDALF Config"))
