# Pytorch Tabular
# Author: Manu Joseph <manujoseph@gmail.com>
# For license information, see LICENSE.TXT
"""Tabnet Model Config"""
from dataclasses import dataclass, field

from pytorch_tabular.config import ModelConfig


@dataclass
class TabNetModelConfig(ModelConfig):
    """Model configuration
    Args:
        n_d (int): Dimension of the prediction  layer (usually between 4 and 64)

        n_a (int): Dimension of the attention  layer (usually between 4 and 64)

        n_steps (int): Number of sucessive steps in the newtork (usually betwenn 3 and 10)

        gamma (float): Float above 1, scaling factor for attention updates (usually betwenn 1.0 to 2.0)

        n_independent (int): Number of independent GLU layer in each GLU block (default 2)

        n_shared (int): Number of independent GLU layer in each GLU block (default 2)

        virtual_batch_size (int): Batch size for Ghost Batch Normalization

        mask_type (str): Either 'sparsemax' or 'entmax' : this is the masking function to use. Choices are:
                [`sparsemax`,`entmax`].


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

    n_d: int = field(
        default=8,
        metadata={"help": "Dimension of the prediction  layer (usually between 4 and 64)"},
    )
    n_a: int = field(
        default=8,
        metadata={"help": "Dimension of the attention  layer (usually between 4 and 64)"},
    )
    n_steps: int = field(
        default=3,
        metadata={"help": "Number of sucessive steps in the newtork (usually betwenn 3 and 10)"},
    )
    gamma: float = field(
        default=1.3,
        metadata={"help": "Float above 1, scaling factor for attention updates (usually betwenn 1.0 to 2.0)"},
    )
    n_independent: int = field(
        default=2,
        metadata={"help": "Number of independent GLU layer in each GLU block (default 2)"},
    )
    n_shared: int = field(
        default=2,
        metadata={"help": "Number of independent GLU layer in each GLU block (default 2)"},
    )
    virtual_batch_size: int = field(
        default=128,
        metadata={"help": "Batch size for Ghost Batch Normalization"},
    )
    mask_type: str = field(
        default="sparsemax",
        metadata={
            "help": "Either 'sparsemax' or 'entmax' : this is the masking function to use",
            "choices": ["sparsemax", "entmax"],
        },
    )
    _module_src: str = field(default="models.tabnet")
    _model_name: str = field(default="TabNetModel")
    _config_name: str = field(default="TabNetModelConfig")


# if __name__ == "__main__":
#     from pytorch_tabular.utils import generate_doc_dataclass
#     print(generate_doc_dataclass(TabNetModelConfig))
