from pytorch_tabular.config import ModelConfig
from dataclasses import MISSING, dataclass, field
from typing import List, Optional, Tuple
import os
from omegaconf import OmegaConf


@dataclass
class TabnetModelConfig(ModelConfig):
    """Model configuration
    Args:
        layers (str): Hypher-separated number of layers and units in the classification head. eg. 32-64-32.
        batch_norm_continuous_input (bool): If True, we will normalize the contiinuous layer by passing it through a BatchNorm layer
        activation (str): The activation type in the classification head. The default activaion in PyTorch like ReLU, TanH, LeakyReLU, etc. https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity
        embedding_dims (Union[List[int], NoneType]): The dimensions of the embedding for each categorical column as a list of tuples (cardinality, embedding_dim). If left empty, will infer using the cardinality of the categorical column using the rule min(50, (x + 1) // 2)
        embedding_dropout (float): probability of an embedding element to be zeroed.
        dropout (float): probability of an classification element to be zeroed.
        use_batch_norm (bool): Whether to use batch normalization in the classification head
        initialization (str): Initialization scheme for the linear layersChoices are: kaiming xavier random
        target_range (Union[List, NoneType]): The range in which we should limit the output variable. Typically used for Regression problems. If left empty, will not apply any restrictions
        
        task (str): Specify whether the problem is regression of classification.Choices are: regression classification
        learning_rate (float): The learning rate of the model
        loss (Union[str, NoneType]): The loss function to be applied. By Default it is MSELoss for regression and CrossEntropyLoss for classification. Unless you are sure what you are doing, leave it at MSELoss or L1Loss for regression and CrossEntropyLoss for classification
        metrics (Union[List[str], NoneType]): the list of metrics you need to track during training. The metrics should be one of the metrics mplemented in PyTorch Lightning. By default, it is Accuracy if classification and MeanSquaredLogError for regression
        metrics_params (Union[List, NoneType]): The parameters to be passed to the Metrics initialized

    Raises:
        NotImplementedError: [description]
    """

    n_d: int = field(
        default=8,
        metadata={
            "help": "Dimension of the prediction  layer (usually between 4 and 64)"
        },
    )
    n_a: int = field(
        default=8,
        metadata={
            "help": "Dimension of the attention  layer (usually between 4 and 64)"
        },
    )
    n_steps: int = field(
        default=3,
        metadata={
            "help": "Number of sucessive steps in the newtork (usually betwenn 3 and 10)"
        },
    )
    gamma: float = field(
        default=1.3,
        metadata={
            "help": "Float above 1, scaling factor for attention updates (usually betwenn 1.0 to 2.0)"
        },
    )
    embedding_dims: Optional[List[int]] = field(
        default=None,
        metadata={
            "help": "The dimensions of the embedding for each categorical column as a list of tuples (cardinality, embedding_dim). If left empty, will infer using the cardinality of the categorical column using the rule min(50, (x + 1) // 2)"
        },
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
        metadata={
            "help": "Batch size for Ghost Batch Normalization"
        },
    )
    mask_type: str = field(
        default="sparsemax",
        metadata={
            "help": "Either 'sparsemax' or 'entmax' : this is the masking function to use",
            "choices": ["sparsemax", "entmax"],
        },
    )
    # TODO  Currently only works for single regression tasks
    target_range: Optional[List] = field(
        default=None,
        metadata={
            "help": "The range in which we should limit the output variable. Typically used for Regression problems. If left empty, will not apply any restrictions"
        },
    )
    _module_src: str = field(
        default="tabnet"
    )
    _model_name: str = field(
        default="TabNetModel"
    )

    # def __post_init__(self):
    #     assert self._module_src == "category_embedding", "Do not change attributes starting with _"
    #     assert self._model_name == "CategoryEmbeddingModel", "Do not change attributes starting wtih _"


# conf = OmegaConf.structured(ModelConfig(task='regression', loss="custom"))
# print(OmegaConf.to_yaml(conf))
# desc = "Optimizer and Learning Rate Scheduler configuration."
# doc_str = f"{desc}\nArgs:"
# for key in OptimizerConfig.__dataclass_fields__.keys():
#     atr = OptimizerConfig.__dataclass_fields__[key]
#     if atr.init:
#         type = str(atr.type).replace("<class '","").replace("'>","").replace("typing.","")
#         help_str = atr.metadata["help"]
#         if "choices" in atr.metadata.keys():
#             help_str += f'Choices are: {" ".join([str(ch) for ch in atr.metadata["choices"]])}'
#         doc_str+=f'\n\t\t{key} ({type}): {help_str}'

# print(doc_str)

# config = Config()
# config.parse_args(["--overfit-batches","10"])
# config.generate_yaml_config()
# print(config.overfit_batches)
# config = Config.read_from_yaml("run_config.yml")
# print(config.overfit_batches)
# print(config.profiler)
# parser = ArgumentParser(config)
# parser.parse_args(["--overfit-batches","10"])