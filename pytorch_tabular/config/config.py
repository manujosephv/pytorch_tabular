# Pytorch Tabular
# Author: Manu Joseph <manujoseph@gmail.com>
# For license information, see LICENSE.TXT
"""Config"""
from dataclasses import MISSING, dataclass, field
from typing import List, Optional, Tuple
import os
from omegaconf import OmegaConf

# from omegaconf.dictconfig import DictConfig


def _read_yaml(filename):
    import yaml
    import re

    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        "tag:yaml.org,2002:float",
        re.compile(
            """^(?:
        [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$""",
            re.X,
        ),
        list("-+0123456789."),
    )
    with open(filename, "r") as file:
        config = yaml.load(file, loader)
    return config


def _validate_choices(cls):
    for key in cls.__dataclass_fields__.keys():
        atr = cls.__dataclass_fields__[key]
        if atr.init:
            if "choices" in atr.metadata.keys():
                if getattr(cls, key) not in atr.metadata.get("choices"):
                    raise ValueError(
                        f"{getattr(cls, key)} is not a valid choice for {key}. Please choose from on of the following: {atr.metadata['choices']}"
                    )


@dataclass
class DataConfig:
    """Data configuration.

    Args:
        target (List[str]): A list of strings with the names of the target column(s)

        continuous_cols (List[str]): Column names of the numeric fields. Defaults to []

        categorical_cols (List): Column names of the categorical fields to treat differently. Defaults to []

        date_columns (List): (Column names, Freq) tuples of the date fields. For eg. a field named introduction_date
            and with a monthly frequency should have an entry ('intro_date','M'}

        encode_date_columns (bool): Whether or not to encode the derived variables from date

        validation_split (Union[float, NoneType]): Percentage of Training rows to keep aside as validation.
            Used only if Validation Data is not given separately

        continuous_feature_transform (Union[str, NoneType]): Whether or not to transform the features before modelling.
            By default it is turned off.Choices are: None "yeo-johnson" "box-cox" "quantile_normal" "quantile_uniform"

        normalize_continuous_features (bool): Flag to normalize the input features(continuous)

        quantile_noise (int): NOT IMPLEMENTED. If specified fits QuantileTransformer on data with added
            gaussian noise with std = :quantile_noise: * data.std ; this will cause discrete values to be more separable.
            Please note that this transformation does NOT apply gaussian noise to the resulting data,
            the noise is only applied for QuantileTransformer

        num_workers (Union[int, NoneType]): The number of workers used for data loading. For Windows always set to 0
    """

    target: List[str] = field(
        default=MISSING,
        metadata={"help": "A list of strings with the names of the target column(s)"},
    )
    continuous_cols: List[str] = field(
        default_factory=list,
        metadata={"help": "Column names of the numeric fields. Defaults to []"},
    )
    categorical_cols: List = field(
        default_factory=list,
        metadata={
            "help": "Column names of the categorical fields to treat differently. Defaults to []"
        },
    )
    date_columns: List = field(
        default_factory=list,
        metadata={
            "help": "(Column names, Freq) tuples of the date fields. For eg. a field named introduction_date and with a monthly frequency should have an entry ('intro_date','M'}"
        },
    )

    encode_date_columns: bool = field(
        default=True,
        metadata={"help": "Whether or not to encode the derived variables from date"},
    )
    validation_split: Optional[float] = field(
        default=0.2,
        metadata={
            "help": "Percentage of Training rows to keep aside as validation. Used only if Validation Data is not given separately"
        },
    )
    continuous_feature_transform: Optional[str] = field(
        default=None,
        metadata={
            "help": "Whether or not to transform the features before modelling. By default it is turned off.",
            "choices": [
                None,
                "yeo-johnson",
                "box-cox",
                "quantile_normal",
                "quantile_uniform",
            ],
        },
    )
    normalize_continuous_features: bool = field(
        default=True,
        metadata={"help": "Flag to normalize the input features(continuous)"},
    )
    quantile_noise: int = field(
        default=0,
        metadata={
            "help": "NOT IMPLEMENTED. If specified fits QuantileTransformer on data with added gaussian noise with std = :quantile_noise: * data.std ; this will cause discrete values to be more separable. Please not that this transformation does NOT apply gaussian noise to the resulting data, the noise is only applied for QuantileTransformer"
        },
    )
    num_workers: Optional[int] = field(
        default=0,
        metadata={
            "help": "The number of workers used for data loading. For windows always set to 0"
        },
    )

    categorical_dim: int = field(init=False)
    continuous_dim: int = field(init=False)
    # output_dim: int = field(init=False)

    def __post_init__(self):
        assert (
            len(self.categorical_cols)
            + len(self.continuous_cols)
            + len(self.date_columns)
            > 0
        ), "There should be at-least one feature defined in categorical, continuous, or date columns"
        self.categorical_dim = (
            len(self.categorical_cols) if self.categorical_cols is not None else 0
        )
        self.continuous_dim = (
            len(self.continuous_cols) if self.continuous_cols is not None else 0
        )
        _validate_choices(self)


@dataclass
class TrainerConfig:
    """Trainer configuration
    Args:
        batch_size (int): Number of samples in each batch of training

        fast_dev_run (bool): Quick Debug Run of Val

        max_epochs (int): Maximum number of epochs to be run

        min_epochs (int): Minimum number of epochs to be run

        gpus (int): The index of the GPU to be used. If `None`, will use CPU

        accumulate_grad_batches (int): Accumulates grads every k batches or as set up in the dict.
            Trainer also calls optimizer.step() for the last indivisible step number.

        auto_lr_find (bool): Runs a learning rate finder algorithm (see this paper) when calling trainer.tune(),
            to find optimal initial learning rate.

        check_val_every_n_epoch (int): Check val every n train epochs.

        gradient_clip_val (float): Gradient clipping value

        overfit_batches (float): Uses this much data of the training set. If nonzero, will use the same training set
            for validation and testing. If the training dataloaders have shuffle=True, Lightning will automatically disable it.
            Useful for quickly debugging or trying to overfit on purpose.

        profiler (Union[str, NoneType]): To profile individual steps during training and assist in identifying bottlenecks.
            Choices are: 'None' 'simple' 'advanced'

        early_stopping (str): The loss/metric that needed to be monitored for early stopping. If None, there will be no early stopping

        early_stopping_min_delta (float): The minimum delta in the loss/metric which qualifies as an improvement in early stopping

        early_stopping_mode (str): The direction in which the loss/metric should be optimized. Choices are `max` and `min`

        early_stopping_patience (int): The number of epochs to wait until there is no further improvements in loss/metric

        checkpoints (str): The loss/metric that needed to be monitored for checkpoints. If None, there will be no checkpoints

        checkpoints_path (str): The path where the saved models will be

        checkpoints_name(Optional[str]): The name under which the models will be saved.
            If left blank, first it will look for `run_name` in experiment_config and if that is also None
            then it will use a generic name like task_version.

        checkpoints_mode (str): The direction in which the loss/metric should be optimized

        checkpoints_save_top_k (int): The number of best models to save

        load_best (bool): Flag to load the best model saved during training

        track_grad_norm (int): Track and Log Gradient Norms in the logger.
            -1 by default means no tracking. 1 for the L1 norm, 2 for L2 norm, etc.
    """

    batch_size: int = field(
        default=64, metadata={"help": "Number of samples in each batch of training"}
    )
    fast_dev_run: bool = field(
        default=False, metadata={"help": "Quick Debug Run of Val"}
    )
    max_epochs: int = field(
        default=10, metadata={"help": "Maximum number of epochs to be run"}
    )
    min_epochs: int = field(
        default=1, metadata={"help": "Minimum number of epochs to be run"}
    )
    gpus: Optional[int] = field(
        default=None,
        metadata={"help": "The index of the GPU to be used. If None, will use CPU"},
    )
    accumulate_grad_batches: int = field(
        default=1,
        metadata={
            "help": "Accumulates grads every k batches or as set up in the dict. Trainer also calls optimizer.step() for the last indivisible step number."
        },
    )
    auto_lr_find: bool = field(
        default=False,
        metadata={
            "help": "Runs a learning rate finder algorithm (see this paper) when calling trainer.tune(), to find optimal initial learning rate."
        },
    )
    check_val_every_n_epoch: int = field(
        default=1, metadata={"help": "Check val every n train epochs."}
    )
    gradient_clip_val: float = field(
        default=0.0, metadata={"help": "Gradient clipping value"}
    )
    overfit_batches: float = field(
        default=0.0,
        metadata={
            "help": "Uses this much data of the training set. If nonzero, will use the same training set for validation and testing. If the training dataloaders have shuffle=True, Lightning will automatically disable it. Useful for quickly debugging or trying to overfit on purpose."
        },
    )
    profiler: Optional[str] = field(
        default=None,
        metadata={
            "help": "To profile individual steps during training and assist in identifying bottlenecks. None, simple or advanced",
            "choices": [None, "simple", "advanced"],
        },
    )
    early_stopping: Optional[str] = field(
        default="valid_loss",
        metadata={
            "help": "The loss/metric that needed to be monitored for early stopping. If None, there will be no early stopping"
        },
    )
    early_stopping_min_delta: float = field(
        default=0.001,
        metadata={
            "help": "The minimum delta in the loss/metric which qualifies as an improvement in early stopping"
        },
    )
    early_stopping_mode: str = field(
        default="min",
        metadata={
            "help": "The direction in which the loss/metric should be optimized",
            "choices": ["max", "min"],
        },
    )
    early_stopping_patience: int = field(
        default=3,
        metadata={
            "help": "The number of epochs to wait until there is no further improvements in loss/metric"
        },
    )
    checkpoints: Optional[str] = field(
        default="valid_loss",
        metadata={
            "help": "The loss/metric that needed to be monitored for checkpoints. If None, there will be no checkpoints"
        },
    )
    checkpoints_path: str = field(
        default="saved_models",
        metadata={"help": "The path where the saved models will be"},
    )
    checkpoints_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name under which the models will be saved. If left blank, first it will look for `run_name` in experiment_config and if that is also None then it will use a generic name like task_version."
        }
    )
    checkpoints_mode: str = field(
        default="min",
        metadata={"help": "The direction in which the loss/metric should be optimized"},
    )
    checkpoints_save_top_k: int = field(
        default=1,
        metadata={"help": "The number of best models to save"},
    )
    load_best: bool = field(
        default=True,
        metadata={"help": "Flag to load the best model saved during training"},
    )
    track_grad_norm: int = field(
        default=-1,
        metadata={
            "help": "Track and Log Gradient Norms in the logger. -1 by default means no tracking. 1 for the L1 norm, 2 for L2 norm, etc."
        },
    )

    def __post_init__(self):
        _validate_choices(self)


@dataclass
class ExperimentConfig:
    """Experiment configuration. Experiment Tracking with WandB and Tensorboard
    Args:
            project_name (str): The name of the project under which all runs will be logged.
                For Tensorboard this defines the folder under which the logs will be saved and
                for W&B it defines the project name.

            run_name (Union[str, NoneType]): The name of the run; a specific identifier to
                recognize the run. If left blank, will be assigned a auto-generated name

            exp_watch (Union[str, NoneType]): The level of logging required.
                Can be `gradients`, `parameters`, `all` or `None`. Defaults to None

            log_target (str): Determines where logging happens - Tensorboard or W&BChoices are: wandb tensorboard

            log_logits (bool): Turn this on to log the logits as a histogram in W&B

            exp_log_freq (int): step count between logging of gradients and parameters.

            _exp_version_manager (str): The location of the yaml file which manages versions of experiments
    """

    project_name: str = field(
        default=MISSING,
        metadata={
            "help": "The name of the project under which all runs will be logged. For Tensorboard this defines the folder under which the logs will be saved and for W&B it defines the project name"
        },
    )

    run_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the run; a specific identifier to recognize the run. If left blank, will be assigned a auto-generated name"
        },
    )
    exp_watch: Optional[str] = field(
        default=None,
        metadata={
            "help": "The level of logging required.  Can be `gradients`, `parameters`, `all` or `None`. Defaults to None",
            "choices": ["gradients", "parameters", "all", None],
        },
    )

    log_target: str = field(
        default="tensorboard",
        metadata={
            "help": "Determines where logging happens - Tensorboard or W&B",
            "choices": ["wandb", "tensorboard"],
        },
    )
    log_logits: bool = field(
        default=False,
        metadata={"help": "Turn this on to log the logits as a histogram in W&B"},
    )

    exp_log_freq: int = field(
        default=100,
        metadata={"help": "step count between logging of gradients and parameters."},
    )

    def __post_init__(self):
        _validate_choices(self)
        if self.log_target == "wandb":
            try:
                import wandb
            except ImportError:
                raise ImportError(
                    "No W&B installation detected. `pip install wandb` to install W&B if you set log_target as `wandb`"
                )


@dataclass
class OptimizerConfig:
    """Optimizer and Learning Rate Scheduler configuration.
    Args:
        optimizer (str): Any of the standard optimizers from 
            [torch.optim](https://pytorch.org/docs/stable/optim.html#algorithms). Defaults to `Adam`"

        optimizer_params (dict): The parameters for the optimizer. If left blank, will use default parameters.

        lr_scheduler (Union[str, NoneType]): The name of the LearningRateScheduler to use, if any, from [torch.optim.lr_scheduler](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate). 
            If None, will not use any scheduler. Defaults to `None`

        lr_scheduler_params (Union[dict, NoneType]): The parameters for the LearningRateScheduler. If left blank, will use default parameters.

        lr_scheduler_monitor_metric (Union[str, NoneType]): Used with `ReduceLROnPlateau`, where the plateau is decided based on this metric
    """

    optimizer: str = field(
        default="Adam",
        metadata={"help": "Any of the standard optimizers from [torch.optim](https://pytorch.org/docs/stable/optim.html#algorithms)."},
    )
    optimizer_params: dict = field(
        default_factory=lambda: {"weight_decay": 0, "amsgrad": False},
        metadata={
            "help": "The parameters for the optimizer. If left blank, will use default parameters."
        },
    )
    lr_scheduler: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the LearningRateScheduler to use, if any, from [torch.optim.lr_scheduler](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate). If None, will not use any scheduler. Defaults to `None`",
        },
    )
    lr_scheduler_params: Optional[dict] = field(
        default_factory=lambda: {},
        metadata={
            "help": "The parameters for the LearningRateScheduler. If left blank, will use default parameters."
        },
    )

    lr_scheduler_monitor_metric: Optional[str] = field(
        default="valid_loss",
        metadata={
            "help": "Used with ReduceLROnPlateau, where the plateau is decided based on this metric"
        },
    )

    @staticmethod
    def read_from_yaml(filename: str = "config/optimizer_config.yml"):
        config = _read_yaml(filename)
        if config["lr_scheduler_params"] is None:
            config["lr_scheduler_params"] = {}
        return OptimizerConfig(**config)


class ExperimentRunManager:
    def __init__(
        self,
        exp_version_manager: str = ".tmp/exp_version_manager.yml",
    ) -> None:
        """The manages the versions of the experiments based on the name. It is a simple dictionary(yaml) based lookup.
        Primary purpose is to avoid overwriting of saved models while runing the training without changing the experiment name.

        Args:
            exp_version_manager (str, optional): The path of the yml file which acts as version control.
            Defaults to ".tmp/exp_version_manager.yml".
        """
        super().__init__()
        self._exp_version_manager = exp_version_manager
        if os.path.exists(exp_version_manager):
            self.exp_version_manager = OmegaConf.load(exp_version_manager)
        else:
            self.exp_version_manager = OmegaConf.create({})
            os.makedirs(".tmp", exist_ok=True)
            with open(self._exp_version_manager, "w") as file:
                OmegaConf.save(config=self.exp_version_manager, f=file)

    def update_versions(self, name):
        if name in self.exp_version_manager.keys():
            uid = self.exp_version_manager[name] + 1
        else:
            uid = 1
        self.exp_version_manager[name] = uid
        with open(self._exp_version_manager, "w") as file:
            OmegaConf.save(config=self.exp_version_manager, f=file)
        return uid


@dataclass
class ModelConfig:
    """Base Model configuration
    Args:
        task (str): Specify whether the problem is regression of classification.Choices are: regression classification

        learning_rate (float): The learning rate of the model

        loss (Union[str, NoneType]): The loss function to be applied.
            By Default it is MSELoss for regression and CrossEntropyLoss for classification.
            Unless you are sure what you are doing, leave it at MSELoss or L1Loss for regression and CrossEntropyLoss for classification

        metrics (Union[List[str], NoneType]): the list of metrics you need to track during training.
            The metrics should be one of the metrics implemented in PyTorch Lightning.
            By default, it is Accuracy if classification and MeanSquaredLogError for regression

        metrics_params (Union[List, NoneType]): The parameters to be passed to the metrics function

        target_range (Union[List, NoneType]): The range in which we should limit the output variable. Currently ignored for multi-target regression
            Typically used for Regression problems. If left empty, will not apply any restrictions

    Raises:
        NotImplementedError: Raises an error if task is not regression or classification
    """

    task: str = field(
        # default="regression",
        metadata={
            "help": "Specify whether the problem is regression of classification.",
            "choices": ["regression", "classification"],
        }
    )
    learning_rate: float = field(
        default=1e-3, metadata={"help": "The learning rate of the model"}
    )
    loss: Optional[str] = field(
        default=None,
        metadata={
            "help": "The loss function to be applied. By Default it is MSELoss for regression and CrossEntropyLoss for classification. Unless you are sure what you are doing, leave it at MSELoss or L1Loss for regression and CrossEntropyLoss for classification"
        },
    )
    metrics: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "the list of metrics you need to track during training. The metrics should be one of the functional metrics implemented in PyTorch Lightning. By default, it is accuracy if classification and MeanSquaredLogError for regression"
        },
    )
    metrics_params: Optional[List] = field(
        default=None,
        metadata={"help": "The parameters to be passed to the metrics function"},
    )
    target_range: Optional[List] = field(
        default=None,
        metadata={
            "help": "The range in which we should limit the output variable. Currently ignored for multi-target regression. Typically used for Regression problems. If left empty, will not apply any restrictions"
        },
    )

    def __post_init__(self):
        if self.task == "regression":
            self.loss = "MSELoss" if self.loss is None else self.loss
            self.metrics = (
                ["mean_squared_error"] if self.metrics is None else self.metrics
            )
            self.metrics_params = [{}]
        elif self.task == "classification":
            self.loss = "CrossEntropyLoss" if self.loss is None else self.loss
            self.metrics = ["accuracy"] if self.metrics is None else self.metrics
            self.metrics_params = (
                [{} for _ in self.metrics]
                if self.metrics_params is None
                else self.metrics_params
            )
        else:
            raise NotImplementedError(
                f"{self.task} is not a valid task. Should be one of {self.__dataclass_fields__['task'].metadata['choices']}"
            )
        assert len(self.metrics) == len(
            self.metrics_params
        ), "metrics and metric_params should have same length"
        _validate_choices(self)


# conf = OmegaConf.structured(ModelConfig(task='regression', loss="custom"))
# print(OmegaConf.to_yaml(conf))
# cls = ModelConfig
# desc = "Configuration for Data."
# doc_str = f"{desc}\nArgs:"
# for key in cls.__dataclass_fields__.keys():
#     atr = cls.__dataclass_fields__[key]
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
