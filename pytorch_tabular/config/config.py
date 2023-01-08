# Pytorch Tabular
# Author: Manu Joseph <manujoseph@gmail.com>
# For license information, see LICENSE.TXT
"""Config"""
import logging
import os
import re
import warnings
from dataclasses import dataclass, field, MISSING
from typing import Any, Dict, Iterable, List, Optional

from omegaconf import OmegaConf

from pytorch_tabular.models.common import heads

logger = logging.getLogger(__name__)


def _read_yaml(filename):
    import re

    import yaml

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
        target (Union[List[str], NoneType]): A list of strings with the names of the target column(s).
            It is mandatory for all except SSL tasks.

        continuous_cols (List): Column names of the numeric fields. Defaults to []

        categorical_cols (List): Column names of the categorical fields to treat differently. Defaults to []

        date_columns (List): (Column names, Freq) tuples of the date fields.
            For eg. a field named introduction_date and with a monthly frequency should have an entry ('intro_date','M'}

        encode_date_columns (bool): Whether or not to encode the derived variables from date

        validation_split (Union[float, NoneType]): Percentage of Training rows to keep aside as validation.
            Used only if Validation Data is not given separately

        continuous_feature_transform (Union[str, NoneType]): Whether or not to transform the features before modelling.
            By default it is turned off.. Choices are: [`None`,`yeo-johnson`,`box-cox`,`quantile_normal`,`quantile_uniform`].

        normalize_continuous_features (bool): Flag to normalize the input features(continuous)

        quantile_noise (int): NOT IMPLEMENTED. If specified fits QuantileTransformer on data with added gaussian noise
            with std = :quantile_noise: * data.std ; this will cause discrete values to be more separable.
            Please not that this transformation does NOT apply gaussian noise to the resulting data, the noise is only
            applied for QuantileTransformer

        num_workers (Union[int, NoneType]): The number of workers used for data loading. For Windows always set to 0 to avoid errors.

        pin_memory (bool): Whether or not to pin memory for data loading.

        handle_unknown_categories (bool): Whether or not to handle unknown or new values in categorical columns as unknown

        handle_missing_values (bool): Whether or not to handle missing values in categorical columns as unknown
    """

    target: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "A list of strings with the names of the target column(s). It is mandatory for all except SSL tasks."
        },
    )
    continuous_cols: List = field(
        default_factory=list,
        metadata={"help": "Column names of the numeric fields. Defaults to []"},
    )
    categorical_cols: List = field(
        default_factory=list,
        metadata={"help": "Column names of the categorical fields to treat differently. Defaults to []"},
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
        metadata={"help": "The number of workers used for data loading. For windows always set to 0"},
    )
    pin_memory: bool = field(
        default=True,
        metadata={"help": "Whether or not to pin memory for data loading."},
    )
    handle_unknown_categories: bool = field(
        default=True,
        metadata={"help": "Whether or not to handle unknown or new values in categorical columns as unknown"},
    )
    handle_missing_values: bool = field(
        default=True,
        metadata={"help": "Whether or not to handle missing values in categorical columns as unknown"},
    )
    # categorical_dim: int = field(init=False)
    # continuous_dim: int = field(init=False)
    # output_dim: int = field(init=False)

    def __post_init__(self):
        assert (
            len(self.categorical_cols) + len(self.continuous_cols) + len(self.date_columns) > 0
        ), "There should be at-least one feature defined in categorical, continuous, or date columns"
        # self.categorical_dim = (
        #     len(self.categorical_cols) if self.categorical_cols is not None else 0
        # )
        # self.continuous_dim = (
        #     len(self.continuous_cols) if self.continuous_cols is not None else 0
        # )
        _validate_choices(self)
        if os.name == "nt" and self.num_workers != 0:
            print("Windows does not support num_workers > 0. Setting num_workers to 0")
            self.num_workers = 0


@dataclass
class InferredConfig:
    """
    Configuration inferred from the data during `fit` of the TabularDatamodule

    Args:
        categorical_dim (int): The number of categorical features

        continuous_dim (int): The number of continuous features

        output_dim (Optional[int], optional): The number of output targets. Defaults to None.

        categorical_cardinality (Optional[List[int]], optional): The number of unique values in categorical features.
            Defaults to None.

        embedding_dims (Optional[List], optional): The dimensions of the embedding for each categorical column as a
            list of tuples (cardinality, embedding_dim). Defaults to None.
    """

    categorical_dim: int = field(
        metadata={"help": "The number of categorical features"},
    )
    continuous_dim: int = field(
        metadata={"help": "The number of continuous features"},
    )
    output_dim: Optional[int] = field(
        default=None,
        metadata={"help": "The number of output targets"},
    )
    categorical_cardinality: Optional[List[int]] = field(
        default=None,
        metadata={"help": "The number of unique values in categorical features"},
    )
    embedding_dims: Optional[List] = field(
        default=None,
        metadata={
            "help": "The dimensions of the embedding for each categorical column as a list of tuples "
            "(cardinality, embedding_dim)."
        },
    )
    embedded_cat_dim: int = field(
        init=False,
        metadata={"help": "The number of features or dimensions of the embedded categorical features"},
    )

    def __post_init__(self):
        if self.embedding_dims is not None:
            assert all(
                [(isinstance(t, Iterable) and len(t) == 2) for t in self.embedding_dims]
            ), "embedding_dims must be a list of tuples (cardinality, embedding_dim)"
            self.embedded_cat_dim = sum([t[1] for t in self.embedding_dims])
        else:
            self.embedded_cat_dim = 0


@dataclass
class TrainerConfig:
    """Trainer configuration
    Args:
        batch_size (int): Number of samples in each batch of training

        fast_dev_run (bool): runs n if set to ``n`` (int) else 1 if set to ``True`` batch(es) of train, val
                and test to find any bugs (ie: a sort of unit test).

        max_epochs (int): Maximum number of epochs to be run

        min_epochs (Union[int, NoneType]): Force training for at least these many epochs. 1 by default

        max_time (Union[int, NoneType]): Stop training after this amount of time has passed. Disabled by
                default (None)

        gpus (Union[int, NoneType]): Number of gpus to train on (int). -1 uses all available GPUs. By
                default uses CPU (None)

        accumulate_grad_batches (int): Accumulates grads every k batches or as set up in the dict. Trainer
                also calls optimizer.step() for the last indivisible step number.

        auto_lr_find (bool): Runs a learning rate finder algorithm (see this paper) when calling
                trainer.tune(), to find optimal initial learning rate.

        auto_select_gpus (bool): If enabled and `gpus` is an integer, pick available gpus automatically.
                This is especially useful when GPUs are configured to be in 'exclusive mode', such that only one
                process at a time can access them.

        check_val_every_n_epoch (int): Check val every n train epochs.

        gradient_clip_val (float): Gradient clipping value

        overfit_batches (float): Uses this much data of the training set. If nonzero, will use the same
                training set for validation and testing. If the training dataloaders have shuffle=True, Lightning
                will automatically disable it. Useful for quickly debugging or trying to overfit on purpose.

        deterministic (bool): If true enables cudnn.deterministic. Might make your system slower, but
                ensures reproducibility.

        profiler (Union[str, NoneType]): To profile individual steps during training and assist in
                identifying bottlenecks. None, simple or advanced, pytorch. Choices are:
                [`None`,`simple`,`advanced`,`pytorch`].

        early_stopping (Union[str, NoneType]): The loss/metric that needed to be monitored for early
                stopping. If None, there will be no early stopping

        early_stopping_min_delta (float): The minimum delta in the loss/metric which qualifies as an
                improvement in early stopping

        early_stopping_mode (str): The direction in which the loss/metric should be optimized. Choices are:
                [`max`,`min`].

        early_stopping_patience (int): The number of epochs to wait until there is no further improvements
                in loss/metric

        checkpoints (Union[str, NoneType]): The loss/metric that needed to be monitored for checkpoints. If
                None, there will be no checkpoints

        checkpoints_path (str): The path where the saved models will be

        checkpoints_every_n_epochs (int): Number of training steps between checkpoints

        checkpoints_name (Union[str, NoneType]): The name under which the models will be saved. If left
                blank, first it will look for `run_name` in experiment_config and if that is also None then it
                will use a generic name like task_version.

        checkpoints_mode (str): The direction in which the loss/metric should be optimized

        checkpoints_save_top_k (int): The number of best models to save

        load_best (bool): Flag to load the best model saved during training

        track_grad_norm (int): Track and Log Gradient Norms in the logger. -1 by default means no tracking.
                1 for the L1 norm, 2 for L2 norm, etc.

        progress_bar (str): Progress bar type. Can be one of: `none`, `simple`, `rich`. Defaults to `rich`.

        precision (int): Precision of the model. Can be one of: `32`, `16`, `64`. Defaults to `32`..
                Choices are: [`32`,`16`,`64`].

        seed (int): Seed for random number generators. Defaults to 42

        trainer_kwargs (Dict[str, Any]): Additional kwargs to be passed to PyTorch Lightning Trainer. See
                https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.trainer.html#pytorch_lightning.trainer.Trainer
    """

    batch_size: int = field(default=64, metadata={"help": "Number of samples in each batch of training"})
    fast_dev_run: bool = field(
        default=False,
        metadata={
            "help": "runs n if set to ``n`` (int) else 1 if set to ``True`` batch(es) of train, val and test to find any bugs (ie: a sort of unit test)."
        },
    )
    max_epochs: int = field(default=10, metadata={"help": "Maximum number of epochs to be run"})
    min_epochs: Optional[int] = field(
        default=1,
        metadata={"help": "Force training for at least these many epochs. 1 by default"},
    )
    max_time: Optional[int] = field(
        default=None,
        metadata={"help": "Stop training after this amount of time has passed. Disabled by default (None)"},
    )
    gpus: Optional[int] = field(
        default=None,
        metadata={
            "help": "DEPRECATED: Number of gpus to train on (int). -1 uses all available GPUs. By default uses CPU (None)"
        },
    )
    accelerator: Optional[str] = field(
        default="auto",
        metadata={
            "help": "The accelerator to use for training. Can be one of 'cpu','gpu','tpu','ipu','auto'. Defaults to 'auto'",
            "choices": ["cpu", "gpu", "tpu", "ipu", "auto"],
        },
    )
    devices: Optional[int] = field(
        default=-1,
        metadata={
            "help": "Number of devices to train on (int). -1 uses all available devices. By default uses all available devices (-1)",
        },
    )
    devices_list: Optional[List[int]] = field(
        default=None,
        metadata={
            "help": "List of devices to train on (list). If specified, takes precedence over `devices` argument. Defaults to None",
        },
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
    auto_select_gpus: bool = field(
        default=True,
        metadata={
            "help": "If enabled and `devices` is an integer, pick available gpus automatically. This is especially useful when GPUs are configured to be in 'exclusive mode', such that only one process at a time can access them."
        },
    )
    check_val_every_n_epoch: int = field(default=1, metadata={"help": "Check val every n train epochs."})
    gradient_clip_val: float = field(default=0.0, metadata={"help": "Gradient clipping value"})
    overfit_batches: float = field(
        default=0.0,
        metadata={
            "help": "Uses this much data of the training set. If nonzero, will use the same training set for validation and testing. If the training dataloaders have shuffle=True, Lightning will automatically disable it. Useful for quickly debugging or trying to overfit on purpose."
        },
    )
    deterministic: bool = field(
        default=False,
        metadata={
            "help": "If true enables cudnn.deterministic. Might make your system slower, but ensures reproducibility."
        },
    )
    profiler: Optional[str] = field(
        default=None,
        metadata={
            "help": "To profile individual steps during training and assist in identifying bottlenecks. None, simple or advanced, pytorch",
            "choices": [None, "simple", "advanced", "pytorch"],
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
        metadata={"help": "The minimum delta in the loss/metric which qualifies as an improvement in early stopping"},
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
        metadata={"help": "The number of epochs to wait until there is no further improvements in loss/metric"},
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
    checkpoints_every_n_epochs: int = field(
        default=1,
        metadata={"help": "Number of training steps between checkpoints"},
    )
    checkpoints_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name under which the models will be saved. If left blank, first it will look for `run_name` in experiment_config and if that is also None then it will use a generic name like task_version."
        },
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
    progress_bar: str = field(
        default="rich",
        metadata={"help": "Progress bar type. Can be one of: `none`, `simple`, `rich`. Defaults to `rich`."},
    )
    precision: int = field(
        default=32,
        metadata={
            "help": "Precision of the model. Can be one of: `32`, `16`, `64`. Defaults to `32`.",
            "choices": [32, 16, 64],
        },
    )
    seed: int = field(
        default=42,
        metadata={"help": "Seed for random number generators. Defaults to 42"},
    )
    trainer_kwargs: Dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "help": "Additional kwargs to be passed to PyTorch Lightning Trainer. See https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.trainer.html#pytorch_lightning.trainer.Trainer"
        },
    )

    def __post_init__(self):
        _validate_choices(self)
        if self.gpus is not None:
            warnings.warn(
                "The `gpus` argument is deprecated in favor of `accelerator` and will be removed in a future version of PyTorch Tabular. Please use `accelerator='gpu'` instead.",
                DeprecationWarning,
            )
            if self.devices is None:
                self.devices = self.gpus
            if self.accelerator is None:
                self.accelerator = "gpu"
        else:
            if self.accelerator is None:
                self.accelerator = "cpu"
        delattr(self, "gpus")
        if self.devices_list is not None:
            warnings.warn("Ignoring devices in favor of devices_list")
            self.devices = self.devices_list
        delattr(self, "devices_list")


@dataclass
class ExperimentConfig:
    """Experiment configuration. Experiment Tracking with WandB and Tensorboard
    Args:
            project_name (str): The name of the project under which all runs will be logged.
                For Tensorboard this defines the folder under which the logs will be saved and
                for W&B it defines the project name.

            run_name (Optional[str, NoneType]): The name of the run; a specific identifier to
                recognize the run. If left blank, will be assigned a auto-generated name

            exp_watch (Optional[str, NoneType]): The level of logging required.
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
                import wandb  # noqa: F401
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

        lr_scheduler (Optional[str, NoneType]): The name of the LearningRateScheduler to use, if any, from [torch.optim.lr_scheduler](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate).
            If None, will not use any scheduler. Defaults to `None`

        lr_scheduler_params (Optional[dict, NoneType]): The parameters for the LearningRateScheduler. If left blank, will use default parameters.

        lr_scheduler_monitor_metric (Optional[str, NoneType]): Used with `ReduceLROnPlateau`, where the plateau is decided based on this metric
    """

    optimizer: str = field(
        default="Adam",
        metadata={
            "help": "Any of the standard optimizers from [torch.optim](https://pytorch.org/docs/stable/optim.html#algorithms)."
        },
    )
    optimizer_params: Dict = field(
        default_factory=lambda: {"weight_decay": 0, "amsgrad": False},
        metadata={"help": "The parameters for the optimizer. If left blank, will use default parameters."},
    )
    lr_scheduler: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the LearningRateScheduler to use, if any, from [torch.optim.lr_scheduler](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate). If None, will not use any scheduler. Defaults to `None`",
        },
    )
    lr_scheduler_params: Optional[Dict] = field(
        default_factory=lambda: {},
        metadata={"help": "The parameters for the LearningRateScheduler. If left blank, will use default parameters."},
    )

    lr_scheduler_monitor_metric: Optional[str] = field(
        default="valid_loss",
        metadata={"help": "Used with ReduceLROnPlateau, where the plateau is decided based on this metric"},
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
        exp_version_manager: str = ".pt_tmp/exp_version_manager.yml",
    ) -> None:
        """The manages the versions of the experiments based on the name. It is a simple dictionary(yaml) based lookup.
        Primary purpose is to avoid overwriting of saved models while runing the training without changing the experiment name.

        Args:
            exp_version_manager (str, optional): The path of the yml file which acts as version control.
            Defaults to ".pt_tmp/exp_version_manager.yml".
        """
        super().__init__()
        self._exp_version_manager = exp_version_manager
        if os.path.exists(exp_version_manager):
            self.exp_version_manager = OmegaConf.load(exp_version_manager)
        else:
            self.exp_version_manager = OmegaConf.create({})
            os.makedirs(os.path.split(exp_version_manager)[0], exist_ok=True)
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
        task (str): Specify whether the problem is regression or classification. `backbone` is a task which
                considers the model as a backbone to generate features. Mostly used internally for SSL and related
                tasks.. Choices are: [`regression`,`classification`,`backbone`].

        head (Union[str, NoneType]): The head to be used for the model. Should be one of the heads defined
                in `pytorch_tabular.models.common.heads`. Defaults to  LinearHead. Choices are:
                [`None`,`LinearHead`,`MixtureDensityHead`].

        head_config (Union[Dict, NoneType]): The config as a dict which defines the head. If left empty,
                will be initialized as default linear head.

        embedding_dims (Union[List, NoneType]): The dimensions of the embedding for each categorical column
                as a list of tuples (cardinality, embedding_dim). If left empty, will infer using the cardinality
                of the categorical column using the rule min(50, (x + 1) // 2)

        learning_rate (float): The learning rate of the model. Defaults to 1e-3.

        loss (Union[str, NoneType]): The loss function to be applied. By Default it is MSELoss for
                regression and CrossEntropyLoss for classification. Unless you are sure what you are doing, leave
                it at MSELoss or L1Loss for regression and CrossEntropyLoss for classification

        metrics (Union[List[str], NoneType]): the list of metrics you need to track during training. The
                metrics should be one of the functional metrics implemented in ``torchmetrics.functional``. By default, it is
                accuracy if classification and mean_squared_error for regression

        metrics_params (Union[List, NoneType]): The parameters to be passed to the metrics function.
                For eg. f1_score for multi-class needs a parameter `average` to fully define the metric.

        target_range (Union[List, NoneType]): The range in which we should limit the output variable.
                Currently ignored for multi-target regression. Typically used for Regression problems. If left
                empty, will not apply any restrictions

        seed (int): The seed for reproducibility. Defaults to 42

    Raises:
        NotImplementedError: Raises an error if task is not regression or classification
    """

    task: str = field(
        # default="regression",
        metadata={
            "help": "Specify whether the problem is regression or classification. `backbone` is a task which considers the model as a backbone to generate features. Mostly used internally for SSL and related tasks.",
            "choices": ["regression", "classification", "backbone"],
        }
    )

    head: Optional[str] = field(
        default="LinearHead",
        metadata={
            "help": "The head to be used for the model. Should be one of the heads defined in `pytorch_tabular.models.common.heads`. Defaults to  LinearHead",
            "choices": [None, "LinearHead", "MixtureDensityHead"],
        },
    )

    head_config: Optional[Dict] = field(
        default_factory=lambda: {"layers": ""},
        metadata={
            "help": "The config as a dict which defines the head. If left empty, will be initialized as default linear head."
        },
    )
    embedding_dims: Optional[List] = field(
        default=None,
        metadata={
            "help": "The dimensions of the embedding for each categorical column as a list of tuples "
            "(cardinality, embedding_dim). If left empty, will infer using the cardinality of the "
            "categorical column using the rule min(50, (x + 1) // 2)"
        },
    )
    embedding_dropout: float = field(
        default=0.1,
        metadata={"help": "Dropout to be applied to the Categorical Embedding. Defaults to 0.1"},
    )
    batch_norm_continuous_input: bool = field(
        default=True,
        metadata={
            "help": "If True, we will normalize the contiinuous layer by passing it through a BatchNorm layer. DEPRECATED - Use head and head_config instead"
        },
    )

    # use_batch_norm: bool = field(
    #     default=False,
    #     metadata={
    #         "help": "Flag to include a BatchNorm layer after each Linear Layer+DropOut. Defaults to False"
    #     },
    # )
    # initialization: str = field(
    #     default="kaiming",
    #     metadata={
    #         "help": "Initialization scheme for the linear layers. Defaults to `kaiming`",
    #         "choices": ["kaiming", "xavier", "random"],
    #     },
    # )

    learning_rate: float = field(
        default=1e-3,
        metadata={"help": "The learning rate of the model. Defaults to 1e-3."},
    )
    loss: Optional[str] = field(
        default=None,
        metadata={
            "help": "The loss function to be applied. By Default it is MSELoss for regression "
            "and CrossEntropyLoss for classification. Unless you are sure what you are doing, "
            "leave it at MSELoss or L1Loss for regression and CrossEntropyLoss for classification"
        },
    )
    metrics: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "the list of metrics you need to track during training. The metrics should be one "
            "of the functional metrics implemented in ``torchmetrics``. By default, "
            "it is accuracy if classification and mean_squared_error for regression"
        },
    )
    metrics_params: Optional[List] = field(
        default=None,
        metadata={"help": "The parameters to be passed to the metrics function"},
    )
    target_range: Optional[List] = field(
        default=None,
        metadata={
            "help": "The range in which we should limit the output variable. "
            "Currently ignored for multi-target regression. Typically used for Regression problems. "
            "If left empty, will not apply any restrictions"
        },
    )
    seed: int = field(
        default=42,
        metadata={"help": "The seed for reproducibility. Defaults to 42"},
    )

    _module_src: str = field(default="models")
    _model_name: str = field(default="Model")
    _backbone_name: str = field(default="Backbone")
    _config_name: str = field(default="Config")

    def __post_init__(self):
        if self.task == "regression":
            self.loss = "MSELoss" if self.loss is None else self.loss
            self.metrics = ["mean_squared_error"] if self.metrics is None else self.metrics
            self.metrics_params = [{} for _ in self.metrics] if self.metrics_params is None else self.metrics_params
        elif self.task == "classification":
            self.loss = "CrossEntropyLoss" if self.loss is None else self.loss
            self.metrics = ["accuracy"] if self.metrics is None else self.metrics
            self.metrics_params = [{} for _ in self.metrics] if self.metrics_params is None else self.metrics_params
        elif self.task == "backbone":
            self.loss = None
            self.metrics = None
            self.metrics_params = None
            if self.head is not None:
                logger.warning("`head` is not a valid parameter for backbone task. Making `head=None`")
                self.head = None
                self.head_config = None
        else:
            raise NotImplementedError(
                f"{self.task} is not a valid task. Should be one of "
                f"{self.__dataclass_fields__['task'].metadata['choices']}"
            )
        if self.metrics is not None:
            assert len(self.metrics) == len(self.metrics_params), "metrics and metric_params should have same length"

        # if len(self.head_config) == 0:  # No Head Config provided
        #     raise ValueError("`head_config` cannot be empty")
        # else:
        if self.task != "backbone":
            assert self.head in dir(heads.blocks), f"{self.head} is not a valid head"
            _head_callable = getattr(heads.blocks, self.head)
            ideal_head_config = _head_callable._config_template
            invalid_keys = set(self.head_config.keys()) - set(ideal_head_config.__dict__.keys())
            assert len(invalid_keys) == 0, f"`head_config` has some invalid keys: {invalid_keys}"

        # For Custom models, setting these values for compatibility
        if not hasattr(self, "_config_name"):
            self._config_name = type(self).__name__
        if not hasattr(self, "_model_name"):
            self._model_name = re.sub("[Cc]onfig", "Model", self._config_name)
        if not hasattr(self, "_backbone_name"):
            self._backbone_name = re.sub("[Cc]onfig", "Backbone", self._config_name)
        _validate_choices(self)


@dataclass
class SSLModelConfig:
    """Base SSLModel Configuration
    Args:
        encoder_config (Union[ModelConfig, NoneType]): The config of the encoder to be used for
                the model. Should be one of the model configs defined in PyTorch Tabular

        decoder_config (Union[ModelConfig, NoneType]): The config of decoder to be used for the
                model. Should be one of the model configs defined in PyTorch Tabular. Defaults to nn.Identity

        embedding_dims (Union[List, NoneType]): The dimensions of the embedding for each categorical column
                as a list of tuples (cardinality, embedding_dim). If left empty, will infer using the cardinality
                of the categorical column using the rule min(50, (x + 1) // 2)

        learning_rate (float): The learning rate of the model. Defaults to 1e-3

        seed (int): The seed for reproducibility. Defaults to 42
    """

    task: str = field(init=False, default="ssl")

    encoder_config: Optional[ModelConfig] = field(
        default=None,
        metadata={
            "help": "The config of the encoder to be used for the model. Should be one of the model configs defined in PyTorch Tabular",
        },
    )

    decoder_config: Optional[ModelConfig] = field(
        default=None,
        metadata={
            "help": "The config of decoder to be used for the model. Should be one of the model configs defined in PyTorch Tabular. Defaults to nn.Identity",
        },
    )

    embedding_dims: Optional[List] = field(
        default=None,
        metadata={
            "help": "The dimensions of the embedding for each categorical column as a list of tuples "
            "(cardinality, embedding_dim). If left empty, will infer using the cardinality of the "
            "categorical column using the rule min(50, (x + 1) // 2)"
        },
    )
    embedding_dropout: float = field(
        default=0.1,
        metadata={"help": "Dropout to be applied to the Categorical Embedding. Defaults to 0.1"},
    )
    batch_norm_continuous_input: bool = field(
        default=True,
        metadata={
            "help": "If True, we will normalize the contiinuous layer by passing it through a BatchNorm layer. DEPRECATED - Use head and head_config instead"
        },
    )
    learning_rate: float = field(
        default=1e-3,
        metadata={"help": "The learning rate of the model. Defaults to 1e-3"},
    )
    seed: int = field(
        default=42,
        metadata={"help": "The seed for reproducibility. Defaults to 42"},
    )

    _module_src: str = field(default="models")
    _model_name: str = field(default="Model")
    _config_name: str = field(default="Config")

    def __post_init__(self):
        assert self.task == "ssl", f"task should be ssl, got {self.task}"
        # For Custom models, setting these values for compatibility
        if not hasattr(self, "_config_name"):
            self._config_name = type(self).__name__
        if not hasattr(self, "_model_name"):
            self._model_name = re.sub("[Cc]onfig", "Model", self._config_name)
        _validate_choices(self)


# if __name__ == "__main__":
#     import textwrap
#     print(generate_doc_dataclass(SSLModelConfig, desc="Base SSLModel Configuration"))
