# Pytorch Tabular
# Author: Manu Joseph <manujoseph@gmail.com>
# For license information, see LICENSE.TXT
"""Tabular Model."""

import html
import inspect
import json
import os
import uuid
import warnings
from collections import defaultdict
from functools import partial
from pathlib import Path
from pprint import pformat
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torchmetrics
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from pandas import DataFrame
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks.gradient_accumulation_scheduler import (
    GradientAccumulationScheduler,
)
from pytorch_lightning.tuner.tuning import Tuner
from pytorch_lightning.utilities.model_summary import summarize
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from rich import print as rich_print
from rich.pretty import pprint
from sklearn.base import TransformerMixin
from sklearn.model_selection import BaseCrossValidator, KFold, StratifiedKFold
from torch import nn

from pytorch_tabular.config import (
    DataConfig,
    ExperimentConfig,
    ExperimentRunManager,
    ModelConfig,
    OptimizerConfig,
    TrainerConfig,
)
from pytorch_tabular.config.config import InferredConfig
from pytorch_tabular.models.base_model import BaseModel, _CaptumModel, _GenericModel
from pytorch_tabular.models.common.layers.embeddings import (
    Embedding1dLayer,
    Embedding2dLayer,
    PreEncoded1dLayer,
)
from pytorch_tabular.tabular_datamodule import TabularDatamodule
from pytorch_tabular.utils import (
    OOMException,
    OutOfMemoryHandler,
    count_parameters,
    get_logger,
    getattr_nested,
    pl_load,
    suppress_lightning_logs,
)

try:
    import captum.attr

    CAPTUM_INSTALLED = True
except ImportError:
    CAPTUM_INSTALLED = False

logger = get_logger(__name__)


class TabularModel:
    def __init__(
        self,
        config: Optional[DictConfig] = None,
        data_config: Optional[Union[DataConfig, str]] = None,
        model_config: Optional[Union[ModelConfig, str]] = None,
        optimizer_config: Optional[Union[OptimizerConfig, str]] = None,
        trainer_config: Optional[Union[TrainerConfig, str]] = None,
        experiment_config: Optional[Union[ExperimentConfig, str]] = None,
        model_callable: Optional[Callable] = None,
        model_state_dict_path: Optional[Union[str, Path]] = None,
        verbose: bool = True,
        suppress_lightning_logger: bool = False,
    ) -> None:
        """The core model which orchestrates everything from initializing the datamodule, the model, trainer, etc.

        Args:
            config (Optional[Union[DictConfig, str]], optional): Single OmegaConf DictConfig object or
                the path to the yaml file holding all the config parameters. Defaults to None.

            data_config (Optional[Union[DataConfig, str]], optional):
                DataConfig object or path to the yaml file. Defaults to None.

            model_config (Optional[Union[ModelConfig, str]], optional):
                A subclass of ModelConfig or path to the yaml file.
                Determines which model to run from the type of config. Defaults to None.

            optimizer_config (Optional[Union[OptimizerConfig, str]], optional):
                OptimizerConfig object or path to the yaml file. Defaults to None.

            trainer_config (Optional[Union[TrainerConfig, str]], optional):
                TrainerConfig object or path to the yaml file. Defaults to None.

            experiment_config (Optional[Union[ExperimentConfig, str]], optional):
                ExperimentConfig object or path to the yaml file.
                If Provided configures the experiment tracking. Defaults to None.

            model_callable (Optional[Callable], optional):
                If provided, will override the model callable that will be loaded from the config.
                Typically used when providing Custom Models

            model_state_dict_path (Optional[Union[str, Path]], optional):
                If provided, will load the state dict after initializing the model from config.

            verbose (bool): turns off and on the logging. Defaults to True.

            suppress_lightning_logger (bool): If True, will suppress the default logging from PyTorch Lightning.
                Defaults to False.

        """
        super().__init__()
        if suppress_lightning_logger:
            suppress_lightning_logs()
        self.verbose = verbose
        self.exp_manager = ExperimentRunManager()
        if config is None:
            assert any(c is not None for c in (data_config, model_config, optimizer_config, trainer_config)), (
                "If `config` is None, `data_config`, `model_config`,"
                " `trainer_config`, and `optimizer_config` cannot be None"
            )
            data_config = self._read_parse_config(data_config, DataConfig)
            model_config = self._read_parse_config(model_config, ModelConfig)
            trainer_config = self._read_parse_config(trainer_config, TrainerConfig)
            optimizer_config = self._read_parse_config(optimizer_config, OptimizerConfig)
            if model_config.task != "ssl":
                assert data_config.target is not None, (
                    "`target` in data_config should not be None for" f" {model_config.task} task"
                )
            if experiment_config is None:
                if self.verbose:
                    logger.info("Experiment Tracking is turned off")
                self.track_experiment = False
                self.config = OmegaConf.merge(
                    OmegaConf.to_container(data_config),
                    OmegaConf.to_container(model_config),
                    OmegaConf.to_container(trainer_config),
                    OmegaConf.to_container(optimizer_config),
                )
            else:
                experiment_config = self._read_parse_config(experiment_config, ExperimentConfig)
                self.track_experiment = True
                self.config = OmegaConf.merge(
                    OmegaConf.to_container(data_config),
                    OmegaConf.to_container(model_config),
                    OmegaConf.to_container(trainer_config),
                    OmegaConf.to_container(experiment_config),
                    OmegaConf.to_container(optimizer_config),
                )
        else:
            self.config = config
            if hasattr(config, "log_target") and (config.log_target is not None):
                # experiment_config = OmegaConf.structured(experiment_config)
                self.track_experiment = True
            else:
                if self.verbose:
                    logger.info("Experiment Tracking is turned off")
                self.track_experiment = False

        self.run_name, self.uid = self._get_run_name_uid()
        if self.track_experiment:
            self._setup_experiment_tracking()
        else:
            self.logger = None

        self.exp_manager = ExperimentRunManager()
        if model_callable is None:
            self.model_callable = getattr_nested(self.config._module_src, self.config._model_name)
            self.custom_model = False
        else:
            self.model_callable = model_callable
            self.custom_model = True
        self.model_state_dict_path = model_state_dict_path
        self._is_config_updated_with_data = False
        self._run_validation()
        self._is_fitted = False

    @property
    def has_datamodule(self):
        if hasattr(self, "datamodule") and self.datamodule is not None:
            return True
        else:
            return False

    @property
    def has_model(self):
        if hasattr(self, "model") and self.model is not None:
            return True
        else:
            return False

    @property
    def is_fitted(self):
        return self._is_fitted

    @property
    def name(self):
        if self.has_model:
            return self.model.__class__.__name__
        else:
            return self.config._model_name

    @property
    def num_params(self):
        if self.has_model:
            return count_parameters(self.model)

    def _run_validation(self):
        """Validates the Config params and throws errors if something is wrong."""
        if self.config.task == "regression":
            if self.config.target_range is not None:
                if (
                    (len(self.config.target_range) != len(self.config.target))
                    or any(len(range_) != 2 for range_ in self.config.target_range)
                    or any(range_[0] > range_[1] for range_ in self.config.target_range)
                ):
                    raise ValueError(
                        "Targe Range, if defined, should be list tuples of length"
                        " two(min,max). The length of the list should be equal to hte"
                        " length of target columns"
                    )

    def _read_parse_config(self, config, cls):
        if isinstance(config, str):
            if os.path.exists(config):
                _config = OmegaConf.load(config)
                if cls == ModelConfig:
                    cls = getattr_nested(_config._module_src, _config._config_name)
                config = cls(
                    **{
                        k: v
                        for k, v in _config.items()
                        if (k in cls.__dataclass_fields__.keys()) and (cls.__dataclass_fields__[k].init)
                    }
                )
            else:
                raise ValueError(f"{config} is not a valid path")
        config = OmegaConf.structured(config)
        return config

    def _get_run_name_uid(self) -> Tuple[str, int]:
        """Gets the name of the experiment and increments version by 1.

        Returns:
            tuple[str, int]: Returns the name and version number

        """
        if hasattr(self.config, "run_name") and self.config.run_name is not None:
            name = self.config.run_name
        elif hasattr(self.config, "checkpoints_name") and self.config.checkpoints_name is not None:
            name = self.config.checkpoints_name
        else:
            name = self.config.task
        uid = self.exp_manager.update_versions(name)
        return name, uid

    def _setup_experiment_tracking(self):
        """Sets up the Experiment Tracking Framework according to the choices made in the Experimentconfig."""
        if self.config.log_target == "tensorboard":
            self.logger = pl.loggers.TensorBoardLogger(
                name=self.run_name,
                save_dir=self.config.project_name,
                version=self.uid,
            )
        elif self.config.log_target == "wandb":
            self.logger = pl.loggers.WandbLogger(
                name=f"{self.run_name}_{self.uid}",
                project=self.config.project_name,
                offline=False,
            )
        else:
            raise NotImplementedError(
                f"{self.config.log_target} is not implemented. Try one of [wandb," " tensorboard]"
            )

    def _prepare_callbacks(self, callbacks=None) -> List:
        """Prepares the necesary callbacks to the Trainer based on the configuration.

        Returns:
            List: A list of callbacks

        """
        callbacks = [] if callbacks is None else callbacks
        if self.config.early_stopping is not None:
            early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(
                monitor=self.config.early_stopping,
                min_delta=self.config.early_stopping_min_delta,
                patience=self.config.early_stopping_patience,
                mode=self.config.early_stopping_mode,
                **self.config.early_stopping_kwargs,
            )
            callbacks.append(early_stop_callback)
        if self.config.checkpoints:
            ckpt_name = f"{self.run_name}-{self.uid}"
            ckpt_name = ckpt_name.replace(" ", "_") + "_{epoch}-{valid_loss:.2f}"
            model_checkpoint = pl.callbacks.ModelCheckpoint(
                monitor=self.config.checkpoints,
                dirpath=self.config.checkpoints_path,
                filename=ckpt_name,
                save_top_k=self.config.checkpoints_save_top_k,
                mode=self.config.checkpoints_mode,
                every_n_epochs=self.config.checkpoints_every_n_epochs,
                **self.config.checkpoints_kwargs,
            )
            callbacks.append(model_checkpoint)
            self.config.enable_checkpointing = True
        else:
            self.config.enable_checkpointing = False
        if self.config.progress_bar == "rich" and self.config.trainer_kwargs.get("enable_progress_bar", True):
            callbacks.append(RichProgressBar())
        if self.verbose:
            logger.debug(f"Callbacks used: {callbacks}")
        return callbacks

    def _prepare_trainer(self, callbacks: List, max_epochs: int = None, min_epochs: int = None) -> pl.Trainer:
        """Prepares the Trainer object.

        Args:
            callbacks (List): A list of callbacks to be used
            max_epochs (int, optional): Maximum number of epochs to train for. Defaults to None.
            min_epochs (int, optional): Minimum number of epochs to train for. Defaults to None.

        Returns:
            pl.Trainer: A PyTorch Lightning Trainer object

        """
        if self.verbose:
            logger.info("Preparing the Trainer")
        if max_epochs is not None:
            self.config.max_epochs = max_epochs
        if min_epochs is not None:
            self.config.min_epochs = min_epochs
        # Getting Trainer Arguments from the init signature
        trainer_sig = inspect.signature(pl.Trainer.__init__)
        trainer_args = [p for p in trainer_sig.parameters.keys() if p != "self"]
        trainer_args_config = {k: v for k, v in self.config.items() if k in trainer_args}
        # For some weird reason, checkpoint_callback is not appearing in the Trainer vars
        trainer_args_config["enable_checkpointing"] = self.config.enable_checkpointing
        # turn off progress bar if progress_bar=='none'
        trainer_args_config["enable_progress_bar"] = self.config.progress_bar != "none"
        # Adding trainer_kwargs from config to trainer_args
        trainer_args_config.update(self.config.trainer_kwargs)
        if trainer_args_config["devices"] == -1:
            # Setting devices to auto if -1 so that lightning will use all available GPUs/CPUs
            trainer_args_config["devices"] = "auto"
        return pl.Trainer(
            logger=self.logger,
            callbacks=callbacks,
            **trainer_args_config,
        )

    def _check_and_set_target_transform(self, target_transform):
        if target_transform is not None:
            if isinstance(target_transform, Iterable):
                assert len(target_transform) == 2, (
                    "If `target_transform` is a tuple, it should have and only have"
                    " forward and backward transformations"
                )
            elif isinstance(target_transform, TransformerMixin):
                pass
            else:
                raise ValueError(
                    "`target_transform` should wither be an sklearn Transformer or a" " tuple of callables."
                )
        if self.config.task == "classification" and target_transform is not None:
            logger.warning("For classification task, target transform is not used. Ignoring the" " parameter")
            target_transform = None
        return target_transform

    def _prepare_for_training(self, model, datamodule, callbacks=None, max_epochs=None, min_epochs=None):
        self.callbacks = self._prepare_callbacks(callbacks)
        self.trainer = self._prepare_trainer(self.callbacks, max_epochs, min_epochs)
        self.model = model
        self.datamodule = datamodule

    @classmethod
    def _load_weights(cls, model, path: Union[str, Path]) -> None:
        """Loads the model weights in the specified directory.

        Args:
            path (str): The path to the file to load the model from

        Returns:
            None

        """
        ckpt = pl_load(path, map_location=lambda storage, loc: storage)
        model.load_state_dict(ckpt.get("state_dict") or ckpt)

    @classmethod
    def load_model(cls, dir: str, map_location=None, strict=True):
        """Loads a saved model from the directory.

        Args:
            dir (str): The directory where the model wa saved, along with the checkpoints
            map_location (Union[Dict[str, str], str, device, int, Callable, None]) : If your checkpoint
                saved a GPU model and you now load on CPUs or a different number of GPUs, use this to map
                to the new setup. The behaviour is the same as in torch.load()
            strict (bool) : Whether to strictly enforce that the keys in checkpoint_path match the keys
                returned by this module's state dict. Default: True.

        Returns:
            TabularModel (TabularModel): The saved TabularModel

        """
        config = OmegaConf.load(os.path.join(dir, "config.yml"))
        datamodule = joblib.load(os.path.join(dir, "datamodule.sav"))
        if (
            hasattr(config, "log_target")
            and (config.log_target is not None)
            and os.path.exists(os.path.join(dir, "exp_logger.sav"))
        ):
            logger = joblib.load(os.path.join(dir, "exp_logger.sav"))
        else:
            logger = None
        if os.path.exists(os.path.join(dir, "callbacks.sav")):
            callbacks = joblib.load(os.path.join(dir, "callbacks.sav"))
            # Excluding Gradient Accumulation Scheduler Callback as we are creating
            # a new one in trainer
            callbacks = [c for c in callbacks if not isinstance(c, GradientAccumulationScheduler)]
        else:
            callbacks = []
        if os.path.exists(os.path.join(dir, "custom_model_callable.sav")):
            model_callable = joblib.load(os.path.join(dir, "custom_model_callable.sav"))
            custom_model = True
        else:
            model_callable = getattr_nested(config._module_src, config._model_name)
            # model_callable = getattr(
            #     getattr(models, config._module_src), config._model_name
            # )
            custom_model = False
        inferred_config = datamodule.update_config(config)
        inferred_config = OmegaConf.structured(inferred_config)
        model_args = {
            "config": config,
            "inferred_config": inferred_config,
        }
        custom_params = joblib.load(os.path.join(dir, "custom_params.sav"))
        if custom_params.get("custom_loss") is not None:
            model_args["loss"] = "MSELoss"  # For compatibility. Not Used
        if custom_params.get("custom_metrics") is not None:
            model_args["metrics"] = ["mean_squared_error"]  # For compatibility. Not Used
            model_args["metrics_params"] = [{}]  # For compatibility. Not Used
            model_args["metrics_prob_inputs"] = [False]  # For compatibility. Not Used
        if custom_params.get("custom_optimizer") is not None:
            model_args["optimizer"] = "Adam"  # For compatibility. Not Used
        if custom_params.get("custom_optimizer_params") is not None:
            model_args["optimizer_params"] = {}  # For compatibility. Not Used

        # Initializing with default metrics, losses, and optimizers. Will revert once initialized
        try:
            model = model_callable.load_from_checkpoint(
                checkpoint_path=os.path.join(dir, "model.ckpt"),
                map_location=map_location,
                strict=strict,
                **model_args,
            )
        except RuntimeError as e:
            if (
                "Unexpected key(s) in state_dict" in str(e)
                and "loss.weight" in str(e)
                and "custom_loss.weight" in str(e)
            ):
                # Custom loss will be loaded after the model is initialized
                # continuing with strict=False
                model = model_callable.load_from_checkpoint(
                    checkpoint_path=os.path.join(dir, "model.ckpt"),
                    map_location=map_location,
                    strict=False,
                    **model_args,
                )
            else:
                raise e
        if custom_params.get("custom_optimizer") is not None:
            model.custom_optimizer = custom_params["custom_optimizer"]
        if custom_params.get("custom_optimizer_params") is not None:
            model.custom_optimizer_params = custom_params["custom_optimizer_params"]
        if custom_params.get("custom_loss") is not None:
            model.loss = custom_params["custom_loss"]
        if custom_params.get("custom_metrics") is not None:
            model.custom_metrics = custom_params.get("custom_metrics")
            model.hparams.metrics = [m.__name__ for m in custom_params.get("custom_metrics")]
            model.hparams.metrics_params = [{}]
            model.hparams.metrics_prob_input = custom_params.get("custom_metrics_prob_inputs")
        model._setup_loss()
        model._setup_metrics()
        tabular_model = cls(config=config, model_callable=model_callable)
        tabular_model.model = model
        tabular_model.custom_model = custom_model
        tabular_model.datamodule = datamodule
        tabular_model.callbacks = callbacks
        tabular_model.trainer = tabular_model._prepare_trainer(callbacks=callbacks)
        # tabular_model.trainer.model = model
        tabular_model.logger = logger
        return tabular_model

    def prepare_dataloader(
        self,
        train: DataFrame,
        validation: Optional[DataFrame] = None,
        train_sampler: Optional[torch.utils.data.Sampler] = None,
        target_transform: Optional[Union[TransformerMixin, Tuple]] = None,
        seed: Optional[int] = 42,
        cache_data: str = "memory",
    ) -> TabularDatamodule:
        """Prepares the dataloaders for training and validation.

        Args:
            train (DataFrame): Training Dataframe

            validation (Optional[DataFrame], optional):
                If provided, will use this dataframe as the validation while training.
                Used in Early Stopping and Logging. If left empty, will use 20% of Train data as validation.
                Defaults to None.

            train_sampler (Optional[torch.utils.data.Sampler], optional):
                Custom PyTorch batch samplers which will be passed to the DataLoaders.
                Useful for dealing with imbalanced data and other custom batching strategies

            target_transform (Optional[Union[TransformerMixin, Tuple(Callable)]], optional):
                If provided, applies the transform to the target before modelling and inverse the transform during
                prediction. The parameter can either be a sklearn Transformer which has an inverse_transform method, or
                a tuple of callables (transform_func, inverse_transform_func)

            seed (Optional[int], optional): Random seed for reproducibility. Defaults to 42.

            cache_data (str): Decides how to cache the data in the dataloader. If set to
                "memory", will cache in memory. If set to a valid path, will cache in that path. Defaults to "memory".
        Returns:
            TabularDatamodule: The prepared datamodule

        """
        if self.verbose:
            logger.info("Preparing the DataLoaders")
        target_transform = self._check_and_set_target_transform(target_transform)

        datamodule = TabularDatamodule(
            train=train,
            validation=validation,
            config=self.config,
            target_transform=target_transform,
            train_sampler=train_sampler,
            seed=seed,
            cache_data=cache_data,
            verbose=self.verbose,
        )
        datamodule.prepare_data()
        datamodule.setup("fit")
        return datamodule

    def prepare_model(
        self,
        datamodule: TabularDatamodule,
        loss: Optional[torch.nn.Module] = None,
        metrics: Optional[List[Callable]] = None,
        metrics_prob_inputs: Optional[List[bool]] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        optimizer_params: Dict = None,
    ) -> BaseModel:
        """Prepares the model for training.

        Args:
            datamodule (TabularDatamodule): The datamodule

            loss (Optional[torch.nn.Module], optional): Custom Loss functions which are not in standard pytorch library

            metrics (Optional[List[Callable]], optional): Custom metric functions(Callable) which has the
                signature metric_fn(y_hat, y) and works on torch tensor inputs

            metrics_prob_inputs (Optional[List[bool]], optional): This is a mandatory parameter for
                classification metrics. If the metric function requires probabilities as inputs, set this to True.
                The length of the list should be equal to the number of metrics. Defaults to None.

            optimizer (Optional[torch.optim.Optimizer], optional):
                Custom optimizers which are a drop in replacements for standard PyTorch optimizers.
                This should be the Class and not the initialized object

            optimizer_params (Optional[Dict], optional): The parameters to initialize the custom optimizer.

        Returns:
            BaseModel: The prepared model

        """
        if self.verbose:
            logger.info(f"Preparing the Model: {self.config._model_name}")
        # Fetching the config as some data specific configs have been added in the datamodule
        self.inferred_config = self._read_parse_config(datamodule.update_config(self.config), InferredConfig)
        model = self.model_callable(
            self.config,
            custom_loss=loss,  # Unused in SSL tasks
            custom_metrics=metrics,  # Unused in SSL tasks
            custom_metrics_prob_inputs=metrics_prob_inputs,  # Unused in SSL tasks
            custom_optimizer=optimizer,
            custom_optimizer_params=optimizer_params or {},
            inferred_config=self.inferred_config,
        )
        # Data Aware Initialization(for the models that need it)
        model.data_aware_initialization(datamodule)
        if self.model_state_dict_path is not None:
            self._load_weights(model, self.model_state_dict_path)
        if self.track_experiment and self.config.log_target == "wandb":
            self.logger.watch(model, log=self.config.exp_watch, log_freq=self.config.exp_log_freq)
        return model

    def train(
        self,
        model: pl.LightningModule,
        datamodule: TabularDatamodule,
        callbacks: Optional[List[pl.Callback]] = None,
        max_epochs: int = None,
        min_epochs: int = None,
        handle_oom: bool = True,
    ) -> pl.Trainer:
        """Trains the model.

        Args:
            model (pl.LightningModule): The PyTorch Lightning model to be trained.

            datamodule (TabularDatamodule): The datamodule

            callbacks (Optional[List[pl.Callback]], optional):
                List of callbacks to be used during training. Defaults to None.

            max_epochs (Optional[int]): Overwrite maximum number of epochs to be run. Defaults to None.

            min_epochs (Optional[int]): Overwrite minimum number of epochs to be run. Defaults to None.

            handle_oom (bool): If True, will try to handle OOM errors elegantly. Defaults to True.

        Returns:
            pl.Trainer: The PyTorch Lightning Trainer instance

        """
        self._prepare_for_training(model, datamodule, callbacks, max_epochs, min_epochs)
        train_loader, val_loader = (
            self.datamodule.train_dataloader(),
            self.datamodule.val_dataloader(),
        )
        self.model.train()
        if self.config.auto_lr_find and (not self.config.fast_dev_run):
            if self.verbose:
                logger.info("Auto LR Find Started")
            with OutOfMemoryHandler(handle_oom=handle_oom) as oom_handler:
                result = Tuner(self.trainer).lr_find(
                    self.model,
                    train_dataloaders=train_loader,
                    val_dataloaders=val_loader,
                )
            if oom_handler.oom_triggered:
                raise OOMException(
                    "OOM detected during LR Find. Try reducing your batch_size or the"
                    " model parameters." + "/n" + "Original Error: " + oom_handler.oom_msg
                )
            if self.verbose:
                logger.info(
                    f"Suggested LR: {result.suggestion()}. For plot and detailed"
                    " analysis, use `find_learning_rate` method."
                )
            self.model.reset_weights()
            # Parameters in models needs to be initialized again after LR find
            self.model.data_aware_initialization(self.datamodule)
        self.model.train()
        if self.verbose:
            logger.info("Training Started")
        with OutOfMemoryHandler(handle_oom=handle_oom) as oom_handler:
            self.trainer.fit(self.model, train_loader, val_loader)
        if oom_handler.oom_triggered:
            raise OOMException(
                "OOM detected during Training. Try reducing your batch_size or the"
                " model parameters."
                "/n" + "Original Error: " + oom_handler.oom_msg
            )
        self._is_fitted = True
        if self.track_experiment and self.config.log_target == "wandb":
            self.logger.experiment.unwatch(self.model)
        if self.verbose:
            logger.info("Training the model completed")
        if self.config.load_best:
            self.load_best_model()
        return self.trainer

    def fit(
        self,
        train: Optional[DataFrame],
        validation: Optional[DataFrame] = None,
        loss: Optional[torch.nn.Module] = None,
        metrics: Optional[List[Callable]] = None,
        metrics_prob_inputs: Optional[List[bool]] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        optimizer_params: Dict = None,
        train_sampler: Optional[torch.utils.data.Sampler] = None,
        target_transform: Optional[Union[TransformerMixin, Tuple]] = None,
        max_epochs: Optional[int] = None,
        min_epochs: Optional[int] = None,
        seed: Optional[int] = 42,
        callbacks: Optional[List[pl.Callback]] = None,
        datamodule: Optional[TabularDatamodule] = None,
        cache_data: str = "memory",
        handle_oom: bool = True,
    ) -> pl.Trainer:
        """The fit method which takes in the data and triggers the training.

        Args:
            train (DataFrame): Training Dataframe

            validation (Optional[DataFrame], optional):
                If provided, will use this dataframe as the validation while training.
                Used in Early Stopping and Logging. If left empty, will use 20% of Train data as validation.
                Defaults to None.

            loss (Optional[torch.nn.Module], optional): Custom Loss functions which are not in standard pytorch library

            metrics (Optional[List[Callable]], optional): Custom metric functions(Callable) which has the
                signature metric_fn(y_hat, y) and works on torch tensor inputs. y_hat is expected to be of shape
                (batch_size, num_classes) for classification and (batch_size, 1) for regression and y is expected to be
                of shape (batch_size, 1)

            metrics_prob_inputs (Optional[List[bool]], optional): This is a mandatory parameter for
                classification metrics. If the metric function requires probabilities as inputs, set this to True.
                The length of the list should be equal to the number of metrics. Defaults to None.

            optimizer (Optional[torch.optim.Optimizer], optional):
                Custom optimizers which are a drop in replacements for
                standard PyTorch optimizers. This should be the Class and not the initialized object

            optimizer_params (Optional[Dict], optional): The parameters to initialize the custom optimizer.

            train_sampler (Optional[torch.utils.data.Sampler], optional):
                Custom PyTorch batch samplers which will be passed
                to the DataLoaders. Useful for dealing with imbalanced data and other custom batching strategies

            target_transform (Optional[Union[TransformerMixin, Tuple(Callable)]], optional):
                If provided, applies the transform to the target before modelling and inverse the transform during
                prediction. The parameter can either be a sklearn Transformer
                which has an inverse_transform method, or a tuple of callables (transform_func, inverse_transform_func)

            max_epochs (Optional[int]): Overwrite maximum number of epochs to be run. Defaults to None.

            min_epochs (Optional[int]): Overwrite minimum number of epochs to be run. Defaults to None.

            seed: (int): Random seed for reproducibility. Defaults to 42.

            callbacks (Optional[List[pl.Callback]], optional):
                List of callbacks to be used during training. Defaults to None.

            datamodule (Optional[TabularDatamodule], optional): The datamodule.
                If provided, will ignore the rest of the parameters like train, test etc and use the datamodule.
                Defaults to None.

            cache_data (str): Decides how to cache the data in the dataloader. If set to
                "memory", will cache in memory. If set to a valid path, will cache in that path. Defaults to "memory".

            handle_oom (bool): If True, will try to handle OOM errors elegantly. Defaults to True.

        Returns:
            pl.Trainer: The PyTorch Lightning Trainer instance

        """
        assert self.config.task != "ssl", (
            "`fit` is not valid for SSL task. Please use `pretrain` for" " semi-supervised learning"
        )
        if metrics is not None:
            assert len(metrics) == len(
                metrics_prob_inputs or []
            ), "The length of `metrics` and `metrics_prob_inputs` should be equal"
        seed = seed or self.config.seed
        if seed:
            seed_everything(seed)
        if datamodule is None:
            datamodule = self.prepare_dataloader(
                train,
                validation,
                train_sampler,
                target_transform,
                seed,
                cache_data,
            )
        else:
            if train is not None:
                warnings.warn(
                    "train data and datamodule is provided."
                    " Ignoring the train data and using the datamodule."
                    " Set either one of them to None to avoid this warning."
                )
        model = self.prepare_model(
            datamodule,
            loss,
            metrics,
            metrics_prob_inputs,
            optimizer,
            optimizer_params or {},
        )

        return self.train(model, datamodule, callbacks, max_epochs, min_epochs, handle_oom)

    def pretrain(
        self,
        train: Optional[DataFrame],
        validation: Optional[DataFrame] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        optimizer_params: Dict = None,
        # train_sampler: Optional[torch.utils.data.Sampler] = None,
        max_epochs: Optional[int] = None,
        min_epochs: Optional[int] = None,
        seed: Optional[int] = 42,
        callbacks: Optional[List[pl.Callback]] = None,
        datamodule: Optional[TabularDatamodule] = None,
        cache_data: str = "memory",
    ) -> pl.Trainer:
        """The pretrained method which takes in the data and triggers the training.

        Args:
            train (DataFrame): Training Dataframe

            validation (Optional[DataFrame], optional): If provided, will use this dataframe as the validation while
                training. Used in Early Stopping and Logging. If left empty, will use 20% of Train data as validation.
                Defaults to None.

            optimizer (Optional[torch.optim.Optimizer], optional): Custom optimizers which are a drop in replacements
                for standard PyTorch optimizers. This should be the Class and not the initialized object

            optimizer_params (Optional[Dict], optional): The parameters to initialize the custom optimizer.

            max_epochs (Optional[int]): Overwrite maximum number of epochs to be run. Defaults to None.

            min_epochs (Optional[int]): Overwrite minimum number of epochs to be run. Defaults to None.

            seed: (int): Random seed for reproducibility. Defaults to 42.

            callbacks (Optional[List[pl.Callback]], optional): List of callbacks to be used during training.
                Defaults to None.

            datamodule (Optional[TabularDatamodule], optional): The datamodule. If provided, will ignore the rest of the
                parameters like train, test etc. and use the datamodule. Defaults to None.

            cache_data (str): Decides how to cache the data in the dataloader. If set to
                "memory", will cache in memory. If set to a valid path, will cache in that path. Defaults to "memory".
        Returns:
            pl.Trainer: The PyTorch Lightning Trainer instance

        """
        assert self.config.task == "ssl", (
            f"`pretrain` is not valid for {self.config.task} task. Please use `fit`" " instead."
        )
        seed = seed or self.config.seed
        if seed:
            seed_everything(seed)
        if datamodule is None:
            datamodule = self.prepare_dataloader(
                train,
                validation,
                train_sampler=None,
                target_transform=None,
                seed=seed,
                cache_data=cache_data,
            )
        else:
            if train is not None:
                warnings.warn(
                    "train data and datamodule is provided."
                    " Ignoring the train data and using the datamodule."
                    " Set either one of them to None to avoid this warning."
                )
        model = self.prepare_model(
            datamodule,
            optimizer,
            optimizer_params or {},
        )

        return self.train(model, datamodule, callbacks, max_epochs, min_epochs)

    def create_finetune_model(
        self,
        task: str,
        head: str,
        head_config: Dict,
        train: DataFrame,
        validation: Optional[DataFrame] = None,
        train_sampler: Optional[torch.utils.data.Sampler] = None,
        target_transform: Optional[Union[TransformerMixin, Tuple]] = None,
        target: Optional[str] = None,
        optimizer_config: Optional[OptimizerConfig] = None,
        trainer_config: Optional[TrainerConfig] = None,
        experiment_config: Optional[ExperimentConfig] = None,
        loss: Optional[torch.nn.Module] = None,
        metrics: Optional[List[Union[Callable, str]]] = None,
        metrics_prob_input: Optional[List[bool]] = None,
        metrics_params: Optional[Dict] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        optimizer_params: Dict = None,
        learning_rate: Optional[float] = None,
        target_range: Optional[Tuple[float, float]] = None,
        seed: Optional[int] = 42,
    ):
        """Creates a new TabularModel model using the pretrained weights and the new task and head.

        Args:
            task (str): The task to be performed. One of "regression", "classification"

            head (str): The head to be used for the model. Should be one of the heads defined
                in `pytorch_tabular.models.common.heads`. Defaults to  LinearHead. Choices are:
                [`None`,`LinearHead`,`MixtureDensityHead`].

            head_config (Dict): The config as a dict which defines the head. If left empty,
                will be initialized as default linear head.

            train (DataFrame): The training data with labels

            validation (Optional[DataFrame], optional): The validation data with labels. Defaults to None.

            train_sampler (Optional[torch.utils.data.Sampler], optional): If provided, will be used as a batch sampler
                for training. Defaults to None.

            target_transform (Optional[Union[TransformerMixin, Tuple]], optional): If provided, will be used
                to transform the target before training and inverse transform the predictions.

            target (Optional[str], optional): The target column name if not provided in the initial pretraining stage.
                Defaults to None.

            optimizer_config (Optional[OptimizerConfig], optional):
                If provided, will redefine the optimizer for fine-tuning stage. Defaults to None.

            trainer_config (Optional[TrainerConfig], optional):
                If provided, will redefine the trainer for fine-tuning stage. Defaults to None.

            experiment_config (Optional[ExperimentConfig], optional):
                If provided, will redefine the experiment for fine-tuning stage. Defaults to None.

            loss (Optional[torch.nn.Module], optional):
                If provided, will be used as the loss function for the fine-tuning.
                By default, it is MSELoss for regression and CrossEntropyLoss for classification.

            metrics (Optional[List[Callable]], optional): List of metrics (either callables or str) to be used for the
                fine-tuning stage. If str, it should be one of the functional metrics implemented in
                ``torchmetrics.functional``. Defaults to None.

            metrics_prob_input (Optional[List[bool]], optional): Is a mandatory parameter for classification metrics
                This defines whether the input to the metric function is the probability or the class.
                Length should be same as the number of metrics. Defaults to None.

            metrics_params (Optional[Dict], optional): The parameters for the metrics in the same order as metrics.
                For eg. f1_score for multi-class needs a parameter `average` to fully define the metric.
                Defaults to None.

            optimizer (Optional[torch.optim.Optimizer], optional):
                Custom optimizers which are a drop in replacements for standard PyTorch optimizers. If provided,
                the OptimizerConfig is ignored in favor of this. Defaults to None.

            optimizer_params (Dict, optional): The parameters for the optimizer. Defaults to {}.

            learning_rate (Optional[float], optional): The learning rate to be used. Defaults to 1e-3.

            target_range (Optional[Tuple[float, float]], optional): The target range for the regression task.
                Is ignored for classification. Defaults to None.

            seed (Optional[int], optional): Random seed for reproducibility. Defaults to 42.
        Returns:
            TabularModel (TabularModel): The new TabularModel model for fine-tuning

        """
        config = self.config
        optimizer_params = optimizer_params or {}
        if target is None:
            assert (
                hasattr(config, "target") and config.target is not None
            ), "`target` cannot be None if it was not set in the initial `DataConfig`"
        else:
            assert isinstance(target, list), "`target` should be a list of strings"
            config.target = target
        config.task = task
        # Add code to update configs with newly provided ones
        if optimizer_config is not None:
            for key, value in optimizer_config.__dict__.items():
                config[key] = value
            if len(optimizer_params) > 0:
                config.optimizer_params = optimizer_params
            else:
                config.optimizer_params = {}
        if trainer_config is not None:
            for key, value in trainer_config.__dict__.items():
                config[key] = value
        if experiment_config is not None:
            for key, value in experiment_config.__dict__.items():
                config[key] = value
        else:
            if self.track_experiment:
                # Renaming the experiment run so that a different log is created for finetuning
                if self.verbose:
                    logger.info("Renaming the experiment run for finetuning as" f" {config['run_name'] + '_finetuned'}")
                config["run_name"] = config["run_name"] + "_finetuned"

        config_override = {"target": target} if target is not None else {}
        config_override["task"] = task
        datamodule = self.datamodule.copy(
            train=train,
            validation=validation,
            target_transform=target_transform,
            train_sampler=train_sampler,
            seed=seed,
            config_override=config_override,
        )
        model_callable = _GenericModel
        inferred_config = OmegaConf.structured(datamodule._inferred_config)
        # Adding dummy attributes for compatibility. Not used because custom metrics are provided
        if not hasattr(config, "metrics"):
            config.metrics = "dummy"
        if not hasattr(config, "metrics_params"):
            config.metrics_params = {}
        if not hasattr(config, "metrics_prob_input"):
            config.metrics_prob_input = metrics_prob_input or [False]
        if metrics is not None:
            assert len(metrics) == len(metrics_params), "Number of metrics and metrics_params should be same"
            assert len(metrics) == len(metrics_prob_input), "Number of metrics and metrics_prob_input should be same"
            metrics = [getattr(torchmetrics.functional, m) if isinstance(m, str) else m for m in metrics]
        if task == "regression":
            loss = loss or torch.nn.MSELoss()
            if metrics is None:
                metrics = [torchmetrics.functional.mean_squared_error]
                metrics_params = [{}]
        elif task == "classification":
            loss = loss or torch.nn.CrossEntropyLoss()
            if metrics is None:
                metrics = [torchmetrics.functional.accuracy]
                metrics_params = [
                    {
                        "task": "multiclass",
                        "num_classes": inferred_config.output_dim,
                        "top_k": 1,
                    }
                ]
                metrics_prob_input = [False]
            else:
                for i, mp in enumerate(metrics_params):
                    # For classification task, output_dim == number of classses
                    metrics_params[i]["task"] = mp.get("task", "multiclass")
                    metrics_params[i]["num_classes"] = mp.get("num_classes", inferred_config.output_dim)
                    metrics_params[i]["top_k"] = mp.get("top_k", 1)
        else:
            raise ValueError(f"Task {task} not supported")
        # Forming partial callables using metrics and metric params
        metrics = [partial(m, **mp) for m, mp in zip(metrics, metrics_params)]
        self.model.mode = "finetune"
        if learning_rate is not None:
            config.learning_rate = learning_rate
        config.target_range = target_range
        model_args = {
            "backbone": self.model,
            "head": head,
            "head_config": head_config,
            "config": config,
            "inferred_config": inferred_config,
            "custom_loss": loss,
            "custom_metrics": metrics,
            "custom_metrics_prob_inputs": metrics_prob_input,
            "custom_optimizer": optimizer,
            "custom_optimizer_params": optimizer_params,
        }
        # Initializing with default metrics, losses, and optimizers. Will revert once initialized
        model = model_callable(
            **model_args,
        )
        tabular_model = TabularModel(config=config, verbose=self.verbose)
        tabular_model.model = model
        tabular_model.datamodule = datamodule
        # Setting a flag to identify this as a fine-tune model
        tabular_model._is_finetune_model = True
        return tabular_model

    def finetune(
        self,
        max_epochs: Optional[int] = None,
        min_epochs: Optional[int] = None,
        callbacks: Optional[List[pl.Callback]] = None,
        freeze_backbone: bool = False,
    ) -> pl.Trainer:
        """Finetunes the model on the provided data.

        Args:
            max_epochs (Optional[int], optional): The maximum number of epochs to train for. Defaults to None.

            min_epochs (Optional[int], optional): The minimum number of epochs to train for. Defaults to None.

            callbacks (Optional[List[pl.Callback]], optional): If provided, will be added to the callbacks for Trainer.
                Defaults to None.

            freeze_backbone (bool, optional): If True, will freeze the backbone by tirning off gradients.
                Defaults to False, which means the pretrained weights are also further tuned during fine-tuning.

        Returns:
            pl.Trainer: The trainer object

        """
        assert self._is_finetune_model, (
            "finetune() can only be called on a finetune model created using" " `TabularModel.create_finetune_model()`"
        )
        seed_everything(self.config.seed)
        if freeze_backbone:
            for param in self.model.backbone.parameters():
                param.requires_grad = False
        return self.train(
            self.model,
            self.datamodule,
            callbacks=callbacks,
            max_epochs=max_epochs,
            min_epochs=min_epochs,
        )

    def find_learning_rate(
        self,
        model: pl.LightningModule,
        datamodule: TabularDatamodule,
        min_lr: float = 1e-8,
        max_lr: float = 1,
        num_training: int = 100,
        mode: str = "exponential",
        early_stop_threshold: Optional[float] = 4.0,
        plot: bool = True,
        callbacks: Optional[List] = None,
    ) -> Tuple[float, DataFrame]:
        """Enables the user to do a range test of good initial learning rates, to reduce the amount of guesswork in
        picking a good starting learning rate.

        Args:
            model (pl.LightningModule): The PyTorch Lightning model to be trained.

            datamodule (TabularDatamodule): The datamodule

            min_lr (Optional[float], optional): minimum learning rate to investigate

            max_lr (Optional[float], optional): maximum learning rate to investigate

            num_training (Optional[int], optional): number of learning rates to test

            mode (Optional[str], optional): search strategy, either 'linear' or 'exponential'. If set to
                'linear' the learning rate will be searched by linearly increasing
                after each batch. If set to 'exponential', will increase learning
                rate exponentially.

            early_stop_threshold (Optional[float], optional): threshold for stopping the search. If the
                loss at any point is larger than early_stop_threshold*best_loss
                then the search is stopped. To disable, set to None.

            plot (bool, optional): If true, will plot using matplotlib

            callbacks (Optional[List], optional): If provided, will be added to the callbacks for Trainer.

        Returns:
            The suggested learning rate and the learning rate finder results

        """
        self._prepare_for_training(model, datamodule, callbacks, max_epochs=None, min_epochs=None)
        train_loader, _ = datamodule.train_dataloader(), datamodule.val_dataloader()
        lr_finder = Tuner(self.trainer).lr_find(
            model=self.model,
            train_dataloaders=train_loader,
            val_dataloaders=None,
            min_lr=min_lr,
            max_lr=max_lr,
            num_training=num_training,
            mode=mode,
            early_stop_threshold=early_stop_threshold,
        )
        if plot:
            fig = lr_finder.plot(suggest=True)
            fig.show()
        new_lr = lr_finder.suggestion()
        # cancelling the model and trainer that was loaded
        self.model = None
        self.trainer = None
        self.datamodule = None
        self.callbacks = None
        return new_lr, DataFrame(lr_finder.results)

    def evaluate(
        self,
        test: Optional[DataFrame] = None,
        test_loader: Optional[torch.utils.data.DataLoader] = None,
        ckpt_path: Optional[Union[str, Path]] = None,
        verbose: bool = True,
    ) -> Union[dict, list]:
        """Evaluates the dataframe using the loss and metrics already set in config.

        Args:
            test (Optional[DataFrame]): The dataframe to be evaluated. If not provided, will try to use the
                test provided during fit. If that was also not provided will return an empty dictionary

            test_loader (Optional[torch.utils.data.DataLoader], optional): The dataloader to be used for evaluation.
                If provided, will use the dataloader instead of the test dataframe or the test data provided during fit.
                Defaults to None.

            ckpt_path (Optional[Union[str, Path]], optional): The path to the checkpoint to be loaded. If not provided,
                will try to use the best checkpoint during training.

            verbose (bool, optional): If true, will print the results. Defaults to True.
        Returns:
            The final test result dictionary.

        """
        assert not (test_loader is None and test is None), (
            "Either `test_loader` or `test` should be provided."
            " If `test_loader` is not provided, `test` should be provided."
        )
        if test_loader is None:
            test_loader = self.datamodule.prepare_inference_dataloader(test)
        result = self.trainer.test(
            model=self.model,
            dataloaders=test_loader,
            ckpt_path=ckpt_path,
            verbose=verbose,
        )
        return result

    def _generate_predictions(
        self,
        model,
        inference_dataloader,
        quantiles,
        n_samples,
        ret_logits,
        progress_bar,
        is_probabilistic,
    ):
        point_predictions = []
        quantile_predictions = []
        logits_predictions = defaultdict(list)
        for batch in progress_bar(inference_dataloader):
            for k, v in batch.items():
                if isinstance(v, list) and (len(v) == 0):
                    continue  # Skipping empty list
                batch[k] = v.to(model.device)
            if is_probabilistic:
                samples, ret_value = model.sample(batch, n_samples, ret_model_output=True)
                y_hat = torch.mean(samples, dim=-1)
                quantile_preds = []
                for q in quantiles:
                    quantile_preds.append(torch.quantile(samples, q=q, dim=-1).unsqueeze(1))
            else:
                y_hat, ret_value = model.predict(batch, ret_model_output=True)
            if ret_logits:
                for k, v in ret_value.items():
                    logits_predictions[k].append(v.detach().cpu())
            point_predictions.append(y_hat.detach().cpu())
            if is_probabilistic:
                quantile_predictions.append(torch.cat(quantile_preds, dim=-1).detach().cpu())
        point_predictions = torch.cat(point_predictions, dim=0)
        if point_predictions.ndim == 1:
            point_predictions = point_predictions.unsqueeze(-1)
        if is_probabilistic:
            quantile_predictions = torch.cat(quantile_predictions, dim=0).unsqueeze(-1)
            if quantile_predictions.ndim == 2:
                quantile_predictions = quantile_predictions.unsqueeze(-1)
        return point_predictions, quantile_predictions, logits_predictions

    def _format_predicitons(
        self,
        test,
        point_predictions,
        quantile_predictions,
        logits_predictions,
        quantiles,
        ret_logits,
        include_input_features,
        is_probabilistic,
    ):
        pred_df = test.copy() if include_input_features else DataFrame(index=test.index)
        if self.config.task == "regression":
            point_predictions = point_predictions.numpy()
            # Probabilistic Models are only implemented for Regression
            if is_probabilistic:
                quantile_predictions = quantile_predictions.numpy()
            for i, target_col in enumerate(self.config.target):
                if self.datamodule.do_target_transform:
                    if self.config.target[i] in pred_df.columns:
                        pred_df[self.config.target[i]] = self.datamodule.target_transforms[i].inverse_transform(
                            pred_df[self.config.target[i]].values.reshape(-1, 1)
                        )
                    pred_df[f"{target_col}_prediction"] = self.datamodule.target_transforms[i].inverse_transform(
                        point_predictions[:, i].reshape(-1, 1)
                    )
                    if is_probabilistic:
                        for j, q in enumerate(quantiles):
                            col_ = f"{target_col}_q{int(q*100)}"
                            pred_df[col_] = self.datamodule.target_transforms[i].inverse_transform(
                                quantile_predictions[:, j, i].reshape(-1, 1)
                            )
                else:
                    pred_df[f"{target_col}_prediction"] = point_predictions[:, i]
                    if is_probabilistic:
                        for j, q in enumerate(quantiles):
                            pred_df[f"{target_col}_q{int(q*100)}"] = quantile_predictions[:, j, i].reshape(-1, 1)

        elif self.config.task == "classification":
            start_index = 0
            for i, target_col in enumerate(self.config.target):
                end_index = start_index + self.datamodule._inferred_config.output_cardinality[i]
                prob_prediction = nn.Softmax(dim=-1)(point_predictions[:, start_index:end_index]).numpy()
                start_index = end_index
                for j, class_ in enumerate(self.datamodule.label_encoder[i].classes_):
                    pred_df[f"{target_col}_{class_}_probability"] = prob_prediction[:, j]
                pred_df[f"{target_col}_prediction"] = self.datamodule.label_encoder[i].inverse_transform(
                    np.argmax(prob_prediction, axis=1)
                )
            warnings.warn(
                "Classification prediction column will be renamed to"
                " `{target_col}_prediction` in the next release to maintain"
                " consistency with regression.",
                DeprecationWarning,
            )
        if ret_logits:
            for k, v in logits_predictions.items():
                v = torch.cat(v, dim=0).numpy()
                if v.ndim == 1:
                    v = v.reshape(-1, 1)
                for i in range(v.shape[-1]):
                    if v.shape[-1] > 1:
                        pred_df[f"{k}_{i}"] = v[:, i]
                    else:
                        pred_df[f"{k}"] = v[:, i]
        return pred_df

    def _predict(
        self,
        test: DataFrame,
        quantiles: Optional[List] = [0.25, 0.5, 0.75],
        n_samples: Optional[int] = 100,
        ret_logits=False,
        include_input_features: bool = False,
        device: Optional[torch.device] = None,
        progress_bar: Optional[str] = None,
    ) -> DataFrame:
        """Uses the trained model to predict on new data and return as a dataframe.

        Args:
            test (DataFrame): The new dataframe with the features defined during training
            quantiles (Optional[List]): For probabilistic models like Mixture Density Networks, this specifies
                the different quantiles to be extracted apart from the `central_tendency` and added to the dataframe.
                For other models it is ignored. Defaults to [0.25, 0.5, 0.75]
            n_samples (Optional[int]): Number of samples to draw from the posterior to estimate the quantiles.
                Ignored for non-probabilistic models. Defaults to 100
            ret_logits (bool): Flag to return raw model outputs/logits except the backbone features along
                with the dataframe. Defaults to False
            include_input_features (bool): DEPRECATED: Flag to include the input features in the returned dataframe.
                Defaults to True
            progress_bar: choose progress bar for tracking the progress. "rich" or "tqdm" will set the respective
                progress bars. If None, no progress bar will be shown.

        Returns:
            DataFrame: Returns a dataframe with predictions and features (if `include_input_features=True`).
                If classification, it returns probabilities and final prediction

        """
        assert all(q <= 1 and q >= 0 for q in quantiles), "Quantiles should be a decimal between 0 and 1"
        model = self.model  # default
        if device is not None:
            if isinstance(device, str):
                device = torch.device(device)
            if self.model.device != device:
                model = self.model.to(device)
        model.eval()
        inference_dataloader = self.datamodule.prepare_inference_dataloader(test)
        is_probabilistic = hasattr(model.hparams, "_probabilistic") and model.hparams._probabilistic

        if progress_bar == "rich":
            from rich.progress import track

            progress_bar = partial(track, description="Generating Predictions...")
        elif progress_bar == "tqdm":
            from tqdm.auto import tqdm

            progress_bar = partial(tqdm, description="Generating Predictions...")
        else:
            progress_bar = lambda it: it  # E731
        point_predictions, quantile_predictions, logits_predictions = self._generate_predictions(
            model,
            inference_dataloader,
            quantiles,
            n_samples,
            ret_logits,
            progress_bar,
            is_probabilistic,
        )
        pred_df = self._format_predicitons(
            test,
            point_predictions,
            quantile_predictions,
            logits_predictions,
            quantiles,
            ret_logits,
            include_input_features,
            is_probabilistic,
        )
        return pred_df

    def predict(
        self,
        test: DataFrame,
        quantiles: Optional[List] = [0.25, 0.5, 0.75],
        n_samples: Optional[int] = 100,
        ret_logits=False,
        include_input_features: bool = False,
        device: Optional[torch.device] = None,
        progress_bar: Optional[str] = None,
        test_time_augmentation: Optional[bool] = False,
        num_tta: Optional[float] = 5,
        alpha_tta: Optional[float] = 0.1,
        aggregate_tta: Optional[str] = "mean",
        tta_seed: Optional[int] = 42,
    ) -> DataFrame:
        """Uses the trained model to predict on new data and return as a dataframe.

        Args:
            test (DataFrame): The new dataframe with the features defined during training

            quantiles (Optional[List]): For probabilistic models like Mixture Density Networks, this specifies
                the different quantiles to be extracted apart from the `central_tendency` and added to the dataframe.
                For other models it is ignored. Defaults to [0.25, 0.5, 0.75]

            n_samples (Optional[int]): Number of samples to draw from the posterior to estimate the quantiles.
                Ignored for non-probabilistic models. Defaults to 100

            ret_logits (bool): Flag to return raw model outputs/logits except the backbone features along
                with the dataframe. Defaults to False

            include_input_features (bool): DEPRECATED: Flag to include the input features in the returned dataframe.
                Defaults to True

            progress_bar: choose progress bar for tracking the progress. "rich" or "tqdm" will set the respective
                progress bars. If None, no progress bar will be shown.

            test_time_augmentation (bool): If True, will use test time augmentation to generate predictions.
                The approach is very similar to what is described [here](https://kozodoi.me/blog/20210908/tta-tabular)
                But, we add noise to the embedded inputs to handle categorical features as well.\
                \\(x_{aug} = x_{orig} + \alpha * \\epsilon\\) where \\(\\epsilon \\sim \\mathcal{N}(0, 1)\\)
                Defaults to False
            num_tta (float): The number of augumentations to run TTA for. Defaults to 0.0

            alpha_tta (float): The standard deviation of the gaussian noise to be added to the input features

            aggregate_tta (Union[str, Callable], optional): The function to be used to aggregate the
                predictions from each augumentation. If str, should be one of "mean", "median", "min", or "max"
                for regression. For classification, the previous options are applied to the confidence
                scores (soft voting) and then converted to final prediction. An additional option
                "hard_voting" is available for classification.
                If callable, should be a function that takes in a list of 3D arrays (num_samples, num_cv, num_targets)
                and returns a 2D array of final probabilities (num_samples, num_targets). Defaults to "mean".'

            tta_seed (int): The random seed to be used for the noise added in TTA. Defaults to 42.

        Returns:
            DataFrame: Returns a dataframe with predictions and features (if `include_input_features=True`).
                If classification, it returns probabilities and final prediction

        """
        warnings.warn(
            "`include_input_features` will be deprecated in the next release."
            " Please add index columns to the test dataframe if you want to"
            " retain some features like the key or id",
            DeprecationWarning,
        )
        if test_time_augmentation:
            assert num_tta > 0, "num_tta should be greater than 0"
            assert alpha_tta > 0, "alpha_tta should be greater than 0"
            assert include_input_features is False, "include_input_features cannot be True for TTA."
            if not callable(aggregate_tta):
                assert aggregate_tta in [
                    "mean",
                    "median",
                    "min",
                    "max",
                    "hard_voting",
                ], "aggregate should be one of 'mean', 'median', 'min', 'max', or" " 'hard_voting'"
            if self.config.task == "regression":
                assert aggregate_tta != "hard_voting", "hard_voting is only available for classification"

            torch.manual_seed(tta_seed)

            def add_noise(module, input, output):
                return output + alpha_tta * torch.randn_like(output, memory_format=torch.contiguous_format)

            # Register the hook to the embedding_layer
            handle = self.model.embedding_layer.register_forward_hook(add_noise)
            pred_prob_l = []
            for _ in range(num_tta):
                pred_df = self._predict(
                    test,
                    quantiles,
                    n_samples,
                    ret_logits,
                    include_input_features=False,
                    device=device,
                    progress_bar=progress_bar or "None",
                )
                pred_idx = pred_df.index
                if self.config.task == "classification":
                    pred_prob_l.append(pred_df.values[:, : -len(self.config.target)])
                elif self.config.task == "regression":
                    pred_prob_l.append(pred_df.values)
            pred_df = self._combine_predictions(pred_prob_l, pred_idx, aggregate_tta, None)
            # Remove the hook
            handle.remove()
        else:
            pred_df = self._predict(
                test,
                quantiles,
                n_samples,
                ret_logits,
                include_input_features,
                device,
                progress_bar,
            )
        return pred_df

    @rank_zero_only
    def load_best_model(self) -> None:
        """Loads the best model after training is done."""
        if self.trainer.checkpoint_callback is not None:
            if self.verbose:
                logger.info("Loading the best model")
            ckpt_path = self.trainer.checkpoint_callback.best_model_path
            if ckpt_path != "":
                if self.verbose:
                    logger.debug(f"Model Checkpoint: {ckpt_path}")
                ckpt = pl_load(ckpt_path, map_location=lambda storage, loc: storage)
                self.model.load_state_dict(ckpt["state_dict"])
            else:
                logger.warning("No best model available to load. Did you run it more than 1" " epoch?...")
        else:
            logger.warning(
                "No best model available to load. Checkpoint Callback needs to be" " enabled for this to work"
            )

    def save_datamodule(self, dir: str, inference_only: bool = False) -> None:
        """Saves the datamodule in the specified directory.

        Args:
            dir (str): The path to the directory to save the datamodule
            inference_only (bool): If True, will only save the inference datamodule
                without data. This cannot be used for further training, but can be
                used for inference. Defaults to False.

        """
        if inference_only:
            dm = self.datamodule.inference_only_copy()
        else:
            dm = self.datamodule

        joblib.dump(dm, os.path.join(dir, "datamodule.sav"))

    def save_config(self, dir: str) -> None:
        """Saves the config in the specified directory."""
        with open(os.path.join(dir, "config.yml"), "w") as fp:
            OmegaConf.save(self.config, fp, resolve=True)

    def save_model(self, dir: str, inference_only: bool = False) -> None:
        """Saves the model and checkpoints in the specified directory.

        Args:
            dir (str): The path to the directory to save the model
            inference_only (bool): If True, will only save the inference
                only version of the datamodule

        """
        if os.path.exists(dir) and (os.listdir(dir)):
            logger.warning("Directory is not empty. Overwriting the contents.")
            for f in os.listdir(dir):
                os.remove(os.path.join(dir, f))
        os.makedirs(dir, exist_ok=True)
        self.save_config(dir)
        self.save_datamodule(dir, inference_only=inference_only)
        if hasattr(self.config, "log_target") and self.config.log_target is not None:
            joblib.dump(self.logger, os.path.join(dir, "exp_logger.sav"))
        if hasattr(self, "callbacks"):
            joblib.dump(self.callbacks, os.path.join(dir, "callbacks.sav"))
        self.trainer.save_checkpoint(os.path.join(dir, "model.ckpt"))
        custom_params = {}
        custom_params["custom_loss"] = getattr(self.model, "custom_loss", None)
        custom_params["custom_metrics"] = getattr(self.model, "custom_metrics", None)
        custom_params["custom_metrics_prob_inputs"] = getattr(self.model, "custom_metrics_prob_inputs", None)
        custom_params["custom_optimizer"] = getattr(self.model, "custom_optimizer", None)
        custom_params["custom_optimizer_params"] = getattr(self.model, "custom_optimizer_params", None)
        joblib.dump(custom_params, os.path.join(dir, "custom_params.sav"))
        if self.custom_model:
            joblib.dump(self.model_callable, os.path.join(dir, "custom_model_callable.sav"))

    def save_weights(self, path: Union[str, Path]) -> None:
        """Saves the model weights in the specified directory.

        Args:
            path (str): The path to the file to save the model

        """
        torch.save(self.model.state_dict(), path)

    def load_weights(self, path: Union[str, Path]) -> None:
        """Loads the model weights in the specified directory.

        Args:
            path (str): The path to the file to load the model from

        """
        self._load_weights(self.model, path)

    # TODO Need to test ONNX export
    def save_model_for_inference(
        self,
        path: Union[str, Path],
        kind: str = "pytorch",
        onnx_export_params: Dict = {"opset_version": 12},
    ) -> bool:
        """Saves the model for inference.

        Args:
            path (Union[str, Path]): path to save the model
            kind (str): "pytorch" or "onnx" (Experimental)
            onnx_export_params (Dict): parameters for onnx export to be
                passed to torch.onnx.export

        Returns:
            bool: True if the model was saved successfully

        """
        if kind == "pytorch":
            torch.save(self.model, str(path))
            return True
        elif kind == "onnx":
            # Export the model
            onnx_export_params["input_names"] = ["categorical", "continuous"]
            onnx_export_params["output_names"] = onnx_export_params.get("output_names", ["output"])
            onnx_export_params["dynamic_axes"] = {
                onnx_export_params["input_names"][0]: {0: "batch_size"},
                onnx_export_params["output_names"][0]: {0: "batch_size"},
            }
            cat = torch.zeros(
                self.config.batch_size,
                len(self.config.categorical_cols),
                dtype=torch.int,
            )
            cont = torch.randn(
                self.config.batch_size,
                len(self.config.continuous_cols),
                requires_grad=True,
            )
            x = {"continuous": cont, "categorical": cat}
            torch.onnx.export(self.model, x, str(path), **onnx_export_params)
            return True
        else:
            raise ValueError("`kind` must be either pytorch or onnx")

    def summary(self, model=None, max_depth: int = -1) -> None:
        """Prints a summary of the model.

        Args:
            max_depth (int): The maximum depth to traverse the modules and
                displayed in the summary. Defaults to -1, which means will
                display all the modules.

        """
        if model is not None:
            print(summarize(model, max_depth=max_depth))
        elif self.has_model:
            print(summarize(self.model, max_depth=max_depth))
        else:
            rich_print(f"[bold green]{self.__class__.__name__}[/bold green]")
            rich_print("-" * 100)
            rich_print("[bold yellow]Config[/bold yellow]")
            rich_print("-" * 100)
            pprint(self.config.__dict__["_content"])
            rich_print(
                ":triangular_flag:[bold red]Full Model Summary once model has "
                "been initialized or passed in as an argument[/bold red]"
            )

    def ret_summary(self, model=None, max_depth: int = -1) -> str:
        """Returns a summary of the model as a string.

        Args:
            max_depth (int): The maximum depth to traverse the modules and
                displayed in the summary. Defaults to -1, which means will
                display all the modules.

        Returns:
            str: The summary of the model.

        """
        if model is not None:
            return str(summarize(model, max_depth=max_depth))
        elif self.has_model:
            return str(summarize(self.model, max_depth=max_depth))
        else:
            summary_str = f"{self.__class__.__name__}\n"
            summary_str += "-" * 100 + "\n"
            summary_str += "Config\n"
            summary_str += "-" * 100 + "\n"
            summary_str += pformat(self.config.__dict__["_content"], indent=4, width=80, compact=True)
            summary_str += "\nFull Model Summary once model has been " "initialized or passed in as an argument"
            return summary_str

    def __str__(self) -> str:
        """Returns a readable summary of the TabularModel object."""
        model_name = self.model.__class__.__name__ if self.has_model else self.config._model_name + "(Not Initialized)"
        return f"{self.__class__.__name__}(model={model_name})"

    def __repr__(self) -> str:
        """Returns an unambiguous representation of the TabularModel object."""
        config_str = json.dumps(OmegaConf.to_container(self.config, resolve=True), indent=4)
        ret_str = f"{self.__class__.__name__}(\n"
        if self.has_model:
            ret_str += f"  model={self.model.__class__.__name__},\n"
        else:
            ret_str += f"  model={self.config._model_name} (Not Initialized),\n"
        ret_str += f"  config={config_str},\n"
        return ret_str

    def _repr_html_(self):
        """Generate an HTML representation for Jupyter Notebook."""
        css = """
        <style>
        .main-container {
            font-family: Arial, sans-serif;
            font-size: 14px;
            border: 1px dashed #ccc;
            padding: 10px;
            margin: 10px;
            background-color: #f9f9f9;
        }
        .header {
            background-color: #e8f4fc;
            padding: 5px;
            font-weight: bold;
            text-align: center;
            border-bottom: 1px solid #ccc;
        }
        .section {
            margin: 10px 0;
            padding: 10px;
            border: 1px solid #ccc;
            background-color: #ffffff;
        }
        .step {
            border: 1px solid #ccc;
            background-color: #f0f8ff;
            margin: 5px 0;
            padding: 5px;
        }
        .sub-step {
            margin-left: 20px;
            border: 1px solid #ddd;
            background-color: #f9f9f9;
            padding: 5px;
        }
        .toggle-button {
            cursor: pointer;
            font-size: 12px;
            margin-right: 5px;
        }
        .toggle-button:hover {
            color: #0056b3;
        }
        .hidden {
            display: none;
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        table, th, td {
            border: 1px solid black;
        }
        th, td {
            padding: 5px;
            text-align: left;
        }
        </style>
        <script>
        function toggleVisibility(id) {
            var element = document.getElementById(id);
            if (element.classList.contains('hidden')) {
                element.classList.remove('hidden');
            } else {
                element.classList.add('hidden');
            }
        }
        </script>
        """

        # Header (Main model name)
        uid = str(uuid.uuid4())
        model_status = "" if self.has_model else "(Not Initialized)"
        model_name = self.model.__class__.__name__ if self.has_model else self.config._model_name
        header_html = f"<div class='header'>{html.escape(model_name)}{model_status}</div>"

        # Config Section
        config_html = self._generate_collapsible_section("Model Config", self.config, uid=uid, is_dict=True)

        # Summary Section
        summary_html = (
            ""
            if not self.has_model
            else self._generate_collapsible_section("Model Summary", self._generate_model_summary_table(), uid=uid)
        )

        # Combine sections
        return f"""
        {css}
        <div class='main-container'>
            {header_html}
            {config_html}
            {summary_html}
        </div>
        """

    def _generate_collapsible_section(self, title, content, uid, is_dict=False):
        container_id = title.lower().replace(" ", "_") + uid
        if is_dict:
            content = self._generate_nested_collapsible_sections(
                OmegaConf.to_container(content, resolve=True), container_id
            )
        return f"""
        <div>
            <span
            class="toggle-button"
            onclick="toggleVisibility('{container_id}')"
            >
            &#9654;
            </span>
            <strong>{html.escape(title)}</strong>
            <div id="{container_id}" class="hidden section">
                {content}
            </div>
        </div>
        """

    def _generate_nested_collapsible_sections(self, content, parent_id):
        html_content = ""
        for key, value in content.items():
            if isinstance(value, dict):
                nested_id = f"{parent_id}_{key}".replace(" ", "_")
                nested_id = nested_id + str(uuid.uuid4())
                nested_content = self._generate_nested_collapsible_sections(value, nested_id)
                html_content += f"""
                <div>
                    <span
                    class="toggle-button"
                    onclick="toggleVisibility('{nested_id}')"
                    >
                    &#9654;
                    </span>
                    <strong>{html.escape(key)}</strong>
                    <div id="{nested_id}" class="hidden section">
                        {nested_content}
                    </div>
                </div>
                """
            else:
                html_content += f"<div><strong>{html.escape(key)}:</strong> {html.escape(str(value))}</div>"
        return html_content

    def _generate_model_summary_table(self):
        model_summary = summarize(self.model, max_depth=1)
        table_html = """
        <table>
            <tr>
                <th><b>Layer</b></th>
                <th><b>Type</b></th>
                <th><b>Params</b></th>
                <th><b>In sizes</b></th>
                <th><b>Out sizes</b></th>
            </tr>
        """
        for name, layer in model_summary._layer_summary.items():
            table_html += f"""
                <tr>
                    <td>{html.escape(name)}</td>
                    <td>{html.escape(layer.layer_type)}</td>
                    <td>{html.escape(str(layer.num_parameters))}</td>
                    <td>{html.escape(str(layer.in_size))}</td>
                    <td>{html.escape(str(layer.out_size))}</td>
                </tr>
            """
        table_html += "</table>"
        return table_html

    def feature_importance(self) -> DataFrame:
        """Returns the feature importance of the model as a pandas DataFrame."""
        return self.model.feature_importance()

    def _prepare_input_for_captum(self, test_dl: torch.utils.data.DataLoader) -> Dict:
        tensor_inp = []
        tensor_tgt = []
        for x in test_dl:
            tensor_inp.append(self.model.embed_input(x))
            tensor_tgt.append(x["target"].squeeze(1))
        tensor_inp = torch.cat(tensor_inp, dim=0)
        tensor_tgt = torch.cat(tensor_tgt, dim=0)
        return tensor_inp, tensor_tgt

    def _prepare_baselines_captum(
        self,
        baselines: Union[float, torch.tensor, str],
        test_dl: torch.utils.data.DataLoader,
        do_baselines: bool,
        is_full_baselines: bool,
    ):
        if do_baselines and baselines is not None and isinstance(baselines, str):
            if baselines.startswith("b|"):
                num_samples = int(baselines.split("|")[1])
                tensor_inp_tr = []
                # tensor_tgt_tr = []
                count = 0
                for x in self.datamodule.train_dataloader():
                    tensor_inp_tr.append(self.model.embed_input(x))
                    # tensor_tgt_tr.append(x["target"])
                    count += x["target"].shape[0]
                    if count >= num_samples:
                        break
                tensor_inp_tr = torch.cat(tensor_inp_tr, dim=0)
                # tensor_tgt_tr = torch.cat(tensor_tgt_tr, dim=0)
                baselines = tensor_inp_tr[:num_samples]
                if is_full_baselines:
                    pass
                else:
                    baselines = baselines.mean(dim=0, keepdim=True)
            else:
                raise ValueError(
                    "Invalid value for `baselines`. Please refer to the documentation" " for more details."
                )
        return baselines

    def _handle_categorical_embeddings_attributions(
        self,
        attributions: torch.tensor,
        is_embedding1d: bool,
        is_embedding2d: bool,
        is_embbeding_dims: bool,
    ):
        # post processing to get attributions for categorical features
        if is_embedding1d and is_embbeding_dims:
            if self.model.hparams.categorical_dim > 0:
                cat_attributions = []
                index_counter = self.model.hparams.continuous_dim
                for _, embed_dim in self.model.hparams.embedding_dims:
                    cat_attributions.append(attributions[:, index_counter : index_counter + embed_dim].sum(dim=1))
                    index_counter += embed_dim
                cat_attributions = torch.stack(cat_attributions, dim=1)
                attributions = torch.cat(
                    [
                        attributions[:, : self.model.hparams.continuous_dim],
                        cat_attributions,
                    ],
                    dim=1,
                )
        elif is_embedding2d:
            attributions = attributions.mean(dim=-1)
        return attributions

    def explain(
        self,
        data: DataFrame,
        method: str = "GradientShap",
        method_args: Optional[Dict] = {},
        baselines: Union[float, torch.tensor, str] = None,
        **kwargs,
    ) -> DataFrame:
        """Returns the feature attributions/explanations of the model as a pandas DataFrame. The shape of the returned
        dataframe is (num_samples, num_features)

        Args:
            data (DataFrame): The dataframe to be explained
            method (str): The method to be used for explaining the model.
                It should be one of the Defaults to "GradientShap".
                For more details, refer to https://captum.ai/api/attribution.html
            method_args (Optional[Dict], optional): The arguments to be passed to the initialization
                of the Captum method.
            baselines (Union[float, torch.tensor, str]): The baselines to be used for the explanation.
                If a scalar is provided, will use that value as the baseline for all the features.
                If a tensor is provided, will use that tensor as the baseline for all the features.
                If a string like `b|<num_samples>` is provided, will use that many samples from the train
                Using the whole train data as the baseline is not recommended as it can be
                computationally expensive. By default, PyTorch Tabular uses 10000 samples from the
                train data as the baseline. You can configure this by passing a special string
                "b|<num_samples>" where <num_samples> is the number of samples to be used as the
                baseline. For eg. "b|1000" will use 1000 samples from the train.
                If None, will use default settings like zero in captum(which is method dependent).
                For `GradientShap`, it is the train data.
                Defaults to None.

            **kwargs: Additional keyword arguments to be passed to the Captum method `attribute` function.

        Returns:
            DataFrame: The dataframe with the feature importance

        """
        assert CAPTUM_INSTALLED, "Captum not installed. Please install using `pip install captum` or "
        "install PyTorch Tabular using `pip install pytorch-tabular[extra]`"
        ALLOWED_METHODS = [
            "GradientShap",
            "IntegratedGradients",
            "DeepLift",
            "DeepLiftShap",
            "InputXGradient",
            "FeaturePermutation",
            "FeatureAblation",
            "KernelShap",
        ]
        assert method in ALLOWED_METHODS, f"method should be one of {ALLOWED_METHODS}"
        if isinstance(data, pd.Series):
            data = data.to_frame().T
        if method in ["DeepLiftShap", "KernelShap"]:
            warnings.warn(
                f"{method} is computationally expensive and will take some time. For"
                " faster results, try usingsome other methods like GradientShap,"
                " IntegratedGradients etc."
            )
        if method in ["FeaturePermutation", "FeatureAblation"]:
            assert data.shape[0] > 1, f"{method} only works when the number of samples is greater than 1"
            if len(data) <= 100:
                warnings.warn(
                    f"{method} gives better results when the number of samples is"
                    " large. For better results, try using more samples or some other"
                    " methods like GradientShap which works well on single examples."
                )
        is_full_baselines = method in ["GradientShap", "DeepLiftShap"]
        is_not_supported = self.model._get_name() in [
            "TabNetModel",
            "MDNModel",
            "TabTransformerModel",
        ]
        do_baselines = method not in [
            "Saliency",
            "InputXGradient",
            "FeaturePermutation",
            "LRP",
        ]
        if is_full_baselines and (baselines is None or isinstance(baselines, (float, int))):
            raise ValueError(
                f"baselines cannot be a scalar or None for {method}. Please "
                "provide a tensor or a string like `b|<num_samples>`"
            )
        if is_not_supported:
            raise NotImplementedError(f"Attributions are not implemented for {self.model._get_name()}")

        is_embedding1d = isinstance(self.model.embedding_layer, (Embedding1dLayer, PreEncoded1dLayer))
        is_embedding2d = isinstance(self.model.embedding_layer, Embedding2dLayer)
        # Models like NODE may have no embedding dims (doing leaveOneOut encoding) even if categorical_dim > 0
        is_embbeding_dims = (
            hasattr(self.model.hparams, "embedding_dims") and self.model.hparams.embedding_dims is not None
        )
        if (not is_embedding1d) and (not is_embedding2d):
            raise NotImplementedError(
                "Attributions are not implemented for models with this type of" " embedding layer"
            )
        test_dl = self.datamodule.prepare_inference_dataloader(data)
        self.model.eval()
        # prepare import for Captum
        tensor_inp, tensor_tgt = self._prepare_input_for_captum(test_dl)
        baselines = self._prepare_baselines_captum(baselines, test_dl, do_baselines, is_full_baselines)
        # prepare model for Captum
        try:
            interp_model = _CaptumModel(self.model)
            captum_interp_cls = getattr(captum.attr, method)(interp_model, **method_args)
            if do_baselines:
                attributions = captum_interp_cls.attribute(
                    tensor_inp,
                    baselines=baselines,
                    target=(tensor_tgt if self.config.task == "classification" else None),
                    **kwargs,
                )
            else:
                attributions = captum_interp_cls.attribute(
                    tensor_inp,
                    target=(tensor_tgt if self.config.task == "classification" else None),
                    **kwargs,
                )
            attributions = self._handle_categorical_embeddings_attributions(
                attributions, is_embedding1d, is_embedding2d, is_embbeding_dims
            )
        finally:
            self.model.train()
        assert attributions.shape[1] == self.model.hparams.continuous_dim + self.model.hparams.categorical_dim, (
            "Something went wrong. The number of features in the attributions"
            f" ({attributions.shape[1]}) does not match the number of features in"
            " the model"
            f" ({self.model.hparams.continuous_dim+self.model.hparams.categorical_dim})"
        )
        return pd.DataFrame(
            attributions.detach().cpu().numpy(),
            columns=self.config.continuous_cols + self.config.categorical_cols,
        )

    def _check_cv(self, cv):
        cv = 5 if cv is None else cv
        if isinstance(cv, int):
            if self.config.task == "classification":
                return StratifiedKFold(cv)
            else:
                return KFold(cv)
        elif isinstance(cv, Iterable) and not isinstance(cv, str):
            # An iterable yielding (train, test) splits as arrays of indices.
            return cv
        elif isinstance(cv, BaseCrossValidator):
            return cv
        else:
            raise ValueError("cv must be int, iterable or scikit-learn splitter")

    def _split_kwargs(self, kwargs):
        prep_dl_kwargs = {}
        prep_model_kwargs = {}
        train_kwargs = {}
        # using the defined args in self.prepare_dataloder, self.prepare_model, and self.train
        # to split the kwargs
        for k, v in kwargs.items():
            if k in self.prepare_dataloader.__code__.co_varnames:
                prep_dl_kwargs[k] = v
            elif k in self.prepare_model.__code__.co_varnames:
                prep_model_kwargs[k] = v
            elif k in self.train.__code__.co_varnames:
                train_kwargs[k] = v
            else:
                raise ValueError(f"Invalid keyword argument: {k}")
        return prep_dl_kwargs, prep_model_kwargs, train_kwargs

    def cross_validate(
        self,
        cv: Optional[Union[int, Iterable, BaseCrossValidator]],
        train: DataFrame,
        metric: Optional[Union[str, Callable]] = None,
        return_oof: bool = False,
        groups: Optional[Union[str, np.ndarray]] = None,
        verbose: bool = True,
        reset_datamodule: bool = True,
        handle_oom: bool = True,
        **kwargs,
    ):
        """Cross validate the model.

        Args:
            cv (Optional[Union[int, Iterable, BaseCrossValidator]]): Determines the cross-validation splitting strategy.
                Possible inputs for cv are:

                - None, to use the default 5-fold cross validation (KFold for
                Regression and StratifiedKFold for Classification),
                - integer, to specify the number of folds in a (Stratified)KFold,
                - An iterable yielding (train, test) splits as arrays of indices.
                - A scikit-learn CV splitter.

            train (DataFrame): The training data with labels

            metric (Optional[Union[str, Callable]], optional): The metrics to be used for evaluation.
                If None, will use the first metric in the config. If str is provided, will use that
                metric from the defined ones. If callable is provided, will use that function as the
                metric. We expect callable to be of the form `metric(y_true, y_pred)`. For classification
                problems, The `y_pred` is a dataframe with the probabilities for each class
                (<class>_probability) and a final prediction(prediction). And for Regression, it is a
                dataframe with a final prediction (<target>_prediction).
                Defaults to None.

            return_oof (bool, optional): If True, will return the out-of-fold predictions
                along with the cross validation results. Defaults to False.

            groups (Optional[Union[str, np.ndarray]], optional): Group labels for
                the samples used while splitting. If provided, will be used as the
                `groups` argument for the `split` method of the cross validator.
                If input is str, will use the column in the input dataframe with that
                name as the group labels. If input is array-like, will use that as the
                group. The only constraint is that the group labels should have the
                same size as the number of rows in the input dataframe. Defaults to None.

            verbose (bool, optional): If True, will log the results. Defaults to True.

            reset_datamodule (bool, optional): If True, will reset the datamodule for each iteration.
                It will be slower because we will be fitting the transformations for each fold.
                If False, we take an approximation that once the transformations are fit on the first
                fold, they will be valid for all the other folds. Defaults to True.

            handle_oom (bool, optional): If True, will handle out of memory errors elegantly
            **kwargs: Additional keyword arguments to be passed to the `fit` method of the model.

        Returns:
            DataFrame: The dataframe with the cross validation results

        """
        cv = self._check_cv(cv)
        prep_dl_kwargs, prep_model_kwargs, train_kwargs = self._split_kwargs(kwargs)
        is_callable_metric = False
        if metric is None:
            metric = "test_" + self.config.metrics[0]
        elif isinstance(metric, str):
            metric = metric if metric.startswith("test_") else "test_" + metric
        elif callable(metric):
            is_callable_metric = True

        if isinstance(cv, BaseCrossValidator):
            it = enumerate(cv.split(train, y=train[self.config.target], groups=groups))
        else:
            # when iterable is directly passed
            it = enumerate(cv)
        cv_metrics = []
        datamodule = None
        model = None
        oof_preds = []
        for fold, (train_idx, val_idx) in it:
            if verbose:
                logger.info(f"Running Fold {fold+1}/{cv.get_n_splits()}")
            # train_fold = train.iloc[train_idx]
            # val_fold = train.iloc[val_idx]
            if reset_datamodule:
                datamodule = None
            if datamodule is None:
                # Initialize datamodule and model in the first fold
                # uses train data from this fold to fit all transformers
                datamodule = self.prepare_dataloader(
                    train=train.iloc[train_idx],
                    validation=train.iloc[val_idx],
                    seed=42,
                    **prep_dl_kwargs,
                )
                model = self.prepare_model(datamodule, **prep_model_kwargs)
            else:
                # Preprocess the current fold data using the fitted transformers and save in datamodule
                datamodule.train, _ = datamodule.preprocess_data(train.iloc[train_idx], stage="inference")
                datamodule.validation, _ = datamodule.preprocess_data(train.iloc[val_idx], stage="inference")

            # Train the model
            handle_oom = train_kwargs.pop("handle_oom", handle_oom)
            self.train(model, datamodule, handle_oom=handle_oom, **train_kwargs)
            if return_oof or is_callable_metric:
                preds = self.predict(train.iloc[val_idx], include_input_features=False)
                oof_preds.append(preds)
            if is_callable_metric:
                cv_metrics.append(metric(train.iloc[val_idx][self.config.target], preds))
            else:
                result = self.evaluate(train.iloc[val_idx], verbose=False)
                cv_metrics.append(result[0][metric])
            if verbose:
                logger.info(f"Fold {fold+1}/{cv.get_n_splits()} score: {cv_metrics[-1]}")
            self.model.reset_weights()
        return cv_metrics, oof_preds

    def _combine_predictions(
        self,
        pred_prob_l: List[DataFrame],
        pred_idx: Union[pd.Index, List],
        aggregate: Union[str, Callable],
        weights: Optional[List[float]] = None,
    ):
        if aggregate == "mean":
            bagged_pred = np.average(pred_prob_l, axis=0, weights=weights)
        elif aggregate == "median":
            bagged_pred = np.median(pred_prob_l, axis=0)
        elif aggregate == "min":
            bagged_pred = np.min(pred_prob_l, axis=0)
        elif aggregate == "max":
            bagged_pred = np.max(pred_prob_l, axis=0)
        elif aggregate == "hard_voting" and self.config.task == "classification":
            pred_l = [np.argmax(p, axis=1) for p in pred_prob_l]
            final_pred = np.apply_along_axis(
                lambda x: np.argmax(np.bincount(x)),
                axis=0,
                arr=pred_l,
            )
        elif callable(aggregate):
            bagged_pred = aggregate(pred_prob_l)
        if self.config.task == "classification":
            # FIXME need to iterate .label_encoder[x]
            classes = self.datamodule.label_encoder[0].classes_
            if aggregate == "hard_voting":
                pred_df = pd.DataFrame(
                    np.concatenate(pred_prob_l, axis=1),
                    columns=[f"{c}_probability_fold_{i}" for i in range(len(pred_prob_l)) for c in classes],
                    index=pred_idx,
                )
                pred_df["prediction"] = classes[final_pred]
            else:
                final_pred = classes[np.argmax(bagged_pred, axis=1)]
                pred_df = pd.DataFrame(
                    bagged_pred,
                    # FIXME
                    columns=[f"{c}_probability" for c in self.datamodule.label_encoder[0].classes_],
                    index=pred_idx,
                )
                pred_df["prediction"] = final_pred
        elif self.config.task == "regression":
            pred_df = pd.DataFrame(bagged_pred, columns=self.config.target, index=pred_idx)
        else:
            raise NotImplementedError(f"Task {self.config.task} not supported for bagging")
        return pred_df

    def bagging_predict(
        self,
        cv: Optional[Union[int, Iterable, BaseCrossValidator]],
        train: DataFrame,
        test: DataFrame,
        groups: Optional[Union[str, np.ndarray]] = None,
        verbose: bool = True,
        reset_datamodule: bool = True,
        return_raw_predictions: bool = False,
        aggregate: Union[str, Callable] = "mean",
        weights: Optional[List[float]] = None,
        handle_oom: bool = True,
        **kwargs,
    ):
        """Bagging predict on the test data.

        Args:
            cv (Optional[Union[int, Iterable, BaseCrossValidator]]): Determines the cross-validation splitting strategy.
                Possible inputs for cv are:

                - None, to use the default 5-fold cross validation (KFold for
                Regression and StratifiedKFold for Classification),
                - integer, to specify the number of folds in a (Stratified)KFold,
                - An iterable yielding (train, test) splits as arrays of indices.
                - A scikit-learn CV splitter.

            train (DataFrame): The training data with labels

            test (DataFrame): The test data to be predicted

            groups (Optional[Union[str, np.ndarray]], optional): Group labels for
                the samples used while splitting. If provided, will be used as the
                `groups` argument for the `split` method of the cross validator.
                If input is str, will use the column in the input dataframe with that
                name as the group labels. If input is array-like, will use that as the
                group. The only constraint is that the group labels should have the
                same size as the number of rows in the input dataframe. Defaults to None.

            verbose (bool, optional): If True, will log the results. Defaults to True.

            reset_datamodule (bool, optional): If True, will reset the datamodule for each iteration.
                It will be slower because we will be fitting the transformations for each fold.
                If False, we take an approximation that once the transformations are fit on the first
                fold, they will be valid for all the other folds. Defaults to True.

            return_raw_predictions (bool, optional): If True, will return the raw predictions
                from each fold. Defaults to False.

            aggregate (Union[str, Callable], optional): The function to be used to aggregate the
                predictions from each fold. If str, should be one of "mean", "median", "min", or "max"
                for regression. For classification, the previous options are applied to the confidence
                scores (soft voting) and then converted to final prediction. An additional option
                "hard_voting" is available for classification.
                If callable, should be a function that takes in a list of 3D arrays (num_samples, num_cv, num_targets)
                and returns a 2D array of final probabilities (num_samples, num_targets). Defaults to "mean".

            weights (Optional[List[float]], optional): The weights to be used for aggregating the predictions
                from each fold. If None, will use equal weights. This is only used when `aggregate` is "mean".
                Defaults to None.

            handle_oom (bool, optional): If True, will handle out of memory errors elegantly

            **kwargs: Additional keyword arguments to be passed to the `fit` method of the model.

        Returns:
            DataFrame: The dataframe with the bagged predictions.

        """
        if weights is not None:
            assert len(weights) == cv.n_splits, "Number of weights should be equal to the number of folds"
        assert self.config.task in [
            "classification",
            "regression",
        ], "Bagging is only available for classification and regression"
        if not callable(aggregate):
            assert aggregate in ["mean", "median", "min", "max", "hard_voting"], (
                "aggregate should be one of 'mean', 'median', 'min', 'max', or" " 'hard_voting'"
            )
        if self.config.task == "regression":
            assert aggregate != "hard_voting", "hard_voting is only available for classification"
        cv = self._check_cv(cv)
        prep_dl_kwargs, prep_model_kwargs, train_kwargs = self._split_kwargs(kwargs)
        pred_prob_l = []
        datamodule = None
        model = None
        for fold, (train_idx, val_idx) in enumerate(cv.split(train, y=train[self.config.target], groups=groups)):
            if verbose:
                logger.info(f"Running Fold {fold+1}/{cv.get_n_splits()}")
            train_fold = train.iloc[train_idx]
            val_fold = train.iloc[val_idx]
            if reset_datamodule:
                datamodule = None
            if datamodule is None:
                # Initialize datamodule and model in the first fold
                # uses train data from this fold to fit all transformers
                datamodule = self.prepare_dataloader(train=train_fold, validation=val_fold, seed=42, **prep_dl_kwargs)
                model = self.prepare_model(datamodule, **prep_model_kwargs)
            else:
                # Preprocess the current fold data using the fitted transformers and save in datamodule
                datamodule.train, _ = datamodule.preprocess_data(train_fold, stage="inference")
                datamodule.validation, _ = datamodule.preprocess_data(val_fold, stage="inference")

            # Train the model
            handle_oom = train_kwargs.pop("handle_oom", handle_oom)
            self.train(model, datamodule, handle_oom=handle_oom, **train_kwargs)
            fold_preds = self.predict(test, include_input_features=False)
            pred_idx = fold_preds.index
            if self.config.task == "classification":
                pred_prob_l.append(fold_preds.values[:, : -len(self.config.target)])
            elif self.config.task == "regression":
                pred_prob_l.append(fold_preds.values)
            if verbose:
                logger.info(f"Fold {fold+1}/{cv.get_n_splits()} prediction done")
            self.model.reset_weights()
        pred_df = self._combine_predictions(pred_prob_l, pred_idx, aggregate, weights)
        if return_raw_predictions:
            return pred_df, pred_prob_l
        else:
            return pred_df
