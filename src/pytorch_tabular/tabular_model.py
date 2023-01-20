# Pytorch Tabular
# Author: Manu Joseph <manujoseph@gmail.com>
# For license information, see LICENSE.TXT
"""Tabular Model"""
import copy
import inspect
import os
import warnings
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torchmetrics
from lightning_lite.utilities.seed import seed_everything
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks.gradient_accumulation_scheduler import GradientAccumulationScheduler
from pytorch_lightning.utilities.model_summary import summarize
from rich.progress import track
from sklearn.base import TransformerMixin
from sklearn.preprocessing import LabelEncoder
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
from pytorch_tabular.models.base_model import _GenericModel, BaseModel
from pytorch_tabular.tabular_datamodule import TabularDatamodule
from pytorch_tabular.utils import get_logger, getattr_nested, pl_load

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
    ) -> None:
        """The core model which orchestrates everything from initializing the datamodule, the model, trainer, etc.

        Args:
            config (Optional[Union[DictConfig, str]], optional): Single OmegaConf DictConfig object or
                the path to the yaml file holding all the config parameters. Defaults to None.

            data_config (Optional[Union[DataConfig, str]], optional): DataConfig object or path to the yaml file. Defaults to None.

            model_config (Optional[Union[ModelConfig, str]], optional): A subclass of ModelConfig or path to the yaml file.
                Determines which model to run from the type of config. Defaults to None.

            optimizer_config (Optional[Union[OptimizerConfig, str]], optional): OptimizerConfig object or path to the yaml file.
                Defaults to None.

            trainer_config (Optional[Union[TrainerConfig, str]], optional): TrainerConfig object or path to the yaml file.
                Defaults to None.

            experiment_config (Optional[Union[ExperimentConfig, str]], optional): ExperimentConfig object or path to the yaml file.
                If Provided configures the experiment tracking. Defaults to None.

            model_callable (Optional[Callable], optional): If provided, will override the model callable that will be loaded from the config.
                Typically used when providing Custom Models

            model_state_dict_path (Optional[Union[str, Path]], optional): If provided, will load the state dict after initializing the model from config.
        """
        super().__init__()
        self.exp_manager = ExperimentRunManager()
        if config is None:
            assert (
                (data_config is not None)
                or (model_config is not None)
                or (optimizer_config is not None)
                or (trainer_config is not None)
            ), "If `config` is None, `data_config`, `model_config`, `trainer_config`, and `optimizer_config` cannot be None"
            data_config = self._read_parse_config(data_config, DataConfig)
            model_config = self._read_parse_config(model_config, ModelConfig)
            trainer_config = self._read_parse_config(trainer_config, TrainerConfig)
            optimizer_config = self._read_parse_config(optimizer_config, OptimizerConfig)
            if model_config.task != "ssl":
                assert (
                    data_config.target is not None
                ), f"`target` in data_config should not be None for {model_config.task} task"
            if experiment_config is None:
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
                logger.info("Experiment Tracking is turned off")
                self.track_experiment = False

        self.name, self.uid = self._get_run_name_uid()
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

    def _run_validation(self):
        """Validates the Config params and throws errors if something is wrong"""
        if self.config.task == "classification":
            if len(self.config.target) > 1:
                raise NotImplementedError("Multi-Target Classification is not implemented.")
        if self.config.task == "regression":
            if self.config.target_range is not None:
                if (
                    (len(self.config.target_range) != len(self.config.target))
                    or any([len(range_) != 2 for range_ in self.config.target_range])
                    or any([range_[0] > range_[1] for range_ in self.config.target_range])
                ):
                    raise ValueError(
                        "Targe Range, if defined, should be list tuples of length two(min,max). The length of the list should be equal to hte length of target columns"
                    )
        if self.config.task == "ssl":
            assert (
                not self.config.handle_unknown_categories
            ), "SSL only supports handle_unknown_categories=False. Please set this in your DataConfig"
            assert (
                not self.config.handle_missing_values
            ), "SSL only supports handle_missing_values=False. Please set this in your DataConfig"

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
        """Gets the name of the experiment and increments version by 1

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
        """Sets up the Experiment Tracking Framework according to the choices made in the Experimentconfig"""
        if self.config.log_target == "tensorboard":
            self.logger = pl.loggers.TensorBoardLogger(
                name=self.name, save_dir=self.config.project_name, version=self.uid
            )
        elif self.config.log_target == "wandb":
            self.logger = pl.loggers.WandbLogger(
                name=f"{self.name}_{self.uid}",
                project=self.config.project_name,
                offline=False,
            )
        else:
            raise NotImplementedError(f"{self.config.log_target} is not implemented. Try one of [wandb, tensorboard]")

    def _prepare_callbacks(self, callbacks=None) -> List:
        """Prepares the necesary callbacks to the Trainer based on the configuration

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
            ckpt_name = f"{self.name}-{self.uid}"
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
        if self.config.progress_bar == "rich":
            callbacks.append(RichProgressBar())
        logger.debug(f"Callbacks used: {callbacks}")
        return callbacks

    def _prepare_trainer(self, callbacks: List, max_epochs: int = None, min_epochs: int = None) -> pl.Trainer:
        """Prepares the Trainer object
        Args:
            callbacks (List): A list of callbacks to be used
            max_epochs (int, optional): Maximum number of epochs to train for. Defaults to None.
            min_epochs (int, optional): Minimum number of epochs to train for. Defaults to None.

        Returns:
            pl.Trainer: A PyTorch Lightning Trainer object
        """
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
        return pl.Trainer(
            logger=self.logger,
            callbacks=callbacks,
            **trainer_args_config,
        )

    def _check_and_set_target_transform(self, target_transform):
        if target_transform is not None:
            if isinstance(target_transform, Iterable):
                assert (
                    len(target_transform) == 2
                ), "If `target_transform` is a tuple, it should have and only have forward and backward transformations"
            elif isinstance(target_transform, TransformerMixin):
                pass
            else:
                raise ValueError("`target_transform` should wither be an sklearn Transformer or a tuple of callables.")
        if self.config.task == "classification" and target_transform is not None:
            logger.warning("For classification task, target transform is not used. Ignoring the parameter")
            target_transform = None
        return target_transform

    def _prepare_for_training(self, model, datamodule, callbacks=None, max_epochs=None, min_epochs=None):
        self.callbacks = self._prepare_callbacks(callbacks)
        self.trainer = self._prepare_trainer(self.callbacks, max_epochs, min_epochs)
        self.model = model
        self.datamodule = datamodule

    @classmethod
    def _load_weights(cls, model, path: Union[str, Path]) -> None:
        """Loads the model weights in the specified directory

        Args:
            path (str): The path to the file to load the model from

        Returns:
            None
        """
        ckpt = pl_load(path, map_location=lambda storage, loc: storage)
        if "state_dict" in ckpt.keys():
            model.load_state_dict(ckpt["state_dict"])
        else:
            model.load_state_dict(ckpt)

    @classmethod
    def load_model(cls, dir: str, map_location=None, strict=True):
        """Loads a saved model from the directory

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
        if custom_params.get("custom_optimizer") is not None:
            model_args["optimizer"] = "Adam"  # For compatibility. Not Used
        if custom_params.get("custom_optimizer_params") is not None:
            model_args["optimizer_params"] = {}  # For compatibility. Not Used

        # Initializing with default metrics, losses, and optimizers. Will revert once initialized
        model = model_callable.load_from_checkpoint(
            checkpoint_path=os.path.join(dir, "model.ckpt"),
            map_location=map_location,
            strict=strict,
            **model_args,
        )
        # Updating config with custom parameters for experiment tracking
        if custom_params.get("custom_loss") is not None:
            model.custom_loss = custom_params["custom_loss"]
        if custom_params.get("custom_metrics") is not None:
            model.custom_metrics = custom_params["custom_metrics"]
        if custom_params.get("custom_optimizer") is not None:
            model.custom_optimizer = custom_params["custom_optimizer"]
        if custom_params.get("custom_optimizer_params") is not None:
            model.custom_optimizer_params = custom_params["custom_optimizer_params"]
        model._setup_loss()
        model._setup_metrics()
        tabular_model = cls(config=config, model_callable=model_callable)
        tabular_model.model = model
        tabular_model.custom_model = custom_model
        tabular_model.datamodule = datamodule
        tabular_model.callbacks = callbacks
        tabular_model.trainer = tabular_model._prepare_trainer(callbacks=callbacks)
        tabular_model.trainer.model = model
        tabular_model.logger = logger
        return tabular_model

    @classmethod
    def load_from_checkpoint(cls, dir: str, map_location=None, strict=True):
        """(Deprecated: Use `load_model` instead) Loads a saved model from the directory

        Args:
            dir (str): The directory where the model was saved, along with the checkpoints
            map_location (Union[Dict[str, str], str, device, int, Callable, None]) : If your checkpoint
                saved a GPU model and you now load on CPUs or a different number of GPUs, use this to map
                to the new setup. The behaviour is the same as in torch.load()
            strict (bool) : Whether to strictly enforce that the keys in checkpoint_path match the keys
                returned by this module's state dict. Default: True.

        Returns:
            TabularModel (TabularModel): The saved TabularModel
        """

        warnings.warn(
            "`load_from_checkpoint` is deprecated. Use `load_model` instead.",
            DeprecationWarning,
        )
        return cls.load_model(dir, map_location, strict)

    def prepare_dataloader(
        self,
        train: pd.DataFrame,
        validation: Optional[pd.DataFrame] = None,
        test: Optional[pd.DataFrame] = None,
        train_sampler: Optional[torch.utils.data.Sampler] = None,
        target_transform: Optional[Union[TransformerMixin, Tuple]] = None,
        seed: Optional[int] = 42,
    ) -> TabularDatamodule:
        """Prepares the dataloaders for training and validation.

        Args:
            train (pd.DataFrame): Training Dataframe

            validation (Optional[pd.DataFrame], optional): If provided, will use this dataframe as the validation while training.
                Used in Early Stopping and Logging. If left empty, will use 20% of Train data as validation. Defaults to None.

            test (Optional[pd.DataFrame], optional): If provided, will use as the hold-out data,
                which you'll be able to check performance after the model is trained. Defaults to None.

            train_sampler (Optional[torch.utils.data.Sampler], optional): Custom PyTorch batch samplers which will be passed to the DataLoaders. Useful for dealing with imbalanced data and other custom batching strategies

            target_transform (Optional[Union[TransformerMixin, Tuple(Callable)]], optional): If provided, applies the transform to the target before modelling
                and inverse the transform during prediction. The parameter can either be a sklearn Transformer which has an inverse_transform method, or
                a tuple of callables (transform_func, inverse_transform_func)

            seed (Optional[int], optional): Random seed for reproducibility. Defaults to 42.

        Returns:
            TabularDatamodule: The prepared datamodule
        """
        if test is not None:
            warnings.warn(
                "Providing test data in `fit` is deprecated and will be removed in next major release. Plese use `evaluate` for evaluating on test data"
            )
        logger.info("Preparing the DataLoaders")
        target_transform = self._check_and_set_target_transform(target_transform)

        datamodule = TabularDatamodule(
            train=train,
            validation=validation,
            config=self.config,
            test=test,
            target_transform=target_transform,
            train_sampler=train_sampler,
            seed=seed,
        )
        datamodule.prepare_data()
        datamodule.setup("fit")
        return datamodule

    def prepare_model(
        self,
        datamodule: TabularDatamodule,
        loss: Optional[torch.nn.Module] = None,
        metrics: Optional[List[Callable]] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        optimizer_params: Dict = {},
    ) -> BaseModel:
        """Prepares the model for training.

        Args:
            datamodule (TabularDatamodule): The datamodule

            loss (Optional[torch.nn.Module], optional): Custom Loss functions which are not in standard pytorch library

            metrics (Optional[List[Callable]], optional): Custom metric functions(Callable) which has the
                signature metric_fn(y_hat, y) and works on torch tensor inputs

            optimizer (Optional[torch.optim.Optimizer], optional): Custom optimizers which are a drop in replacements for standard PyToch optimizers.
                This should be the Class and not the initialized object

            optimizer_params (Optional[Dict], optional): The parmeters to initialize the custom optimizer.

        Returns:
            BaseModel: The prepared model

        """
        logger.info(f"Preparing the Model: {self.config._model_name}")
        # Fetching the config as some data specific configs have been added in the datamodule
        self.inferred_config = self._read_parse_config(datamodule.update_config(self.config), InferredConfig)
        model = self.model_callable(
            self.config,
            custom_loss=loss,  # Unused in SSL tasks
            custom_metrics=metrics,  # Unused in SSL tasks
            custom_optimizer=optimizer,
            custom_optimizer_params=optimizer_params,
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
    ) -> pl.Trainer:
        """Trains the model.

        Args:
            model (pl.LightningModule): The PyTorch Lightning model to be trained.

            datamodule (TabularDatamodule): The datamodule

            callbacks (Optional[List[pl.Callback]], optional): List of callbacks to be used during training. Defaults to None.

            max_epochs (Optional[int]): Overwrite maximum number of epochs to be run. Defaults to None.

            min_epochs (Optional[int]): Overwrite minimum number of epochs to be run. Defaults to None.

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
            logger.info("Auto LR Find Started")
            result = self.trainer.tune(self.model, train_loader, val_loader)
            logger.info(
                f"Suggested LR: {result['lr_find'].suggestion()}. For plot and detailed analysis, use `find_learning_rate` method."
            )
            # Parameters in models needs to be initialized again after LR find
            self.model.data_aware_initialization(self.datamodule)
        self.model.train()
        logger.info("Training Started")
        self.trainer.fit(self.model, train_loader, val_loader)
        logger.info("Training the model completed")
        if self.config.load_best:
            self.load_best_model()
        return self.trainer

    def fit(
        self,
        train: Optional[pd.DataFrame],
        validation: Optional[pd.DataFrame] = None,
        test: Optional[pd.DataFrame] = None,  # TODO: Deprecate test in next version
        loss: Optional[torch.nn.Module] = None,
        metrics: Optional[List[Callable]] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        optimizer_params: Dict = {},
        train_sampler: Optional[torch.utils.data.Sampler] = None,
        target_transform: Optional[Union[TransformerMixin, Tuple]] = None,
        max_epochs: Optional[int] = None,
        min_epochs: Optional[int] = None,
        seed: Optional[int] = 42,
        callbacks: Optional[List[pl.Callback]] = None,
        datamodule: Optional[TabularDatamodule] = None,
    ) -> pl.Trainer:
        """The fit method which takes in the data and triggers the training

        Args:
            train (pd.DataFrame): Training Dataframe

            validation (Optional[pd.DataFrame], optional): If provided, will use this dataframe as the validation while training.
                Used in Early Stopping and Logging. If left empty, will use 20% of Train data as validation. Defaults to None.

            test (Optional[pd.DataFrame], optional): If provided, will use as the hold-out data,
                which you'll be able to check performance after the model is trained. Defaults to None.
                DEPRECATED. Will be removed in the next version.

            loss (Optional[torch.nn.Module], optional): Custom Loss functions which are not in standard pytorch library

            metrics (Optional[List[Callable]], optional): Custom metric functions(Callable) which has the
                signature metric_fn(y_hat, y) and works on torch tensor inputs

            optimizer (Optional[torch.optim.Optimizer], optional): Custom optimizers which are a drop in replacements for
                standard PyToch optimizers. This should be the Class and not the initialized object

            optimizer_params (Optional[Dict], optional): The parmeters to initialize the custom optimizer.

            train_sampler (Optional[torch.utils.data.Sampler], optional): Custom PyTorch batch samplers which will be passed
                to the DataLoaders. Useful for dealing with imbalanced data and other custom batching strategies

            target_transform (Optional[Union[TransformerMixin, Tuple(Callable)]], optional): If provided, applies the transform to the
                target before modelling and inverse the transform during prediction. The parameter can either be a sklearn Transformer
                which has an inverse_transform method, or a tuple of callables (transform_func, inverse_transform_func)

            max_epochs (Optional[int]): Overwrite maximum number of epochs to be run. Defaults to None.

            min_epochs (Optional[int]): Overwrite minimum number of epochs to be run. Defaults to None.

            seed: (int): Random seed for reproducibility. Defaults to 42.

            callbacks (Optional[List[pl.Callback]], optional): List of callbacks to be used during training. Defaults to None.

            datamodule (Optional[TabularDatamodule], optional): The datamodule. If provided, will ignore the rest of the parameters
                like train, test etc and use the datamodule. Defaults to None.

        Returns:
            pl.Trainer: The PyTorch Lightning Trainer instance
        """
        assert (
            self.config.task != "ssl"
        ), "`fit` is not valid for SSL task. Please use `pretrain` for semi-supervised learning"
        seed = seed if seed is not None else self.config.seed
        seed_everything(seed)
        if datamodule is None:
            datamodule = self.prepare_dataloader(train, validation, test, train_sampler, target_transform, seed)
        else:
            if train is not None:
                warnings.warn(
                    "train data is provided but datamodule is provided. Ignoring the train data and using the datamodule"
                )
            if test is not None:
                warnings.warn(
                    "Providing test data in `fit` is deprecated and will be removed in next major release. Plese use `evaluate` for evaluating on test data"
                )
        model = self.prepare_model(
            datamodule,
            loss,
            metrics,
            optimizer,
            optimizer_params,
        )

        return self.train(model, datamodule, callbacks, max_epochs, min_epochs)

    def pretrain(
        self,
        train: Optional[pd.DataFrame],
        validation: Optional[pd.DataFrame] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        optimizer_params: Dict = {},
        # train_sampler: Optional[torch.utils.data.Sampler] = None,
        max_epochs: Optional[int] = None,
        min_epochs: Optional[int] = None,
        seed: Optional[int] = 42,
        callbacks: Optional[List[pl.Callback]] = None,
        datamodule: Optional[TabularDatamodule] = None,
    ) -> pl.Trainer:
        """The pretrained method which takes in the data and triggers the training

        Args:
            train (pd.DataFrame): Training Dataframe

            validation (Optional[pd.DataFrame], optional): If provided, will use this dataframe as the validation while training.
                Used in Early Stopping and Logging. If left empty, will use 20% of Train data as validation. Defaults to None.

            optimizer (Optional[torch.optim.Optimizer], optional): Custom optimizers which are a drop in replacements for
                standard PyToch optimizers. This should be the Class and not the initialized object

            optimizer_params (Optional[Dict], optional): The parmeters to initialize the custom optimizer.

            max_epochs (Optional[int]): Overwrite maximum number of epochs to be run. Defaults to None.

            min_epochs (Optional[int]): Overwrite minimum number of epochs to be run. Defaults to None.

            seed: (int): Random seed for reproducibility. Defaults to 42.

            callbacks (Optional[List[pl.Callback]], optional): List of callbacks to be used during training. Defaults to None.

            datamodule (Optional[TabularDatamodule], optional): The datamodule. If provided, will ignore the rest of the
                parameters like train, test etc and use the datamodule. Defaults to None.

        Returns:
            pl.Trainer: The PyTorch Lightning Trainer instance
        """
        assert (
            self.config.task == "ssl"
        ), f"`pretrain` is not valid for {self.config.task} task. Please use `fit` instead."
        seed = seed if seed is not None else self.config.seed
        seed_everything(seed)
        if datamodule is None:
            datamodule = self.prepare_dataloader(
                train,
                validation,
                test=None,
                train_sampler=None,
                target_transform=None,
                seed=seed,
            )
        else:
            if train is not None:
                warnings.warn(
                    "train data is provided but datamodule is provided. Ignoring the train data and using the datamodule"
                )
        model = self.prepare_model(
            datamodule,
            optimizer,
            optimizer_params,
        )

        return self.train(model, datamodule, callbacks, max_epochs, min_epochs)

    def create_finetune_model(
        self,
        task: str,
        head: str,
        head_config: Dict,
        target: Optional[str] = None,
        optimizer_config: Optional[OptimizerConfig] = None,
        trainer_config: Optional[TrainerConfig] = None,
        experiment_config: Optional[ExperimentConfig] = None,
        loss: Optional[torch.nn.Module] = None,
        metrics: Optional[List[Union[Callable, str]]] = None,
        metrics_params: Optional[Dict] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        optimizer_params: Dict = {},
        learning_rate: Optional[float] = None,
        target_range: Optional[Tuple[float, float]] = None,
    ):
        """Creates a new TabularModel model using the pretrained weights and the new task and head
        Args:
            task (str): The task to be performed. One of "regression", "classification"

            head (str): The head to be used for the model. Should be one of the heads defined
                in `pytorch_tabular.models.common.heads`. Defaults to  LinearHead. Choices are:
                [`None`,`LinearHead`,`MixtureDensityHead`].

            head_config (Dict): The config as a dict which defines the head. If left empty,
                will be initialized as default linear head.

            target (Optional[str], optional): The target column name if not provided in the initial pretraining stage.
                Defaults to None.

            optimizer_config (Optional[OptimizerConfig], optional): If provided, will redefine the optimizer for fine-tuning
                stage. Defaults to None.

            trainer_config (Optional[TrainerConfig], optional): If provided, will redefine the trainer for fine-tuning stage.
                Defaults to None.

            experiment_config (Optional[ExperimentConfig], optional): If provided, will redefine the experiment for fine-tuning
                stage. Defaults to None.

            loss (Optional[torch.nn.Module], optional): If provided, will be used as the loss function for the fine-tuning.
                By Default it is MSELoss for regression and CrossEntropyLoss for classification.

            metrics (Optional[List[Callable]], optional): List of metrics (either callables or str) to be used for the
                fine-tuning stage. If str, it should be one of the functional metrics implemented in ``torchmetrics.functional``
                Defaults to None.

            metrics_params (Optional[Dict], optional): The parameters for the metrics in the same order as metrics.
                For eg. f1_score for multi-class needs a parameter `average` to fully define the metric. Defaults to None.

            optimizer (Optional[torch.optim.Optimizer], optional): Custom optimizers which are a drop in replacements for
                standard PyTorch optimizers. If provided, the OptimizerConfig is ignored in favor of this. Defaults to None.

            optimizer_params (Dict, optional): The parameters for the optimizer. Defaults to {}.

            learning_rate (Optional[float], optional): The learning rate to be used. Defaults to 1e-3.

            target_range (Optional[Tuple[float, float]], optional): The target range for the regression task.
                Is ignored for classification. Defaults to None.
        Returns:
            TabularModel (TabularModel): The new TabularModel model for fine-tuning
        """
        config = self.config
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
                logger.info(f"Renaming the experiment run for finetuning as {config['run_name'] + '_finetuned'}")
                config["run_name"] = config["run_name"] + "_finetuned"

        datamodule = self.datamodule
        # Setting the attributes from new config
        datamodule.target = config.target
        datamodule.batch_size = config.batch_size
        datamodule.seed = config.seed
        model_callable = _GenericModel
        inferred_config = self.datamodule.update_config(config)
        inferred_config = OmegaConf.structured(inferred_config)
        # Adding dummy attributes for compatibility. Not used because custom metrics are provided
        if not hasattr(config, "metrics"):
            config.metrics = "dummy"
        if not hasattr(config, "metrics_params"):
            config.metrics_params = {}
        if metrics is not None:
            assert len(metrics) == len(metrics_params), "Number of metrics and metrics_params should be same"
            metrics = [getattr(torchmetrics.functional, m) if isinstance(m, str) else m for m in metrics]
        if task == "regression":
            loss = loss if loss is not None else torch.nn.MSELoss()
            if metrics is None:
                metrics = [torchmetrics.functional.mean_squared_error]
                metrics_params = [{}]
        elif task == "classification":
            loss = loss if loss is not None else torch.nn.CrossEntropyLoss()
            if metrics is None:
                metrics = [torchmetrics.functional.accuracy]
                metrics_params = [
                    {
                        "task": "multiclass",
                        "num_classes": inferred_config.output_dim,
                    }
                ]
            else:
                for i, mp in enumerate(metrics_params):
                    if "task" not in mp:
                        # For classification task, output_dim == number of classses
                        metrics_params[i]["task"] = "multiclass"
                    if "num_classes" not in mp:
                        metrics_params[i]["num_classes"] = inferred_config.output_dim
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
            "custom_optimizer": optimizer,
            "custom_optimizer_params": optimizer_params,
        }
        # Initializing with default metrics, losses, and optimizers. Will revert once initialized
        model = model_callable(
            **model_args,
        )
        tabular_model = TabularModel(config=config)
        tabular_model.model = model
        tabular_model.datamodule = datamodule
        # Setting a flag to identify this as a fine-tune model
        tabular_model._is_finetune_model = True
        return tabular_model

    def finetune(
        self,
        train,
        validation: Optional[pd.DataFrame] = None,
        train_sampler: Optional[torch.utils.data.Sampler] = None,
        target_transform: Optional[Union[TransformerMixin, Tuple]] = None,
        max_epochs: Optional[int] = None,
        min_epochs: Optional[int] = None,
        seed: Optional[int] = 42,
        callbacks: Optional[List[pl.Callback]] = None,
        datamodule: Optional[TabularDatamodule] = None,
        freeze_backbone: bool = False,
    ) -> pl.Trainer:
        """Finetunes the model on the provided data
        Args:
            train (pd.DataFrame): The training data with labels

            validation (Optional[pd.DataFrame], optional): The validation data with labels. Defaults to None.

            train_sampler (Optional[torch.utils.data.Sampler], optional): If provided, will be used as a batch sampler
                for training. Defaults to None.

            target_transform (Optional[Union[TransformerMixin, Tuple]], optional): If provided, will be used to transform
                the target before training and inverse transform the predictions.

            max_epochs (Optional[int], optional): The maximum number of epochs to train for. Defaults to None.

            min_epochs (Optional[int], optional): The minimum number of epochs to train for. Defaults to None.

            seed (Optional[int], optional): The seed to be used for training. Defaults to 42.

            callbacks (Optional[List[pl.Callback]], optional): If provided, will be added to the callbacks for Trainer.
                Defaults to None.

            datamodule (Optional[TabularDatamodule], optional): If provided, will be used as the datamodule for training.
                Defaults to None.

            freeze_backbone (bool, optional): If True, will freeze the backbone by tirning off gradients.
                Defaults to False, which means the pretrained weights are also further tuned during fine-tuning.

        Returns:
            pl.Trainer: The trainer object
        """
        assert (
            self._is_finetune_model
        ), "finetune() can only be called on a finetune model created using `TabularModel.create_finetune_model()`"
        seed = seed if seed is not None else self.config.seed
        seed_everything(seed)
        if datamodule is None:
            target_transform = self._check_and_set_target_transform(target_transform)
            self.datamodule._set_target_transform(target_transform)
            if self.config.task == "classification":
                self.datamodule.label_encoder = LabelEncoder()
                self.datamodule.label_encoder.fit(train[self.config.target[0]])
            elif self.config.task == "regression":
                target_transforms = []
                if target_transform is not None:
                    for col in self.config.target:
                        _target_transform = copy.deepcopy(self.datamodule.target_transform_template)
                        _target_transform.fit(train[col].values.reshape(-1, 1))
                        target_transforms.append(_target_transform)
                self.datamodule.target_transforms = target_transforms
            self.datamodule.train = self.datamodule._prepare_inference_data(train)
            if validation is not None:
                self.datamodule.validation = self.datamodule._prepare_inference_data(validation)
            else:
                self.datamodule.validation = None
            self.datamodule.train_sampler = train_sampler
            datamodule = self.datamodule
        else:
            if train is not None:
                warnings.warn(
                    "train data is provided but datamodule is provided. Ignoring the train data and using the datamodule"
                )
        if freeze_backbone:
            for param in self.model.backbone.parameters():
                param.requires_grad = False
        return self.train(
            self.model,
            datamodule,
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
    ) -> Tuple[float, pd.DataFrame]:
        """Enables the user to do a range test of good initial learning rates, to reduce the amount of guesswork in picking a good starting learning rate.

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
        lr_finder = self.trainer.tuner.lr_find(
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
        return new_lr, pd.DataFrame(lr_finder.results)

    def evaluate(
        self,
        test: Optional[pd.DataFrame] = None,
        test_loader: Optional[torch.utils.data.DataLoader] = None,
        ckpt_path: Optional[Union[str, Path]] = None,
        verbose: bool = True,
    ) -> Union[dict, list]:
        """Evaluates the dataframe using the loss and metrics already set in config

        Args:
            test (Optional[pd.DataFrame]): The dataframe to be evaluated. If not provided, will try to use the
                test provided during fit. If that was also not provided will return an empty dictionary

            test_loader (Optional[torch.utils.data.DataLoader], optional): The dataloader to be used for evaluation.
                If provided, will use the dataloader instead of the test dataframe or the test data provided during fit.
                DEPRECATION: providing test data during fit is deprecated and will be removed in a future release.
                Defaults to None.

            ckpt_path (Optional[Union[str, Path]], optional): The path to the checkpoint to be loaded. If not provided, will try to use the
                best checkpoint during training.

            verbose (bool, optional): If true, will print the results. Defaults to True.
        Returns:
            The final test result dictionary.
        """
        if test_loader is None and test is None:
            warnings.warn(
                "Providing test in fit is deprecated. Not providing `test` or `test_loader` in `evaluate` will cause an error in a future release."
            )
        if test_loader is None:
            if test is not None:
                test_loader = self.datamodule.prepare_inference_dataloader(test)
            elif self.datamodule.test is not None:
                warnings.warn(
                    "Providing test in fit is deprecated. Not providing `test` or `test_loader` in `evaluate` will cause an error in a future release."
                )
                test_loader = self.datamodule.test_dataloader()
            else:
                return {}
        result = self.trainer.test(
            model=self.model,
            dataloaders=test_loader,
            ckpt_path=ckpt_path,
            verbose=verbose,
        )
        return result

    def predict(
        self,
        test: pd.DataFrame,
        quantiles: Optional[List] = [0.25, 0.5, 0.75],
        n_samples: Optional[int] = 100,
        ret_logits=False,
        include_input_features: bool = True,
        device: Optional[torch.device] = None,
    ) -> pd.DataFrame:
        """Uses the trained model to predict on new data and return as a dataframe

        Args:
            test (pd.DataFrame): The new dataframe with the features defined during training
            quantiles (Optional[List]): For probabilistic models like Mixture Density Networks, this specifies
                the different quantiles to be extracted apart from the `central_tendency` and added to the dataframe.
                For other models it is ignored. Defaults to [0.25, 0.5, 0.75]
            n_samples (Optional[int]): Number of samples to draw from the posterior to estimate the quantiles.
                Ignored for non-probabilistic models. Defaults to 100
            ret_logits (bool): Flag to return raw model outputs/logits except the backbone features along
                with the dataframe. Defaults to False
            include_input_features (bool): Flag to include the input features in the returned dataframe.
                Defaults to True

        Returns:
            pd.DataFrame: Returns a dataframe with predictions and features (if `include_input_features=True`).
                If classification, it returns probabilities and final prediction
        """
        warnings.warn(
            "Default for `include_input_features` will change from True to False in the next release. Please set it explicitly.",
            DeprecationWarning,
        )
        assert all([q <= 1 and q >= 0 for q in quantiles]), "Quantiles should be a decimal between 0 and 1"
        if device is not None:
            if isinstance(device, str):
                device = torch.device(device)
            if self.model.device != device:
                model = self.model.to(device)
            else:
                model = self.model
        else:
            model = self.model
        model.eval()
        inference_dataloader = self.datamodule.prepare_inference_dataloader(test)
        point_predictions = []
        quantile_predictions = []
        logits_predictions = defaultdict(list)
        is_probabilistic = hasattr(model.hparams, "_probabilistic") and model.hparams._probabilistic
        for batch in track(inference_dataloader, description="Generating Predictions..."):
            for k, v in batch.items():
                if isinstance(v, list) and (len(v) == 0):
                    # Skipping empty list
                    continue
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
                    # if k == "backbone_features":
                    #     continue
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
        if include_input_features:
            pred_df = test.copy()
        else:
            pred_df = pd.DataFrame(index=test.index)
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
                            pred_df[f"{target_col}_q{int(q*100)}"] = self.datamodule.target_transforms[
                                i
                            ].inverse_transform(quantile_predictions[:, j, i].reshape(-1, 1))
                else:
                    pred_df[f"{target_col}_prediction"] = point_predictions[:, i]
                    if is_probabilistic:
                        for j, q in enumerate(quantiles):
                            pred_df[f"{target_col}_q{int(q*100)}"] = quantile_predictions[:, j, i].reshape(-1, 1)

        elif self.config.task == "classification":
            point_predictions = nn.Softmax(dim=-1)(point_predictions).numpy()
            for i, class_ in enumerate(self.datamodule.label_encoder.classes_):
                pred_df[f"{class_}_probability"] = point_predictions[:, i]
            pred_df["prediction"] = self.datamodule.label_encoder.inverse_transform(
                np.argmax(point_predictions, axis=1)
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

    def load_best_model(self) -> None:
        """Loads the best model after training is done"""
        if self.trainer.checkpoint_callback is not None:
            logger.info("Loading the best model")
            ckpt_path = self.trainer.checkpoint_callback.best_model_path
            if ckpt_path != "":
                logger.debug(f"Model Checkpoint: {ckpt_path}")
                ckpt = pl_load(ckpt_path, map_location=lambda storage, loc: storage)
                self.model.load_state_dict(ckpt["state_dict"])
            else:
                logger.warning("No best model available to load. Did you run it more than 1 epoch?...")
        else:
            logger.warning("No best model available to load. Checkpoint Callback needs to be enabled for this to work")

    def save_datamodule(self, dir: str) -> None:
        """Saves the datamodule in the specified directory

        Args:
            dir (str): The path to the directory to save the datamodule
        """
        joblib.dump(self.datamodule, os.path.join(dir, "datamodule.sav"))

    def save_config(self, dir: str) -> None:
        """Saves the config in the specified directory"""
        with open(os.path.join(dir, "config.yml"), "w") as fp:
            OmegaConf.save(self.config, fp, resolve=True)

    def save_model(self, dir: str) -> None:
        """Saves the model and checkpoints in the specified directory

        Args:
            dir (str): The path to the directory to save the model
        """
        if os.path.exists(dir) and (os.listdir(dir)):
            logger.warning("Directory is not empty. Overwriting the contents.")
            for f in os.listdir(dir):
                os.remove(os.path.join(dir, f))
        os.makedirs(dir, exist_ok=True)
        self.save_config(dir)
        self.save_datamodule(dir)
        if hasattr(self.config, "log_target") and self.config.log_target is not None:
            joblib.dump(self.logger, os.path.join(dir, "exp_logger.sav"))
        if hasattr(self, "callbacks"):
            joblib.dump(self.callbacks, os.path.join(dir, "callbacks.sav"))
        self.trainer.save_checkpoint(os.path.join(dir, "model.ckpt"))
        custom_params = {}
        custom_params["custom_loss"] = self.model.custom_loss
        custom_params["custom_metrics"] = self.model.custom_metrics
        custom_params["custom_optimizer"] = self.model.custom_optimizer
        custom_params["custom_optimizer_params"] = self.model.custom_optimizer_params
        joblib.dump(custom_params, os.path.join(dir, "custom_params.sav"))
        if self.custom_model:
            joblib.dump(self.model_callable, os.path.join(dir, "custom_model_callable.sav"))

    def save_weights(self, path: Union[str, Path]) -> None:
        """Saves the model weights in the specified directory

        Args:
            path (str): The path to the file to save the model
        """
        torch.save(self.model.state_dict(), path)

    def load_weights(self, path: Union[str, Path]) -> None:
        """Loads the model weights in the specified directory

        Args:
            path (str): The path to the file to load the model from
        """
        self._load_weights(self.model, path)

    # TODO Need to test ONNX export
    def save_model_for_inference(
        self,
        path: Union[str, Path],
        kind: str = "pytorch",
        onnx_export_params: Dict = dict(opset_version=12),
    ) -> bool:
        """Saves the model for inference

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
            x = dict(continuous=cont, categorical=cat)
            torch.onnx.export(self.model, x, str(path), **onnx_export_params)
            return True
        else:
            raise ValueError("`kind` must be either pytorch or onnx")

    def summary(self, max_depth: int = -1) -> None:
        """Prints a summary of the model

        Args:
            max_depth (int): The maximum depth to traverse the modules and displayed in the summary.
                Defaults to -1, which means will display all the modules.
        """
        print(summarize(self.model, max_depth=max_depth))

    def __str__(self) -> str:
        return self.summary()
