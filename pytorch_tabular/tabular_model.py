# Pytorch Tabular
# Author: Manu Joseph <manujoseph@gmail.com>
# For license information, see LICENSE.TXT
"""Tabular Model"""
import logging
import os
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
from sklearn.base import TransformerMixin
from omegaconf.dictconfig import DictConfig
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.cloud_io import load as pl_load
from omegaconf import OmegaConf
from torch import nn
import joblib

import pytorch_tabular.models as models
from pytorch_tabular.config import (
    DataConfig,
    ExperimentConfig,
    ExperimentRunManager,
    ModelConfig,
    OptimizerConfig,
    TrainerConfig,
)
from pytorch_tabular.tabular_datamodule import TabularDatamodule

logger = logging.getLogger(__name__)


class TabularModel:
    def __init__(
        self,
        config: Optional[DictConfig] = None,
        data_config: Optional[Union[DataConfig, str]] = None,
        model_config: Optional[Union[ModelConfig, str]] = None,
        optimizer_config: Optional[Union[OptimizerConfig, str]] = None,
        trainer_config: Optional[Union[TrainerConfig, str]] = None,
        experiment_config: Optional[Union[ExperimentConfig, str]] = None,
    ) -> None:
        """The core model which orchestrates everything from initializing the datamodule, the model, trainer, etc.

        Args:
            config (Optional[Union[DictConfig, str]], optional): Single OmegaConf DictConfig object or
            the path to the yaml file holding all the config parameters. Defaults to None.
            data_config (Optional[Union[DataConfig, str]], optional): DataConfig object or str to the yaml file. Defaults to None.
            model_config (Optional[Union[ModelConfig, str]], optional): A subclass of ModelConfig or str to the yaml file.
            DEtermines which model to run from the type of config. Defaults to None.
            optimizer_config (Optional[Union[OptimizerConfig, str]], optional): OptimizerConfig object or str to the yaml file.
            Defaults to None.
            trainer_config (Optional[Union[TrainerConfig, str]], optional): TrainerConfig object or str to the yaml file.
            Defaults to None.
            experiment_config (Optional[Union[ExperimentConfig, str]], optional): ExperimentConfig object or str to the yaml file.
            If Provided configures the experiment tracking. Defaults to None.
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
            # Re-routing to Categorical embedding Model if embed_categorical is true for NODE
            if (
                hasattr(model_config, "_model_name")
                and (model_config._model_name == "NODEModel")
                and (model_config.embed_categorical)
                and ("CategoryEmbedding" not in model_config._model_name)
            ):
                model_config._model_name = (
                    "CategoryEmbedding" + model_config._model_name
                )
            trainer_config = self._read_parse_config(trainer_config, TrainerConfig)
            optimizer_config = self._read_parse_config(
                optimizer_config, OptimizerConfig
            )
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
                experiment_config = self._read_parse_config(
                    experiment_config, ExperimentConfig
                )
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
            if not hasattr(config, "log_target") and (config.log_target is not None):
                experiment_config = OmegaConf.structured(experiment_config)
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
        self.model_callable = getattr(
            getattr(models, self.config._module_src), self.config._model_name
        )
        self._run_validation()

    def _run_validation(self):
        if self.config.task == "classification":
            if len(self.config.target) > 1:
                raise NotImplementedError(
                    "Multi-Target Classification is not implemented."
                )
        if self.config.task == "regression":
            if self.config.target_range is not None:
                if (
                    (len(self.config.target_range) != len(self.config.target))
                    or any([len(range_) != 2 for range_ in self.config.target_range])
                    or any(
                        [range_[0] > range_[1] for range_ in self.config.target_range]
                    )
                ):
                    raise ValueError(
                        "Targe Range, if defined, should be list tuples of length two(min,max). The length of the list should be equal to hte length of target columns"
                    )

    def _read_parse_config(self, config, cls):
        if isinstance(config, str):
            if os.path.exists(config):
                _config = OmegaConf.load(config)
                if cls == ModelConfig:
                    cls = getattr(
                        getattr(models, _config._module_src), _config._config_name
                    )
                config = cls(
                    **{
                        k: v
                        for k, v in _config.items()
                        if (k in cls.__dataclass_fields__.keys())
                        and (cls.__dataclass_fields__[k].init)
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
        name = self.config.run_name if self.config.run_name else f"{self.config.task}"
        uid = self.exp_manager.update_versions(name)
        return name, uid

    def _setup_experiment_tracking(self):
        """Sets up the Experiment Tracking Framework according to the choices made in the Experimentconfig

        Raises:
            NotImplementedError: Raises an Error for invalid choices of log_target
        """
        if self.config.log_target == "tensorboard":
            self.logger = pl.loggers.TensorBoardLogger(
                name=self.name, save_dir="tensorboard_logs", version=self.uid
            )
        elif self.config.log_target == "wandb":
            self.logger = pl.loggers.WandbLogger(
                name=f"{self.name}_{self.uid}",
                project=self.config.project_name,
                offline=False,
            )
        else:
            raise NotImplementedError(
                f"{self.config.log_target} is not implemented. Try one of [wandb, tensorboard]"
            )

    def _prepare_callbacks(self) -> List:
        """Prepares the necesary callbacks to the Trainer based on the configuration

        Returns:
            List: A list of callbacks
        """
        callbacks = []
        if self.config.early_stopping is not None:
            early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(
                monitor=self.config.early_stopping,
                min_delta=self.config.early_stopping_min_delta,
                patience=self.config.early_stopping_patience,
                verbose=False,
                mode=self.config.early_stopping_mode,
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
            )
            callbacks.append(model_checkpoint)
            self.config.checkpoint_callback = True
        else:
            self.config.checkpoint_callback = False
        logger.debug(f"Callbacks used: {callbacks}")
        return callbacks

    def data_aware_initialization(self):
        # Need a big batch to initialize properly
        alt_loader = self.datamodule.train_dataloader(batch_size=1024)
        batch = next(iter(alt_loader))
        for k, v in batch.items():
            if isinstance(v, list) and (len(v) == 0):
                # Skipping empty list
                continue
            # batch[k] = v.to("cpu" if self.config.gpu == 0 else "cuda")
            batch[k] = v.to(self.model.device)

        # single forward pass to initialize the ODST
        with torch.no_grad():
            self.model(batch)

    def _prepare_dataloader(self, train, validation, test, target_transform=None):
        logger.info("Preparing the DataLoaders...")
        self.datamodule = TabularDatamodule(
            train=train,
            config=self.config,
            test=test,
            target_transform=target_transform,
        )
        self.datamodule.prepare_data()
        self.datamodule.setup("fit")
        train_loader = self.datamodule.train_dataloader()
        val_loader = self.datamodule.val_dataloader()
        return train_loader, val_loader

    def _prepare_model(self, loss, metrics, optimizer, optimizer_params):
        logger.info(f"Preparing the Model: {self.config._model_name}...")
        # Fetching the config as some data specific configs have been added in the datamodule
        self.config = self.datamodule.config
        self.model = self.model_callable(
            self.config,
            custom_loss=loss,
            custom_metrics=metrics,
            custom_optimizer=optimizer,
            custom_optimizer_params=optimizer_params,
        )
        # Data Aware Initialization (NODE)
        if self.config._model_name in ["CategoryEmbeddingNODEModel", "NODEModel"]:
            self.data_aware_initialization()

    def _prepare_trainer(self):
        logger.info("Preparing the Trainer...")
        trainer_args = vars(pl.Trainer()).keys()
        trainer_args_config = {
            k: v for k, v in self.config.items() if k in trainer_args
        }
        # For some weird reason, checkpoint_callback is not appearing in the Trainer vars
        trainer_args_config["checkpoint_callback"] = self.config.checkpoint_callback
        self.trainer = pl.Trainer(
            logger=self.logger,
            callbacks=self.callbacks,
            **trainer_args_config,
        )

    def load_best_model(self):
        if self.trainer.checkpoint_callback is not None:
            logger.info("Loading the best model...")
            ckpt_path = self.trainer.checkpoint_callback.best_model_path
            logger.debug(f"Model Checkpoint: {ckpt_path}")
            ckpt = pl_load(ckpt_path, map_location=lambda storage, loc: storage)
            self.model.load_state_dict(ckpt["state_dict"])
        else:
            logger.info("No best model available to load. Did you run it more than 1 epoch?...")

    def fit(
        self,
        train: pd.DataFrame,
        valid: Optional[pd.DataFrame] = None,
        test: Optional[pd.DataFrame] = None,
        loss: Optional[torch.nn.Module] = None,
        metrics: Optional[List[Callable]] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        optimizer_params: Dict = {},
        target_transform: Optional[Union[TransformerMixin, Tuple]] = None,
    ) -> None:
        """The fit method which takes in the data and triggers the training

        Args:
            train (pd.DataFrame): Training Dataframe
            valid (Optional[pd.DataFrame], optional): If provided, will use this dataframe as the validation while training.
            Used in Early Stopping and Logging. If left empty, will use 20% of Train data as validation. Defaults to None.
            test (Optional[pd.DataFrame], optional): If provided, will use as the hold-out data,
            which you'll be able to check performance after the model is trained. Defaults to None.
            loss (Optional[torch.nn.Module], optional): Custom Loss functions which are not in standard pytorch library
            metrics (Optional[List[Callable]], optional): Custom metric functions(Callable) which has the signature metric_fn(y_hat, y)
            optimizer (Optional[torch.optim.Optimizer], optional): Custom optimizers which are a drop in replacements for standard PyToch optimizers.
            This should be the Class and not the initialized object
            optimizer_params (Optional[Dict], optional): The parmeters to initialize the custom optimizer.
            target_transform (Optional[Union[TransformerMixin, Tuple(Callable)]], optional): If provided, applies the transform to the target before modelling
            and inverse the transform during prediction. The parameter can either be a sklearn Transformer which has an inverse_transform method, or
            a tuple of callables (transform_func, inverse_transform_func)
        """
        if (target_transform is not None):
            if isinstance(target_transform, Iterable):
                assert (
                    len(target_transform) == 2
                ), "If `target_transform` is a tuple, it should have and only have forward and backward transformations"
            elif isinstance(target_transform, TransformerMixin):
                pass
            else:
                raise ValueError(
                    "`target_transform` should wither be an sklearn Transformer or a tuple of callables."
                )
        if self.config.task=="classification" and target_transform is not None:
            logger.warning("For classification task, target transform is not used. Ignoring the parameter")
            target_transform = None
        train_loader, val_loader = self._prepare_dataloader(
            train, valid, test, target_transform
        )
        self._prepare_model(loss, metrics, optimizer, optimizer_params)

        if self.track_experiment and self.config.log_target == "wandb":
            self.logger.watch(
                self.model, log=self.config.exp_watch, log_freq=self.config.exp_log_freq
            )
        self.callbacks = self._prepare_callbacks()
        self._prepare_trainer()

        if self.config.auto_lr_find and (not self.config.fast_dev_run):
            self.trainer.tune(self.model, train_loader, val_loader)

        # Parameters in NODE needs to be initialized again
        if self.config._model_name in ["CategoryEmbeddingNODEModel", "NODEModel"]:
            self.data_aware_initialization()

        self.trainer.fit(self.model, train_loader, val_loader)
        logger.info("Training the model completed...")
        if self.config.load_best:
            self.load_best_model()

    def evaluate(self, test: Optional[pd.DataFrame]) -> Union[dict, list]:
        """Evaluates the dataframe using the loss and metrics already set in config

        Args:
            test (Optional[pd.DataFrame]): The dataframe to be evaluated. If not provided, will try to use the
            test provided during fit. If that was also not provided will return an empty dictionary

        Returns:
            Union[dict, list]: The final test result dictionary.
        """
        if test is not None:
            test_loader = self.datamodule.prepare_inference_dataloader(test)
        elif self.test is not None:
            test_loader = self.datamodule.test_dataloader()
        else:
            return {}
        result = self.trainer.test(
            test_dataloaders=test_loader,
            ckpt_path="best" if self.config.checkpoints else None,
        )
        return result

    def predict(self, test: pd.DataFrame) -> pd.DataFrame:
        """Uses the trained model to predict on new data and return as a dataframe

        Args:
            test (pd.DataFrame): The new dataframe with the features defined during training

        Returns:
            pd.DataFrame: Returns a dataframe with predictions and features.
            If classification, it returns probabilities and final prediction
        """
        inference_dataloader = self.datamodule.prepare_inference_dataloader(test)
        predictions = []
        for sample in inference_dataloader:
            for k, v in sample.items():
                if isinstance(v, list) and (len(v) == 0):
                    # Skipping empty list
                    continue
                sample[k] = v.to("cpu" if self.config.gpu == 0 else "cuda")
            y_hat = self.model(sample)
            predictions.append(y_hat.detach().cpu())
        predictions = torch.cat(predictions, dim=0)
        if predictions.ndim == 1:
            predictions = predictions.unsqueeze(-1)
        pred_df = test.copy()
        if self.config.task == "regression":
            predictions = predictions.numpy()
            for i, target_col in enumerate(self.config.target):
                if self.datamodule.do_target_transform:
                    if self.config.target[i] in pred_df.columns:
                        pred_df[self.config.target[i]] = self.datamodule.target_transforms[i].inverse_transform(pred_df[self.config.target[i]].values.reshape(-1,1))
                    pred_df[f"{target_col}_prediction"] = self.datamodule.target_transforms[i].inverse_transform(predictions[:, i].reshape(-1,1))
                else:
                    pred_df[f"{target_col}_prediction"] = predictions[:, i]
        elif self.config.task == "classification":
            predictions = nn.Softmax(dim=-1)(predictions).numpy()
            for i, class_ in enumerate(self.datamodule.label_encoder.classes_):
                pred_df[f"{class_}_probability"] = predictions[:, i]
            pred_df[f"prediction"] = self.datamodule.label_encoder.inverse_transform(
                np.argmax(predictions, axis=1)
            )
        return pred_df

    def save_model(self, dir: str):
        if os.path.exists(dir) and (os.listdir(dir)):
            logger.warning("Directory is not empty. Overwriting the contents.")
            for f in os.listdir(dir):
                os.remove(os.path.join(dir, f))
        os.makedirs(dir, exist_ok=True)
        with open(os.path.join(dir, "config.yml"), "w") as fp:
            OmegaConf.save(self.config, fp, resolve=True)
        joblib.dump(self.datamodule, os.path.join(dir, "datamodule.sav"))
        if hasattr(self.config, "log_target") and self.config.log_target is not None:
            joblib.dump(self.logger, os.path.join(dir, "exp_logger.sav"))
        if hasattr(self, "callbacks"):
            joblib.dump(self.callbacks, os.path.join(dir, "callbacks.sav"))
        self.trainer.save_checkpoint(os.path.join(dir, "model.ckpt"))
        # joblib.dump(self.trainer, os.path.join(dir, "trainer.sav"))

    @classmethod
    def load_from_checkpoint(cls, dir: str):
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
        else:
            callbacks = []
        model_callable = getattr(
            getattr(models, config._module_src), config._model_name
        )
        model = model_callable.load_from_checkpoint(
            checkpoint_path=os.path.join(dir, "model.ckpt")
        )
        # trainer = joblib.load(os.path.join(dir, "trainer.sav"))
        tabular_model = cls(config=config)
        tabular_model.model = model
        tabular_model.datamodule = datamodule
        tabular_model.callbacks = callbacks
        tabular_model._prepare_trainer()
        tabular_model.trainer.model = model
        tabular_model.logger = logger
        return tabular_model
