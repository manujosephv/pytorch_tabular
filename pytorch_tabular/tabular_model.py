# Pytorch Tabular
# Author: Manu Joseph <manujoseph@gmail.com>
# For license information, see LICENSE.TXT
"""Tabular Model"""
import logging
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from torch import nn

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
        data_config: DataConfig,
        model_config: ModelConfig,
        optimizer_config: OptimizerConfig,
        trainer_config: TrainerConfig,
        experiment_config: Optional[ExperimentConfig] = None,
    ) -> None:
        """The core model which orchestrates everything from initializing the datamodule, the model, trainer, etc.

        Args:
            data_config (DataConfig): Data Parameters. Refer to DataConfig documentation
            model_config (ModelConfig): Model Parameters. Can be any subclass of Model Config.
            Depending on this class, the model determines to run the corresponding model on the data supplied.
            Refer to individual ModelConfig documentation
            optimizer_config (OptimizerConfig): Optimizer Parameters. Refer to OptimizerConfig documentation
            trainer_config (TrainerConfig): Trainer Parameters. Refer to TrainerConfig Documentation
            experiment_config (Optional[ExperimentConfig], optional): Experiment Tracking Parameters.
            If Provided configures the experiment tracking. Refer to ExperimentConfig documentation. Defaults to None.
        """
        super().__init__()
        data_config = OmegaConf.structured(data_config)
        model_config = OmegaConf.structured(model_config)
        # Re-routing to Categorical embedding Model if embed_categorical is true for NODE
        if (model_config._model_name == "NODEModel") and (
            model_config.embed_categorical
        ):
            model_config._model_name = "CategoryEmbedding" + model_config._model_name
        trainer_config = OmegaConf.structured(trainer_config)
        optimizer_config = OmegaConf.structured(optimizer_config)
        self.exp_manager = ExperimentRunManager()
        if experiment_config is None:
            logger.info("Experiment Tracking is turned off")
            self.track_experiment = False
            self.config = OmegaConf.merge(
                OmegaConf.to_container(data_config),
                OmegaConf.to_container(model_config),
                OmegaConf.to_container(trainer_config),
                OmegaConf.to_container(optimizer_config),
            )
            self.name, self.uid = self._get_run_name_uid()
            self.logger = None
        else:
            experiment_config = OmegaConf.structured(experiment_config)
            self.track_experiment = True
            self.config = OmegaConf.merge(
                OmegaConf.to_container(data_config),
                OmegaConf.to_container(model_config),
                OmegaConf.to_container(trainer_config),
                OmegaConf.to_container(experiment_config),
                OmegaConf.to_container(optimizer_config),
            )
            self.name, self.uid = self._get_run_name_uid()
            self._setup_experiment_tracking()
        self.model_callable = getattr(
            getattr(models, model_config._module_src), model_config._model_name
        )
        self._run_vaidation()

    def _run_vaidation(self):
        if self.config.task == "classification":
            if len(self.config.target) > 1:
                raise NotImplementedError(
                    "Multi-Target Classification is not implemented."
                )
        if self.config.task == "regression":
            if self.config.target_range is not None:
                if (
                    (len(self.config.target_range) != len(self.config.target))
                    or any([len(range_) != 2 for range_ in self.hparams.target_range])
                    or any(
                        [range_[0] > range_[1] for range_ in self.hparams.target_range]
                    )
                ):
                    raise ValueError(
                        "Targe Range, if defined, should be list tuples of length two(min,max). The length of the list should be equal to hte length of target columns"
                    )


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

    def _prepare_dataloader(self, train, validation, test):
        logger.info("Preparing the DataLoaders...")
        self.datamodule = TabularDatamodule(train=train, config=self.config, test=test)
        self.datamodule.prepare_data()
        self.datamodule.setup("fit")
        train_loader = self.datamodule.train_dataloader()
        val_loader = self.datamodule.val_dataloader()
        return train_loader, val_loader

    def _prepare_model(self):
        logger.info(f"Preparing the Model: {self.config._model_name}...")
        # Fetching the config as some data specific configs have been added in the datamodule
        self.config = self.datamodule.config
        self.model = self.model_callable(self.config)
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

    def fit(
        self,
        train: pd.DataFrame,
        valid: Optional[pd.DataFrame] = None,
        test: Optional[pd.DataFrame] = None,
    ) -> None:
        """The fit method which takes in the data and triggers the training

        Args:
            train (pd.DataFrame): Training Dataframe
            valid (Optional[pd.DataFrame], optional): If provided, will use this dataframe as the validation while training.
            Used in Early Stopping and Logging. If left empty, will use 20% of Train data as validation. Defaults to None.
            test (Optional[pd.DataFrame], optional): If provided, will use as the hold-out data,
            which you'll be able to check performance after the model is trained. Defaults to None.
        """
        train_loader, val_loader = self._prepare_dataloader(train, valid, test)
        self._prepare_model()

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
                pred_df[f"{target_col}_prediction"] = predictions[:, i]
        elif self.config.task == "classification":
            predictions = nn.Softmax(dim=-1)(predictions).numpy()
            for i, class_ in enumerate(self.datamodule.label_encoder.classes_):
                pred_df[f"{class_}_probability"] = predictions[:, i]
            pred_df[f"prediction"] = self.datamodule.label_encoder.inverse_transform(
                np.argmax(predictions, axis=1)
            )
        return pred_df
