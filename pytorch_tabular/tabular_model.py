from typing import Optional
import torch
from config.config import (
    DataConfig,
    ExperimentConfig,
    ExperimentRunManager,
    ModelConfig,
    OptimizerConfig,
    TrainerConfig,
)
import pandas as pd
from omegaconf import OmegaConf
from pytorch_tabular.tabular_datamodule import TabularDatamodule
import pytorch_tabular.models as models
from pytorch_tabular.models.category_embedding import CategoryEmbeddingModel
import pytorch_lightning as pl
import numpy as np


class TabularModel:
    def __init__(
        self,
        data_config: DataConfig,
        model_config: ModelConfig,
        optimizer_config: OptimizerConfig,
        trainer_config: TrainerConfig,
        experiment_config: Optional[ExperimentConfig] = None,
    ) -> None:
        super().__init__()
        data_config = OmegaConf.structured(data_config)
        model_config = OmegaConf.structured(model_config)
        trainer_config = OmegaConf.structured(trainer_config)
        optimizer_config = OmegaConf.structured(optimizer_config)
        self.exp_manager = ExperimentRunManager()
        if experiment_config is None:
            self.track_experiment = False
            self.config = OmegaConf.merge(
                OmegaConf.to_container(data_config),
                OmegaConf.to_container(model_config),
                OmegaConf.to_container(trainer_config),
                OmegaConf.to_container(optimizer_config),
            )
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
            self._setup_experiment_tracking()
        self.model_callable = getattr(
            getattr(models, model_config._module_src), model_config._model_name
        )

    def _get_run_name_uid(self):
        name = self.config.run_name if self.config.run_name else f"{self.config.task}"
        uid = self.exp_manager.update_versions(name)
        return name, uid

    def _setup_experiment_tracking(self):
        name, uid = self._get_run_name_uid()
        if self.config.log_target == "tensorboard":
            self.logger = pl.loggers.TensorBoardLogger(
                name=name, save_dir="tensorboard_logs", version=uid
            )
        elif self.config.log_target == "wandb":
            self.logger = pl.loggers.WandbLogger(
                name=f"{name}_{uid}", project=self.config.project_name, offline=False
            )
        else:
            raise NotImplementedError(
                f"{self.config.log_target} is not implemented. Try one of [wandb, tensorboard]"
            )

    def _prepare_callbacks(self):
        name, uid = self._get_run_name_uid()
        self.config.checkpoint_callback = True if self.config.checkpoints else False
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
            ckpt_name = f"{name}-{uid}"
            ckpt_name = ckpt_name.replace(" ", "_") + "_{epoch}-{valid_loss:.2f}"
            model_checkpoint = pl.callbacks.ModelCheckpoint(
                monitor=self.config.checkpoints,
                dirpath=self.config.checkpoints_path,
                filename=ckpt_name,
                save_top_k=self.config.checkpoints_save_top_k,
                mode=self.config.checkpoints_mode,
            )
            callbacks.append(model_checkpoint)
        return callbacks

    def fit(
        self,
        train: pd.DataFrame,
        valid: Optional[pd.DataFrame] = None,
        test: Optional[pd.DataFrame] = None,
    ):
        self.datamodule = TabularDatamodule(train=train, config=self.config, test=test)
        self.datamodule.prepare_data()
        # splits/transforms
        self.datamodule.setup("fit")
        train_loader = self.datamodule.train_dataloader()
        val_loader = self.datamodule.val_dataloader()
        # Fetching the config as some data specific configs have been added in the datamodule
        config = self.datamodule.config
        self.model = self.model_callable(config)
        if self.track_experiment and self.config.log_target == "wandb":
            self.logger.watch(
                self.model, log=self.config.exp_watch, log_freq=self.config.exp_log_freq
            )
        callbacks = self._prepare_callbacks()
        trainer_args = vars(pl.Trainer()).keys()
        self.trainer = pl.Trainer(
            logger=self.logger,
            callbacks=callbacks,
            **{k: v for k, v in config.items() if k in trainer_args},
        )
        if self.config.auto_lr_find and (not self.config.fast_dev_run):
            self.trainer.tune(self.model, train_loader, val_loader)
        self.trainer.fit(self.model, train_loader, val_loader)

    def evaluate(self, test: Optional[pd.DataFrame]):
        if test is not None:
            test_loader = self.datamodule.prepare_inference_dataloader(test)
        else:
            test_loader = self.datamodule.test_dataloader()
        result = self.trainer.test(
            test_dataloaders=test_loader,
            ckpt_path="best" if self.config.checkpoints else None,
        )
        return result

    def predict(self, test: pd.DataFrame):
        inference_dataloader = self.datamodule.prepare_inference_dataloader(test)
        predictions = []
        for sample in inference_dataloader:
            for k, v in sample.items():
                sample[k] = v.to("cpu" if self.config.gpu == 0 else "cuda")
            y_hat = self.model(sample)
            predictions.append(y_hat.detach().cpu())
        predictions = torch.cat(predictions, dim=0).numpy()
        pred_df = test.copy()
        if self.config.task == "regression":
            for i, target_col in enumerate(self.config.target):
                pred_df[f"{target_col}_prediction"] = predictions[:, i]
        elif self.config.task == "classification":
            for i, class_ in enumerate(self.datamodule.label_encoder.classes_):
                pred_df[f"{class_}_probability"] = predictions[:, i]
            pred_df[f"prediction"] = self.datamodule.label_encoder.inverse_transform(
                np.argmax(predictions, axis=1)
            )
        return pred_df
