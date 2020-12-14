import logging
from typing import Dict

import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig
from abc import ABCMeta, abstractmethod

logger = logging.getLogger(__name__)


class BaseModel(pl.LightningModule, metaclass=ABCMeta):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.save_hyperparameters(config)
        # The concatenated output dim of the embedding layer
        self._build_network()
        self._setup_loss()
        self._setup_metrics()

    @abstractmethod
    def _build_network(self):
        pass

    def _setup_loss(self):
        try:
            self.loss = getattr(nn, self.hparams.loss)()
        except AttributeError:
            logger.error(
                f"{self.hparams.loss} is not a valid loss defined in the torch.nn module"
            )

    def _setup_metrics(self):
        self.metrics = []
        task_module = getattr(pl.metrics, self.hparams.task)
        for metric in self.hparams.metrics:
            self.metrics.append(
                getattr(task_module, metric)().to(
                    "cpu" if self.hparams.gpus == 0 else "cuda"
                )
            )

    def calculate_loss(self, y, y_hat, tag):
        if (self.hparams.task == "regression") and (self.hparams.output_dim > 1):
            losses = []
            for i in range(self.hparams.output_dim):
                _loss = self.loss(y_hat[:, i], y[:, i])
                losses.append(_loss)
                self.log(
                    f"{tag}_loss_{i}",
                    _loss,
                    on_epoch=True,
                    on_step=False,
                    logger=True,
                    prog_bar=False,
                )
            computed_loss = torch.stack(losses, dim=0).sum()
        else:
            computed_loss = self.loss(y_hat.squeeze(), y.squeeze())
        self.log(
            f"{tag}_loss",
            computed_loss,
            on_epoch=(tag == "valid"),
            on_step=(tag == "train"),
            logger=True,
            prog_bar=True,
        )
        return computed_loss

    def calculate_metrics(self, y, y_hat, tag):
        metrics = []
        for metric, metric_str in zip(self.metrics, self.hparams.metrics):
            if (self.hparams.task == "regression") and (self.hparams.output_dim > 1):
                _metrics = []
                for i in range(self.hparams.output_dim):
                    _metric = metric(y_hat[:, i], y[:, i])
                    self.log(
                        f"{tag}_{metric_str}_{i}",
                        _metric,
                        on_epoch=True,
                        on_step=False,
                        logger=True,
                        prog_bar=False,
                    )
                    _metrics.append(_metric)
                avg_metric = torch.stack(_metrics, dim=0).sum()
            else:
                avg_metric = metric(y_hat.squeeze(), y.squeeze())
            metrics.append(avg_metric)
            self.log(
                f"{tag}_{metric_str}",
                avg_metric,
                on_epoch=True,
                on_step=False,
                logger=True,
                prog_bar=True,
            )
        return metrics

    @abstractmethod
    def forward(self, x: Dict):
        pass

    def training_step(self, batch, batch_idx):
        y = batch["target"]
        y_hat = self(batch)
        loss = self.calculate_loss(y, y_hat, tag="train")
        _ = self.calculate_metrics(y, y_hat, tag="train")
        return loss

    def validation_step(self, batch, batch_idx):
        y = batch["target"]
        y_hat = self(batch)
        _ = self.calculate_loss(y, y_hat, tag="valid")
        _ = self.calculate_metrics(y, y_hat, tag="valid")
        return y_hat, y

    def test_step(self, batch, batch_idx):
        y = batch["target"]
        y_hat = self(batch)
        _ = self.calculate_loss(y, y_hat, tag="test")
        _ = self.calculate_metrics(y, y_hat, tag="test")
        return y_hat, y

    def configure_optimizers(self):
        self._optimizer = getattr(torch.optim, self.hparams.optimizer)
        opt = self._optimizer(
            self.parameters(),
            lr=self.hparams.learning_rate,
            **self.hparams.optimizer_params,
        )
        if self.hparams.lr_scheduler is not None:
            self._lr_scheduler = getattr(
                torch.optim.lr_scheduler, self.hparams.lr_scheduler
            )
            if isinstance(self._lr_scheduler, torch.optim.lr_scheduler._LRScheduler):
                return {
                    "optimizer": opt,
                    "lr_scheduler": self._lr_scheduler(
                        opt, **self.hparams.lr_scheduler_params
                    ),
                }
            else:
                return {
                    "optimizer": opt,
                    "lr_scheduler": self._lr_scheduler(
                        opt, **self.hparams.lr_scheduler_params
                    ),
                    "monitor": self.hparams.lr_scheduler_monitor_metric,
                }
        else:
            return opt
