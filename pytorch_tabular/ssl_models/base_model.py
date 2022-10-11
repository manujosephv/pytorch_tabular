# Pytorch Tabular
# Author: Manu Joseph <manujoseph@gmail.com>
# For license information, see LICENSE.TXT
"""SSL Base Model"""
from abc import ABCMeta, abstractmethod
import logging
from typing import Callable, Dict, List, Optional, Union
import warnings

import torch
import torch.nn as nn
from omegaconf import DictConfig

# from pytorch_tabular.models.base_model import BaseModel

# import pytorch_tabular.models.ssl.augmentations as augmentations
# import pytorch_tabular.models.ssl.ssl_utils as ssl_utils
# import pytorch_tabular.ssl_models.common.ssl_losses as ssl_losses
import pytorch_lightning as pl

# from pytorch_tabular.utils import loss_contrastive

logger = logging.getLogger(__name__)


class SSLBaseModel(pl.LightningModule, metaclass=ABCMeta):
    def __init__(
        self,
        encoder: Union[pl.LightningDataModule, nn.Module],
        decoder: Union[pl.LightningDataModule, nn.Module],
        config: DictConfig,
        custom_optimizer: Optional[torch.optim.Optimizer] = None,
        custom_optimizer_params: Dict = {},
        **kwargs,
    ):
        assert hasattr(
            encoder, "output_dim"
        ), "An encoder backbone must have an output_dim attribute"
        if isinstance(decoder, nn.Identity):
            decoder.output_dim = encoder.output_dim
        assert hasattr(
            decoder, "output_dim"
        ), "A decoder must have an output_dim attribute"
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.custom_optimizer = custom_optimizer
        self.custom_optimizer_params = custom_optimizer_params
        # Updating config with custom parameters for experiment tracking
        if self.custom_optimizer is not None:
            config.optimizer = str(self.custom_optimizer.__class__.__name__)
        if len(self.custom_optimizer_params) > 0:
            config.optimizer_params = self.custom_optimizer_params
        self.save_hyperparameters(config)
        self._build_network()
        self._setup_loss()
        self._setup_metrics()

    @abstractmethod
    def _setup_loss(self):
        pass

    @abstractmethod
    def _setup_metrics(self):
        pass

    @abstractmethod
    def calculate_loss(self, output, tag):
        pass

    @abstractmethod
    def calculate_metrics(self, output, tag):
        pass

    @abstractmethod
    def forward(self, x: Dict):
        pass
    
    @abstractmethod
    def featurize(self, x: Dict):
        pass

    def data_aware_initialization(self, datamodule):
        pass

    def training_step(self, batch, batch_idx):
        output = self.forward(batch)
        loss = self.calculate_loss(output, tag="train")
        _ = self.calculate_metrics(output, tag="train")
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            output = self.forward(batch)
            _ = self.calculate_loss(output, tag="valid")
            _ = self.calculate_metrics(output, tag="valid")
        return output

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            output = self.forward(batch)
            _ = self.calculate_loss(output, tag="test")
            _ = self.calculate_metrics(output, tag="test")
        return output

    def validation_epoch_end(self, outputs) -> None:
        if hasattr(self.hparams, "log_logits") and self.hparams.log_logits:
            warnings.warn("Logging Logits is disabled for SSL tasks")

    def configure_optimizers(self):
        if self.custom_optimizer is None:
            # Loading from the config
            try:
                self._optimizer = getattr(torch.optim, self.hparams.optimizer)
                opt = self._optimizer(
                    self.parameters(),
                    lr=self.hparams.learning_rate,
                    **self.hparams.optimizer_params,
                )
            except AttributeError as e:
                logger.error(
                    f"{self.hparams.optimizer} is not a valid optimizer defined in the torch.optim module"
                )
                raise e
        else:
            # Loading from custom fit arguments
            self._optimizer = self.custom_optimizer

            opt = self._optimizer(
                self.parameters(),
                lr=self.hparams.learning_rate,
                **self.custom_optimizer_params,
            )
        if self.hparams.lr_scheduler is not None:
            try:
                self._lr_scheduler = getattr(
                    torch.optim.lr_scheduler, self.hparams.lr_scheduler
                )
            except AttributeError as e:
                logger.error(
                    f"{self.hparams.lr_scheduler} is not a valid learning rate sheduler defined in the torch.optim.lr_scheduler module"
                )
                raise e
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
