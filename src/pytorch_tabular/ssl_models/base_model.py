# Pytorch Tabular
# Author: Manu Joseph <manujoseph@gmail.com>
# For license information, see LICENSE.TXT
"""SSL Base Model"""
import warnings
from abc import ABCMeta, abstractmethod
from typing import Dict, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from pytorch_tabular.utils import get_logger, getattr_nested, reset_all_weights

logger = get_logger(__name__)


def safe_merge_config(config: DictConfig, inferred_config: DictConfig) -> DictConfig:
    """Merge two configurations.

    Args:
        base_config: The base configuration.
        custom_config: The custom configuration.

    Returns:
        The merged configuration.
    """
    # using base config values if exist
    if "embedding_dims" in config.keys() and config.embedding_dims is not None:
        inferred_config.embedding_dims = config.embedding_dims
    merged_config = OmegaConf.merge(OmegaConf.to_container(config), OmegaConf.to_container(inferred_config))
    return merged_config


class SSLBaseModel(pl.LightningModule, metaclass=ABCMeta):
    def __init__(
        self,
        config: DictConfig,
        mode: str = "pretrain",
        encoder: Optional[nn.Module] = None,
        decoder: Optional[nn.Module] = None,
        custom_optimizer: Optional[torch.optim.Optimizer] = None,
        custom_optimizer_params: Dict = {},
        **kwargs,
    ):
        super().__init__()
        assert "inferred_config" in kwargs, "inferred_config not found in initialization arguments"
        inferred_config = kwargs["inferred_config"]
        # Merging the config and inferred config
        config = safe_merge_config(config, inferred_config)

        self._setup_encoder_decoder(
            encoder,
            config.encoder_config,
            decoder,
            config.decoder_config,
            inferred_config,
        )
        self.custom_optimizer = custom_optimizer
        self.custom_optimizer_params = custom_optimizer_params
        # Updating config with custom parameters for experiment tracking
        if self.custom_optimizer is not None:
            config.optimizer = str(self.custom_optimizer.__class__.__name__)
        if len(self.custom_optimizer_params) > 0:
            config.optimizer_params = self.custom_optimizer_params
        self.mode = mode
        self._check_and_verify()
        self.save_hyperparameters(config)
        self._build_network()
        self._setup_loss()
        self._setup_metrics()

    def _setup_encoder_decoder(self, encoder, encoder_config, decoder, decoder_config, inferred_config):
        assert (encoder is not None) or (
            encoder_config is not None
        ), "Either encoder or encoder_config must be provided"
        # assert (decoder is not None) or (decoder_config is not None), "Either decoder or decoder_config must be provided"
        if encoder is not None:
            self.encoder = encoder
            self._custom_decoder = True
        else:
            # Since encoder is not provided, we will use the encoder_config
            model_callable = getattr_nested(encoder_config._module_src, encoder_config._backbone_name)
            self.encoder = model_callable(
                safe_merge_config(encoder_config, inferred_config),
                # inferred_config=inferred_config,
            )
        if decoder is not None:
            self.decoder = decoder
            self._custom_encoder = True
        elif decoder_config is not None:
            # Since decoder is not provided, we will use the decoder_config
            model_callable = getattr_nested(decoder_config._module_src, decoder_config._backbone_name)
            self.decoder = model_callable(
                safe_merge_config(decoder_config, inferred_config),
                # inferred_config=inferred_config,
            )
        else:
            self.decoder = nn.Identity()

    def _check_and_verify(self):
        assert hasattr(self.encoder, "output_dim"), "An encoder backbone must have an output_dim attribute"
        if isinstance(self.decoder, nn.Identity):
            self.decoder.output_dim = self.encoder.output_dim
        assert hasattr(self.decoder, "output_dim"), "A decoder must have an output_dim attribute"

    @property
    def embedding_layer(self):
        raise NotImplementedError("`embedding_layer` property needs to be implemented by inheriting classes")

    @property
    def featurizer(self):
        raise NotImplementedError("`featurizer` property needs to be implemented by inheriting classes")

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

    def predict(self, x: Dict, ret_model_output: bool = True):  # ret_model_output only for compatibility
        assert ret_model_output, "ret_model_output must be True in case of SSL predict"
        return self.featurize(x)

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
                logger.error(f"{self.hparams.optimizer} is not a valid optimizer defined in the torch.optim module")
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
                self._lr_scheduler = getattr(torch.optim.lr_scheduler, self.hparams.lr_scheduler)
            except AttributeError as e:
                logger.error(
                    f"{self.hparams.lr_scheduler} is not a valid learning rate sheduler defined in the torch.optim.lr_scheduler module"
                )
                raise e
            if isinstance(self._lr_scheduler, torch.optim.lr_scheduler._LRScheduler):
                return {
                    "optimizer": opt,
                    "lr_scheduler": self._lr_scheduler(opt, **self.hparams.lr_scheduler_params),
                }
            else:
                return {
                    "optimizer": opt,
                    "lr_scheduler": self._lr_scheduler(opt, **self.hparams.lr_scheduler_params),
                    "monitor": self.hparams.lr_scheduler_monitor_metric,
                }
        else:
            return opt

    def reset_weights(self):
        reset_all_weights(self.featurizer)
        reset_all_weights(self.embedding_layer)
