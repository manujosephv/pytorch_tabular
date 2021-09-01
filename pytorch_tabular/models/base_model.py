# Pytorch Tabular
# Author: Manu Joseph <manujoseph@gmail.com>
# For license information, see LICENSE.TXT
"""Base Model"""
import logging
from abc import ABCMeta, abstractmethod
from typing import Callable, Dict, List, Optional

import pytorch_lightning as pl
import torch
import torchmetrics
import torch.nn as nn
from omegaconf import DictConfig

try:
    import wandb
    import plotly.graph_objects as go

    WANDB_INSTALLED = True
except ImportError:
    WANDB_INSTALLED = False


logger = logging.getLogger(__name__)


class BaseModel(pl.LightningModule, metaclass=ABCMeta):
    def __init__(
        self,
        config: DictConfig,
        custom_loss: Optional[torch.nn.Module] = None,
        custom_metrics: Optional[List[Callable]] = None,
        custom_optimizer: Optional[torch.optim.Optimizer] = None,
        custom_optimizer_params: Dict = {},
        **kwargs
    ):
        super().__init__()
        self.custom_loss = custom_loss
        self.custom_metrics = custom_metrics
        self.custom_optimizer = custom_optimizer
        self.custom_optimizer_params = custom_optimizer_params
        #Updating config with custom parameters for experiment tracking
        if self.custom_loss is not None:
            config.loss = str(self.custom_loss)
        if self.custom_metrics is not None:
            config.metrics = [str(m) for m in self.custom_metrics]
            config.metrics_params = [vars(m) for m in self.custom_metrics]
        if self.custom_optimizer is not None:
            config.optimizer = str(self.custom_optimizer.__class__.__name__)
        if self.custom_optimizer_params is not None:
            config.optimizer_params = self.custom_optimizer_params
        self.save_hyperparameters(config)
        # The concatenated output dim of the embedding layer
        self._build_network()
        self._setup_loss()
        self._setup_metrics()

    @abstractmethod
    def _build_network(self):
        pass

    def _setup_loss(self):
        if self.custom_loss is None:
            try:
                self.loss = getattr(nn, self.hparams.loss)()
            except AttributeError as e:
                logger.error(
                    f"{self.hparams.loss} is not a valid loss defined in the torch.nn module"
                )
                raise e
        else:
            self.loss = self.custom_loss

    def _setup_metrics(self):
        if self.custom_metrics is None:
            self.metrics = []
            task_module = torchmetrics.functional
            for metric in self.hparams.metrics:
                try:
                    self.metrics.append(getattr(task_module, metric))
                except AttributeError as e:
                    logger.error(
                        f"{metric} is not a valid functional metric defined in the torchmetrics.functional module"
                    )
                    raise e
        else:
            self.metrics = self.custom_metrics
            self.hparams.metrics = [m.__name__ for m in self.custom_metrics]

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
            #TODO loss fails with batch size of 1
            computed_loss = self.loss(y_hat.squeeze(), y.squeeze())
        self.log(
            f"{tag}_loss",
            computed_loss,
            on_epoch=(tag == "valid"),
            on_step=(tag == "train"),
            # on_step=False,
            logger=True,
            prog_bar=True,
        )
        return computed_loss

    def calculate_metrics(self, y, y_hat, tag):
        metrics = []
        for metric, metric_str, metric_params in zip(
            self.metrics, self.hparams.metrics, self.hparams.metrics_params
        ):
            if (self.hparams.task == "regression"):
                _metrics = []
                for i in range(self.hparams.output_dim):
                    if (
                        metric.__name__
                        == torchmetrics.functional.mean_squared_log_error.__name__
                    ):
                        # MSLE should only be used in strictly positive targets. It is undefined otherwise
                        _metric = metric(
                            torch.clamp(y_hat[:, i], min=0),
                            torch.clamp(y[:, i], min=0),
                            **metric_params,
                        )
                    else:
                        _metric = metric(y_hat[:, i], y[:, i], **metric_params)
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
                y_hat = nn.Softmax(dim=-1)(y_hat.squeeze())
                avg_metric = metric(y_hat, y.squeeze(), **metric_params)
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
    
    def data_aware_initialization(self, datamodule):
        pass

    @abstractmethod
    def forward(self, x: Dict):
        pass

    def predict(self, x: Dict, ret_model_output: bool = False):
        ret_value = self.forward(x)
        if ret_model_output:
            return ret_value.get("logits"), ret_value
        else:
            return ret_value.get("logits")

    def training_step(self, batch, batch_idx):
        y = batch["target"]
        y_hat = self(batch)["logits"]
        loss = self.calculate_loss(y, y_hat, tag="train")
        _ = self.calculate_metrics(y, y_hat, tag="train")
        return loss

    def validation_step(self, batch, batch_idx):
        y = batch["target"]
        y_hat = self(batch)["logits"]
        _ = self.calculate_loss(y, y_hat, tag="valid")
        _ = self.calculate_metrics(y, y_hat, tag="valid")
        return y_hat, y

    def test_step(self, batch, batch_idx):
        y = batch["target"]
        y_hat = self(batch)["logits"]
        _ = self.calculate_loss(y, y_hat, tag="test")
        _ = self.calculate_metrics(y, y_hat, tag="test")
        return y_hat, y

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

    def create_plotly_histogram(self, arr, name, bin_dict=None):
        fig = go.Figure()
        for i in range(arr.shape[-1]):
            fig.add_trace(
                go.Histogram(
                    x=arr[:, i],
                    histnorm="probability",
                    name=f"{name}_{i}",
                    xbins=bin_dict,  # dict(start=0.0, end=1.0, size=0.1),  # bins used for histogram
                )
            )
        # Overlay both histograms
        fig.update_layout(
            barmode="overlay",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )
        # Reduce opacity to see both histograms
        fig.update_traces(opacity=0.5)
        return fig

    def validation_epoch_end(self, outputs) -> None:
        do_log_logits = (
            hasattr(self.hparams, "log_logits")
            and self.hparams.log_logits
            and self.hparams.log_target == "wandb"
            and WANDB_INSTALLED
        )
        if do_log_logits:
            logits = [output[0] for output in outputs]
            logits = torch.cat(logits).detach().cpu()
            fig = self.create_plotly_histogram(logits.unsqueeze(1), "logits")
            wandb.log(
                {
                    "valid_logits": fig,
                    "global_step": self.global_step,
                },
                commit=False,
            )
