import logging
from typing import Dict, List

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from sklearn.metrics import mean_absolute_error, r2_score

logger = logging.getLogger(__name__)


class TabularModel(pl.LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        # if isinstance(config, dict):
        #     config = Namespace(**config)
        # self.hparams = config
        self.save_hyperparameters(config)
        # The concatenated output dim of the embedding layer
        self.embedding_cat_dim = sum([y for x, y in self.hparams.embedding_dims])
        self.continuous_dim = (
            self.hparams.continuous_dim
        )  # auto-calculated by the calling script

        self._build_network()
        self._setup_loss()
        self._setup_metrics()

    def _initialize_layers(self, layer):
        if self.hparams.activation == "ReLU":
            nonlinearity = "relu"
        elif self.hparams.activation == "LeakyReLU":
            nonlinearity = "leaky_relu"
        else:
            if self.hparams.initialization == "kaiming":
                logger.warning(
                    "Kaiming initialization is only recommended for ReLU and LeakyReLU. Resetting to LeakyRelu for nonlinearity in initialization"
                )
                nonlinearity = "leaky_relu"
            else:
                nonlinearity = "relu"

        if self.hparams.initialization == "kaiming":
            nn.init.kaiming_normal_(layer.weight, nonlinearity=nonlinearity)
        elif self.hparams.initialization == "xavier":
            nn.init.xavier_normal_(
                layer.weight,
                gain=nn.init.calculate_gain(nonlinearity)
                if self.hparams.activation in ["ReLU", "LeakyReLU"]
                else 1,
            )
        elif self.hparams.initialization == "random":
            nn.init.normal_(layer)

    def _linear_dropout_bn(self, in_units, out_units, activation, dropout):
        layers = [nn.Linear(in_units, out_units), activation()]
        self._initialize_layers(layers[0])
        if dropout != 0:
            layers.append(nn.Dropout(dropout))
        if self.hparams.use_batch_norm:
            layers.append(nn.BatchNorm1d(num_features=out_units))
        return layers

    def _build_network(self):
        activation = getattr(nn, self.hparams.activation)
        # Embedding layers

        self.embedding_layers = nn.ModuleList(
            [nn.Embedding(x, y) for x, y in self.hparams.embedding_dims]
        )
        # Continuous Layers
        if self.hparams.batch_norm_continuous_input:
            self.normalizing_batch_norm = nn.BatchNorm1d(self.continuous_dim)
        # Linear Layers
        layers = []
        _curr_units = self.embedding_cat_dim + self.continuous_dim
        if self.hparams.embedding_dropout != 0 and self.embedding_cat_dim != 0:
            layers.append(nn.Dropout(self.hparams.embedding_dropout))
        for units in self.hparams.layers.split("-"):
            layers.extend(
                self._linear_dropout_bn(
                    _curr_units,
                    int(units),
                    activation,
                    self.hparams.dropout,
                )
            )
            _curr_units = int(units)
        # Adding the last layer
        layers.append(
            nn.Linear(_curr_units, self.hparams.output_dim)
        )  # output_dim auto-calculated from other config
        self.linear_layers = nn.Sequential(*layers)

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
            self.metrics.append(getattr(task_module, metric)().to("cpu" if self.hparams.gpus==0 else 'cuda'))

    def calculate_loss(self, y, y_hat, tag):
        if self.hparams.output_dim > 1:
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
            computed_loss = self.loss(y_hat, y)
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
            if self.hparams.output_dim > 1:
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
                avg_metric = metric(y_hat, y)
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

    def forward(self, x):
        continuous_data, categorical_data = x["continuous"], x["categorical"]
        if self.embedding_cat_dim != 0:
            x = []
            for i, embedding_layer in enumerate(self.embedding_layers):
                x.append(embedding_layer(categorical_data[:, i]))
            # x = [
            #     embedding_layer(categorical_data[:, i])
            #     for i, embedding_layer in enumerate(self.embedding_layers)
            # ]
            x = torch.cat(x, 1)

        if self.continuous_dim != 0:
            if self.hparams.batch_norm_continuous_input:
                continuous_data = self.normalizing_batch_norm(continuous_data)

            if self.embedding_cat_dim != 0:
                x = torch.cat([x, continuous_data], 1)
            else:
                x = continuous_data

        x = self.linear_layers(x)
        return x

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
