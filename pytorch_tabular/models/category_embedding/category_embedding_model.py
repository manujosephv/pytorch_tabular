# Pytorch Tabular
# Author: Manu Joseph <manujoseph@gmail.com>
# For license information, see LICENSE.TXT
"""Category Embedding Model"""
import logging
from typing import Dict

import torch
import torch.nn as nn
from omegaconf import DictConfig

from ..base_model import BaseModel

logger = logging.getLogger(__name__)


class CategoryEmbeddingModel(BaseModel):
    def __init__(self, config: DictConfig, **kwargs):
        # The concatenated output dim of the embedding layer
        self.embedding_cat_dim = sum([y for x, y in config.embedding_dims])
        self.continuous_dim = (
            config.continuous_dim
        )  # auto-calculated by the calling script
        super().__init__(config, **kwargs)

    def _initialize_layers(self, layer):
        if self.hparams.activation == "ReLU":
            nonlinearity = "relu"
        elif self.hparams.activation == "LeakyReLU":
            nonlinearity = "leaky_relu"
        else:
            if self.hparams.initialization == "kaiming":
                logger.warning(
                    "Kaiming initialization is only recommended for ReLU and LeakyReLU."
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
        layers = []
        if self.hparams.use_batch_norm:
            layers.append(nn.BatchNorm1d(num_features=in_units))
        linear = nn.Linear(in_units, out_units)
        self._initialize_layers(linear)
        layers.extend([linear, activation()])
        if dropout != 0:
            layers.append(nn.Dropout(dropout))
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

    def forward(self, x: Dict):
        continuous_data, categorical_data = x["continuous"], x["categorical"]
        if self.embedding_cat_dim != 0:
            x = []
            # for i, embedding_layer in enumerate(self.embedding_layers):
            #     x.append(embedding_layer(categorical_data[:, i]))
            x = [
                embedding_layer(categorical_data[:, i])
                for i, embedding_layer in enumerate(self.embedding_layers)
            ]
            x = torch.cat(x, 1)

        if self.continuous_dim != 0:
            if self.hparams.batch_norm_continuous_input:
                continuous_data = self.normalizing_batch_norm(continuous_data)

            if self.embedding_cat_dim != 0:
                x = torch.cat([x, continuous_data], 1)
            else:
                x = continuous_data

        x = self.linear_layers(x)
        if (
            (self.hparams.task == "regression")
            and (self.hparams.target_range is not None)
        ):
            for i in range(self.hparams.output_dim):
                y_min, y_max = self.hparams.target_range[i]
                x[:, i] = y_min + nn.Sigmoid()(x[:, i]) * (y_max - y_min)
        return x

