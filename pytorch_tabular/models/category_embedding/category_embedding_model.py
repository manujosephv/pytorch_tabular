# Pytorch Tabular
# Author: Manu Joseph <manujoseph@gmail.com>
# For license information, see LICENSE.TXT
"""Category Embedding Model"""
import logging
from typing import Dict

import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig
from pytorch_tabular.utils import _initialize_layers, _linear_dropout_bn

from ..base_model import BaseModel

logger = logging.getLogger(__name__)


class FeedForwardBackbone(pl.LightningModule):
    def __init__(self, config: DictConfig, **kwargs):
        self.embedding_cat_dim = sum([y for x, y in config.embedding_dims])
        super().__init__()
        self.save_hyperparameters(config)
        self._build_network()

    def _build_network(self):
        activation = getattr(nn, self.hparams.activation)
        # Linear Layers
        layers = []
        _curr_units = self.embedding_cat_dim + self.hparams.continuous_dim
        if self.hparams.embedding_dropout != 0 and self.embedding_cat_dim != 0:
            layers.append(nn.Dropout(self.hparams.embedding_dropout))
        for units in self.hparams.layers.split("-"):
            layers.extend(
                _linear_dropout_bn(
                    self.hparams,
                    _curr_units,
                    int(units),
                    activation,
                    self.hparams.dropout,
                )
            )
            _curr_units = int(units)
        self.linear_layers = nn.Sequential(*layers)
        self.output_dim = _curr_units

    def forward(self, x):
        x = self.linear_layers(x)
        return x


class CategoryEmbeddingModel(BaseModel):
    def __init__(self, config: DictConfig, **kwargs):
        # The concatenated output dim of the embedding layer
        self.embedding_cat_dim = sum([y for x, y in config.embedding_dims])
        super().__init__(config, **kwargs)

    def _build_network(self):
        # Embedding layers
        self.embedding_layers = nn.ModuleList(
            [nn.Embedding(x, y) for x, y in self.hparams.embedding_dims]
        )
        # Continuous Layers
        if self.hparams.batch_norm_continuous_input:
            self.normalizing_batch_norm = nn.BatchNorm1d(self.hparams.continuous_dim)
        # Backbone
        self.backbone = FeedForwardBackbone(self.hparams)
        # Adding the last layer
        self.output_layer = nn.Linear(
            self.backbone.output_dim, self.hparams.output_dim
        )  # output_dim auto-calculated from other config
        _initialize_layers(self.hparams, self.output_layer)
    
    def unpack_input(self, x: Dict):
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

        if self.hparams.continuous_dim != 0:
            if self.hparams.batch_norm_continuous_input:
                continuous_data = self.normalizing_batch_norm(continuous_data)

            if self.embedding_cat_dim != 0:
                x = torch.cat([x, continuous_data], 1)
            else:
                x = continuous_data
        return x

    def forward(self, x: Dict):
        x = self.unpack_input(x)
        x = self.backbone(x)
        y_hat = self.output_layer(x)
        if (self.hparams.task == "regression") and (
            self.hparams.target_range is not None
        ):
            for i in range(self.hparams.output_dim):
                y_min, y_max = self.hparams.target_range[i]
                y_hat[:, i] = y_min + nn.Sigmoid()(y_hat[:, i]) * (y_max - y_min)
        return {"logits": y_hat, "backbone_features": x}
