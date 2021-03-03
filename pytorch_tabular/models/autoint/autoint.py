# Pytorch Tabular
# Author: Manu Joseph <manujoseph@gmail.com>
# For license information, see LICENSE.TXT
# Inspired by https://github.com/rixwew/pytorch-fm/blob/master/torchfm/model/afi.py
"""AutomaticFeatureInteraction Model"""
import logging
from typing import Dict

import torch
import torch.nn as nn
from omegaconf import DictConfig
from pytorch_tabular.utils import _initialize_layers

from ..base_model import BaseModel

logger = logging.getLogger(__name__)


class AutoIntBackbone(BaseModel):
    def __init__(self, config: DictConfig, **kwargs):
        self.embedding_cat_dim = sum([y for x, y in config.embedding_dims])
        super().__init__(config, **kwargs)

    def _linear_dropout_bn(self, in_units, out_units, activation, dropout):
        layers = []
        if self.hparams.use_batch_norm:
            layers.append(nn.BatchNorm1d(num_features=in_units))
        linear = nn.Linear(in_units, out_units)
        _initialize_layers(self.hparams, linear)
        layers.extend([linear, activation()])
        if dropout != 0:
            layers.append(nn.Dropout(dropout))
        return layers

    def _build_network(self):
        # Embedding layers
        self.cat_embedding_layers = nn.ModuleList(
            [nn.Embedding(x, y) for x, y in self.hparams.cat_embedding_dims]
        )
        self.cont_embedding_layers = nn.ModuleList(
            [
                nn.Embedding(1, self.hparams.cont_embedding_dim)
                for i in range(self.hparams.continuous_dim)
            ]
        )
        if self.hparams.embedding_dropout != 0 and self.embedding_cat_dim != 0:
            self.embed_dropout = nn.Dropout(self.hparams.embedding_dropout)
        # if self.hparams.use_batch_norm:
        #     self.normalizing_batch_norm = nn.BatchNorm1d(self.hparams.continuous_dim+self.hparams.embedding_cat_dim)
        if self.hparams.deep_layers:
            activation = getattr(nn, self.hparams.activation)
            # Linear Layers
            layers = []
            _curr_units = self.hparams.continuous_dim + self.embedding_cat_dim
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
            self.linear_layers = nn.Sequential(*layers)
        else:
            _curr_units = self.hparams.continuous_dim + self.embedding_cat_dim

        self.self_attns = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    self.hparams.attn_embed_dim,
                    self.hparams.num_heads,
                    dropout=self.hparams.attn_dropouts,
                )
                for _ in range(self.hparams.num_attn_blocks)
            ]
        )
        self.atten_output_dim = (
            len(self.hparams.continuous_cols + self.hparams.categorical_cols)
            * self.hparams.atten_embed_dim
        )


    def forward(self, x):
        x = self.linear_layers(x)
        return x


class AutoIntModel(BaseModel):
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
        self.backbone = AutoIntBackbone(self.hparams)
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
