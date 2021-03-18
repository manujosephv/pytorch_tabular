# Pytorch Tabular
# Author: Manu Joseph <manujoseph@gmail.com>
# For license information, see LICENSE.TXT
# Inspired by https://github.com/rixwew/pytorch-fm/blob/master/torchfm/model/afi.py
"""AutomaticFeatureInteraction Model"""
import logging
from typing import Dict

import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig

from pytorch_tabular.utils import _initialize_layers, _linear_dropout_bn

from ..base_model import BaseModel

logger = logging.getLogger(__name__)


class AutoIntBackbone(pl.LightningModule):
    def __init__(self, config: DictConfig):
        self.embedding_cat_dim = sum([y for x, y in config.embedding_dims])
        # self.hparams = config
        super().__init__()
        self.save_hyperparameters(config)
        self._build_network()

    def _build_network(self):
        # Category Embedding layers
        self.cat_embedding_layers = nn.ModuleList(
            [
                nn.Embedding(cardinality, self.hparams.embedding_dim)
                for cardinality in self.hparams.categorical_cardinality
            ]
        )
        if self.hparams.batch_norm_continuous_input:
            self.normalizing_batch_norm = nn.BatchNorm1d(self.hparams.continuous_dim)
        # Continuous Embedding Layer
        self.cont_embedding_layer = nn.Embedding(
            self.hparams.continuous_dim, self.hparams.embedding_dim
        )
        if self.hparams.embedding_dropout != 0 and self.embedding_cat_dim != 0:
            self.embed_dropout = nn.Dropout(self.hparams.embedding_dropout)
        # Deep Layers
        _curr_units = self.hparams.embedding_dim
        if self.hparams.deep_layers:
            activation = getattr(nn, self.hparams.activation)
            # Linear Layers
            layers = []
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
        # Projection to Multi-Headed Attention Dims
        self.attn_proj = nn.Linear(_curr_units, self.hparams.attn_embed_dim)
        _initialize_layers(self.hparams, self.attn_proj)
        # Multi-Headed Attention Layers
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
        if self.hparams.has_residuals:
            self.V_res_embedding = torch.nn.Linear(
                _curr_units,
                self.hparams.attn_embed_dim * self.hparams.num_attn_blocks
                if self.hparams.attention_pooling
                else self.hparams.attn_embed_dim,
            )
        self.output_dim = (
            self.hparams.continuous_dim + self.hparams.categorical_dim
        ) * self.hparams.attn_embed_dim
        if self.hparams.attention_pooling:
            self.output_dim = self.output_dim * self.hparams.num_attn_blocks

    def forward(self, x: Dict):
        # (B, N)
        continuous_data, categorical_data = x["continuous"], x["categorical"]
        x = None
        if self.embedding_cat_dim != 0:
            x_cat = [
                embedding_layer(categorical_data[:, i]).unsqueeze(1)
                for i, embedding_layer in enumerate(self.cat_embedding_layers)
            ]
            # (B, N, E)
            x = torch.cat(x_cat, 1)
        if self.hparams.continuous_dim > 0:
            cont_idx = (
                torch.arange(self.hparams.continuous_dim)
                .expand(continuous_data.size(0), -1)
                .to(self.device)
            )
            if self.hparams.batch_norm_continuous_input:
                continuous_data = self.normalizing_batch_norm(continuous_data)
            x_cont = torch.mul(
                continuous_data.unsqueeze(2),
                self.cont_embedding_layer(cont_idx),
            )
            # (B, N, E)
            x = x_cont if x is None else torch.cat([x, x_cont], 1)
        if self.hparams.embedding_dropout != 0 and self.embedding_cat_dim != 0:
            x = self.embed_dropout(x)
        if self.hparams.deep_layers:
            x = self.linear_layers(x)
        # (N, B, E*) --> E* is the Attn Dimention
        cross_term = self.attn_proj(x).transpose(0, 1)
        if self.hparams.attention_pooling:
            attention_ops = []
        for self_attn in self.self_attns:
            cross_term, _ = self_attn(cross_term, cross_term, cross_term)
            if self.hparams.attention_pooling:
                attention_ops.append(cross_term)
        if self.hparams.attention_pooling:
            cross_term = torch.cat(attention_ops, dim=-1)
        # (B, N, E*)
        cross_term = cross_term.transpose(0, 1)
        if self.hparams.has_residuals:
            # (B, N, E*) --> Projecting Embedded input to Attention sub-space
            V_res = self.V_res_embedding(x)
            cross_term = cross_term + V_res
        # (B, NxE*)
        cross_term = nn.ReLU()(cross_term).reshape(-1, self.output_dim)
        return cross_term


class AutoIntModel(BaseModel):
    def __init__(self, config: DictConfig, **kwargs):
        # The concatenated output dim of the embedding layer
        self.embedding_cat_dim = sum([y for x, y in config.embedding_dims])
        super().__init__(config, **kwargs)

    def _build_network(self):
        # Backbone
        self.backbone = AutoIntBackbone(self.hparams)
        self.dropout = nn.Dropout(self.hparams.dropout)
        # Adding the last layer
        self.output_layer = nn.Linear(
            self.backbone.output_dim, self.hparams.output_dim
        )  # output_dim auto-calculated from other config
        _initialize_layers(self.hparams, self.output_layer)

    def forward(self, x: Dict):
        x = self.backbone(x)
        x = self.dropout(x)
        y_hat = self.output_layer(x)
        if (self.hparams.task == "regression") and (
            self.hparams.target_range is not None
        ):
            for i in range(self.hparams.output_dim):
                y_min, y_max = self.hparams.target_range[i]
                y_hat[:, i] = y_min + nn.Sigmoid()(y_hat[:, i]) * (y_max - y_min)
        return {"logits": y_hat, "backbone_features": x}
