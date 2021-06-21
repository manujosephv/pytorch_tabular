# Pytorch Tabular
# Author: Manu Joseph <manujoseph@gmail.com>
# For license information, see LICENSE.TXT
# Inspired by https://github.com/lucidrains/tab-transformer-pytorch/blob/main/tab_transformer_pytorch/tab_transformer_pytorch.py
"""TabTransformer Model"""
import logging
from typing import Dict, OrderedDict

import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig
from einops import rearrange

from pytorch_tabular.utils import _initialize_layers, _linear_dropout_bn
from .components import TransformerEncoderBlock

from ..base_model import BaseModel

logger = logging.getLogger(__name__)

class TabTransformerBackbone(pl.LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.save_hyperparameters(config)
        self._build_network()
        #TODO Add output_dim

    def _build_network(self):
        if len(self.hparams.categorical_cols) > 0:
            # Category Embedding layers
            # self.embedding_dropout = nn.Dropout(self.hparams.embedding_dropout)
            self.cat_embedding_layers = nn.ModuleList(
                [
                    nn.Embedding(cardinality, self.hparams.input_embed_dim)
                    for cardinality in self.hparams.categorical_cardinality
                ]
            )
            if self.hparams.embedding_dropout != 0:
                self.embed_dropout = nn.Dropout(self.hparams.embedding_dropout)
        self.transformer_blocks = OrderedDict()
        for i in range(self.hparams.num_attn_blocks):
            self.transformer_blocks[f"mha_block_{i}"] = TransformerEncoderBlock(
                input_embed_dim=self.hparams.input_embed_dim,
                num_heads=self.hparams.num_heads,
                ff_hidden_multiplier=self.hparams.ff_hidden_multiplier,
                ff_activation = self.hparams.transformer_activation,
                attn_dropout=self.hparams.attn_dropout,
                ff_dropout=self.hparams.ff_dropout,
                add_norm_dropout=self.hparams.add_norm_dropout,
            )
        self.transformer_blocks = nn.Sequential(self.transformer_blocks)

        if self.hparams.batch_norm_continuous_input:
            self.normalizing_batch_norm = nn.BatchNorm1d(self.hparams.continuous_dim)
        # Final MLP Layers
        _curr_units = self.hparams.input_embed_dim*len(self.hparams.categorical_cols) + self.hparams.continuous_dim
        # Linear Layers
        layers = []
        for units in self.hparams.out_ff_layers.split("-"):
            layers.extend(
                _linear_dropout_bn(
                    self.hparams.out_ff_activation,
                    self.hparams.out_ff_initialization,
                    self.hparams.use_batch_norm,
                    _curr_units,
                    int(units),
                    self.hparams.out_ff_dropout,
                )
            )
            _curr_units = int(units)
        self.linear_layers = nn.Sequential(*layers)
        self.output_dim = _curr_units

    def forward(self, x: Dict):
        # (B, N)
        continuous_data, categorical_data = x["continuous"], x["categorical"]
        x = None
        if len(self.hparams.categorical_cols) > 0:
            x_cat = [
                embedding_layer(categorical_data[:, i]).unsqueeze(1)
                for i, embedding_layer in enumerate(self.cat_embedding_layers)
            ]
            # (B, N, E)
            x = torch.cat(x_cat, 1)
            if self.hparams.embedding_dropout != 0:
                x = self.embed_dropout(x)
            for i, block in enumerate(self.transformer_blocks):
                x = block(x)
            #Flatten (Batch, N_Categorical, Hidden) --> (Batch, N_CategoricalxHidden)
            x = rearrange(x, "b n h -> b (n h)")
        if self.hparams.continuous_dim > 0:
            if self.hparams.batch_norm_continuous_input:
                x_cont = self.normalizing_batch_norm(continuous_data)
            else:
                x_cont = continuous_data
            # (B, N, E)
            x = x_cont if x is None else torch.cat([x, x_cont], 1)
        x = self.linear_layers(x)
        return x

class TabTransformerModel(BaseModel):
    def __init__(self, config: DictConfig, **kwargs):
        super().__init__(config, **kwargs)

    def _build_network(self):
        # Backbone
        self.backbone = TabTransformerBackbone(self.hparams)
        self.dropout = nn.Dropout(self.hparams.out_ff_dropout)
        # Adding the last layer
        self.output_layer = nn.Linear(
            self.backbone.output_dim, self.hparams.output_dim
        )  # output_dim auto-calculated from other config
        _initialize_layers(self.hparams.out_ff_activation, self.hparams.out_ff_initialization, self.output_layer)

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
