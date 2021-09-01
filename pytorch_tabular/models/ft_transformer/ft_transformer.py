# Pytorch Tabular
# Author: Manu Joseph <manujoseph@gmail.com>
# For license information, see LICENSE.TXT
"""Feature Tokenizer Transformer Model"""
import logging
import math
from collections import OrderedDict
from typing import Dict

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import DictConfig

from pytorch_tabular.utils import _initialize_layers, _linear_dropout_bn

from ..base_model import BaseModel
from ..common import SharedEmbeddings, TransformerEncoderBlock

logger = logging.getLogger(__name__)


def _initialize_kaiming(x, initialization, d_sqrt_inv):
    if initialization == "kaiming_uniform":
        nn.init.uniform_(x, a=-d_sqrt_inv, b=d_sqrt_inv)
    elif initialization == "kaiming_normal":
        nn.init.normal_(x, std=d_sqrt_inv)
    elif initialization is None:
        pass
    else:
        raise NotImplementedError(f"initialization should be either of `kaiming_normal`, `kaiming_uniform`, `None`")


class AppendCLSToken(nn.Module):
    """Appends the [CLS] token for BERT-like inference."""

    def __init__(self, d_token: int, initialization: str) -> None:
        """Initialize self."""
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(d_token))
        d_sqrt_inv = 1 / math.sqrt(d_token)
        _initialize_kaiming(self.weight, initialization, d_sqrt_inv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform the forward pass."""
        assert x.ndim == 3
        return torch.cat([x, self.weight.view(1, 1, -1).repeat(len(x), 1, 1)], dim=1)


class FTTransformerBackbone(pl.LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        assert config.share_embedding_strategy in [
            "add",
            "fraction",
        ], f"`share_embedding_strategy` should be one of `add` or `fraction`, not {self.hparams.share_embedding_strategy}"
        self.save_hyperparameters(config)
        self._build_network()

    def _build_network(self):
        d_sqrt_inv = 1 / math.sqrt(self.hparams.input_embed_dim)
        if len(self.hparams.categorical_cols) > 0:
            # Category Embedding layers
            if self.hparams.share_embedding:
                self.cat_embedding_layers = nn.ModuleList(
                    [
                        SharedEmbeddings(
                            cardinality,
                            self.hparams.input_embed_dim,
                            add_shared_embed=self.hparams.share_embedding_strategy
                            == "add",
                            frac_shared_embed=self.hparams.shared_embedding_fraction,
                        )
                        for cardinality in self.hparams.categorical_cardinality
                    ]
                )

            else:
                self.cat_embedding_layers = nn.ModuleList(
                    [
                        nn.Embedding(cardinality, self.hparams.input_embed_dim)
                        for cardinality in self.hparams.categorical_cardinality
                    ]
                )
            if self.hparams.embedding_bias:
                self.cat_embedding_bias = nn.Parameter(
                        torch.Tensor(
                            self.hparams.categorical_dim, self.hparams.input_embed_dim
                        )
                    )
                _initialize_kaiming(self.cat_embedding_bias, self.hparams.embedding_initialization, d_sqrt_inv)
            # Continuous Embedding Layer
            self.cont_embedding_layer = nn.Embedding(
                self.hparams.continuous_dim, self.hparams.input_embed_dim
            )
            _initialize_kaiming(self.cont_embedding_layer.weight, self.hparams.embedding_initialization, d_sqrt_inv)
            if self.hparams.embedding_bias:
                self.cont_embedding_bias = nn.Parameter(
                        torch.Tensor(
                            self.hparams.continuous_dim, self.hparams.input_embed_dim
                        )
                    )
                _initialize_kaiming(self.cont_embedding_bias, self.hparams.embedding_initialization, d_sqrt_inv)
            if self.hparams.embedding_dropout != 0:
                self.embed_dropout = nn.Dropout(self.hparams.embedding_dropout)
            self.add_cls = AppendCLSToken(
                d_token=self.hparams.input_embed_dim,
                initialization=self.hparams.embedding_initialization,
            )
        self.transformer_blocks = OrderedDict()
        for i in range(self.hparams.num_attn_blocks):
            self.transformer_blocks[f"mha_block_{i}"] = TransformerEncoderBlock(
                input_embed_dim=self.hparams.input_embed_dim,
                num_heads=self.hparams.num_heads,
                ff_hidden_multiplier=self.hparams.ff_hidden_multiplier,
                ff_activation=self.hparams.transformer_activation,
                attn_dropout=self.hparams.attn_dropout,
                ff_dropout=self.hparams.ff_dropout,
                add_norm_dropout=self.hparams.add_norm_dropout,
                keep_attn=self.hparams.attn_feature_importance #Can use Attn Weights to derive feature importance
            )
        self.transformer_blocks = nn.Sequential(self.transformer_blocks)
        if self.hparams.attn_feature_importance:
            self.attention_weights_ = [None] * self.hparams.num_attn_blocks
        if self.hparams.batch_norm_continuous_input:
            self.normalizing_batch_norm = nn.BatchNorm1d(self.hparams.continuous_dim)
        # Final MLP Layers
        _curr_units = self.hparams.input_embed_dim
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
            if self.hparams.embedding_bias:
                x = x+self.cat_embedding_bias
            if self.hparams.embedding_dropout != 0:
                x = self.embed_dropout(x)
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
            if self.hparams.embedding_bias:
                x_cont = x_cont+self.cont_embedding_bias
            # (B, N, E)
            x = x_cont if x is None else torch.cat([x, x_cont], 1)

        x = self.add_cls(x)
        for i, block in enumerate(self.transformer_blocks):
            x = block(x)
            if self.hparams.attn_feature_importance:
                self.attention_weights_[i] = block.mha.attn_weights
                # self.feature_importance_+=block.mha.attn_weights[:,:,:,-1].sum(dim=1)
                # self._calculate_feature_importance(block.mha.attn_weights)
        if self.hparams.attn_feature_importance:
            self._calculate_feature_importance()
        # Flatten (Batch, N_Categorical, Hidden) --> (Batch, N_CategoricalxHidden)
        # x = rearrange(x, "b n h -> b (n h)")
        # Taking only CLS token for the prediction head
        x = self.linear_layers(x[:, -1])
        return x
    
    #Not Tested Properly
    def _calculate_feature_importance(self):
        # if self.feature_importance_.device != self.device:
        #     self.feature_importance_ = self.feature_importance_.to(self.device)

        n, h, f, _ = self.attention_weights_[0].shape
        L = len(self.attention_weights_)
        self.local_feature_importance = torch.zeros((n,f), device=self.device)
        for attn_weights in self.attention_weights_:
            self.local_feature_importance+=attn_weights[:,:,:,-1].sum(dim=1)
        self.local_feature_importance = (1/(h*L))*self.local_feature_importance[:,:-1]
        self.feature_importance_ = self.local_feature_importance.mean(dim=0)
        # self.feature_importance_count_+=attn_weights.shape[0]


class FTTransformerModel(BaseModel):
    def __init__(self, config: DictConfig, **kwargs):
        super().__init__(config, **kwargs)

    def _build_network(self):
        # Backbone
        self.backbone = FTTransformerBackbone(self.hparams)
        self.dropout = nn.Dropout(self.hparams.out_ff_dropout)
        # Adding the last layer
        self.output_layer = nn.Linear(
            self.backbone.output_dim, self.hparams.output_dim
        )  # output_dim auto-calculated from other config
        _initialize_layers(
            self.hparams.out_ff_activation,
            self.hparams.out_ff_initialization,
            self.output_layer,
        )

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

    def extract_embedding(self):
        if len(self.hparams.categorical_cols) > 0:
            return self.backbone.cat_embedding_layers
        else:
            raise ValueError(
                "Model has been trained with no categorical feature and therefore can't be used as a Categorical Encoder"
            )
    
    def feature_importance(self):
        if self.hparams.attn_feature_importance:
            importance_df = pd.DataFrame({"Features": self.hparams.categorical_cols+self.hparams.continuous_cols, "importance": self.backbone.feature_importance_.detach().cpu().numpy()})
            return importance_df
        else:
            raise ValueError("If you want Feature Importance, `attn_feature_weights` should be `True`.")
