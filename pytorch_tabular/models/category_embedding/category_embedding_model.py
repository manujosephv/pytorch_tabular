# Pytorch Tabular
# Author: Manu Joseph <manujoseph@gmail.com>
# For license information, see LICENSE.TXT
"""Category Embedding Model"""
import logging

import torch.nn as nn
from omegaconf import DictConfig

from pytorch_tabular.models.common.layers import Embedding1dLayer
from pytorch_tabular.utils import _initialize_layers, _linear_dropout_bn

from ..base_model import BaseModel

logger = logging.getLogger(__name__)


class CategoryEmbeddingBackbone(nn.Module):
    def __init__(self, config: DictConfig, **kwargs):
        super().__init__()
        self.hparams = config
        self._build_network()

    def _build_network(self):
        self.embedding = Embedding1dLayer(
            continuous_dim=self.hparams.continuous_dim,
            categorical_embedding_dims=self.hparams.embedding_dims,
            embedding_dropout=self.hparams.embedding_dropout,
            batch_norm_continuous_input=self.hparams.batch_norm_continuous_input,
        )
        # Linear Layers
        layers = []
        _curr_units = self.hparams.embedded_cat_dim + self.hparams.continuous_dim
        for units in self.hparams.layers.split("-"):
            layers.extend(
                _linear_dropout_bn(
                    self.hparams.activation,
                    self.hparams.initialization,
                    self.hparams.use_batch_norm,
                    _curr_units,
                    int(units),
                    self.hparams.dropout,
                )
            )
            _curr_units = int(units)
        self.linear_layers = nn.Sequential(*layers)
        _initialize_layers(
            self.hparams.activation, self.hparams.initialization, self.linear_layers
        )
        self.output_dim = _curr_units

    def forward(self, x):
        x = self.embedding(x)
        x = self.linear_layers(x)
        return x


class CategoryEmbeddingModel(BaseModel):
    def __init__(self, config: DictConfig, **kwargs):
        super().__init__(config, **kwargs)

    def _build_network(self):
        # Backbone
        self.backbone = CategoryEmbeddingBackbone(self.hparams)
        self.head = self._get_head_from_config()

    def extract_embedding(self):
        return self.backbone.embedding.cat_embedding_layers
