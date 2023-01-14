# Pytorch Tabular
# Author: Manu Joseph <manujoseph@gmail.com>
# For license information, see LICENSE.TXT
"""TabNet Model"""
from typing import Dict

import torch
import torch.nn as nn
from omegaconf import DictConfig
from pytorch_tabnet.tab_network import TabNet

from ..base_model import BaseModel


class TabNetBackbone(nn.Module):
    def __init__(self, config: DictConfig, **kwargs):
        super().__init__()
        self.hparams = config
        self._build_network()

    def _build_network(self):
        self.tabnet = TabNet(
            input_dim=self.hparams.continuous_dim + self.hparams.categorical_dim,
            output_dim=self.hparams.output_dim,
            n_d=self.hparams.n_d,
            n_a=self.hparams.n_a,
            n_steps=self.hparams.n_steps,
            gamma=self.hparams.gamma,
            cat_idxs=[i for i in range(self.hparams.categorical_dim)],
            cat_dims=[cardinality for cardinality, _ in self.hparams.embedding_dims],
            cat_emb_dim=[embed_dim for _, embed_dim in self.hparams.embedding_dims],
            n_independent=self.hparams.n_independent,
            n_shared=self.hparams.n_shared,
            epsilon=1e-15,
            virtual_batch_size=self.hparams.virtual_batch_size,
            momentum=0.02,
            mask_type=self.hparams.mask_type,
        )

    def unpack_input(self, x: Dict):
        # unpacking into a tuple
        x = x["categorical"], x["continuous"]
        # eliminating None in case there is no categorical or continuous columns
        x = (item for item in x if len(item) > 0)
        x = torch.cat(tuple(x), dim=1)
        return x

    def forward(self, x: Dict):
        # unpacking into a tuple
        x = self.unpack_input(x)
        # Returns output and Masked Loss. We only need the output
        x, _ = self.tabnet(x)
        return x


class TabNetModel(BaseModel):
    def __init__(self, config: DictConfig, **kwargs):
        assert config.task in [
            "regression",
            "classification",
        ], "TabNet is only implemented for Regression and Classification"
        super().__init__(config, **kwargs)

    @property
    def backbone(self):
        return self._backbone

    @property
    def embedding_layer(self):
        return self._embedding_layer

    @property
    def head(self):
        return self._head

    def _build_network(self):
        # TabNet has its own embedding layer.
        # So we are not using the embedding layer from BaseModel
        self._embedding_layer = nn.Identity()
        self._backbone = TabNetBackbone(self.hparams)
        setattr(self.backbone, "output_dim", self.hparams.output_dim)
        # TabNet has its own head
        self._head = nn.Identity()

    def extract_embedding(self):
        raise ValueError("Extracting Embeddings is not supported by Tabnet. Please use another compatible model")
