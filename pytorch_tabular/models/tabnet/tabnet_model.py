# Pytorch Tabular
# Author: Manu Joseph <manujoseph@gmail.com>
# For license information, see LICENSE.TXT
"""TabNet Model"""
import logging
from typing import Dict

import torch
import torch.nn as nn
from omegaconf import DictConfig
from pytorch_tabnet.tab_network import TabNet

from ..base_model import BaseModel

logger = logging.getLogger(__name__)


class TabNetModel(BaseModel):
    def __init__(self, config: DictConfig, **kwargs):
        super().__init__(config, **kwargs)

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

    def forward(self, x: Dict):
        # unpacking into a tuple
        x = x["categorical"], x["continuous"]
        # eliminating None in case there is no categorical or continuous columns
        x = (item for item in x if len(item) > 0)
        x = torch.cat(tuple(x), dim=1)
        # Returns output and Masked Loss. We only need the output
        x, _ = self.tabnet(x)
        if (
            (self.hparams.task == "regression")
            and (self.hparams.target_range is not None)
        ):
            for i in range(self.hparams.output_dim):
                y_min, y_max = self.hparams.target_range[i]
                x[:, i] = y_min + nn.Sigmoid()(x[:, i]) * (y_max - y_min)
        return x
