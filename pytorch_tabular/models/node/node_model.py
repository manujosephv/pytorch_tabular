import logging
from typing import Dict

import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig
from .architecture_blocks import DenseODSTBlock

from ..base_model import BaseModel
from . import utils as utils

logger = logging.getLogger(__name__)


class NODEModel(BaseModel):
    def __init__(self, config: DictConfig):
        super().__init__(config)

    def _build_network(self):
        self.dense_block = DenseODSTBlock(
            input_dim=self.hparams.continuous_dim + self.hparams.categorical_dim,
            num_trees=self.hparams.num_trees,
            num_layers=self.hparams.num_layers,
            tree_output_dim=self.hparams.output_dim
            + self.hparams.additional_tree_output_dim,
            max_features=self.hparams.max_features,
            input_dropout=self.hparams.input_dropout,
            depth=self.hparams.depth,
            choice_function=getattr(utils, self.hparams.choice_function),
            bin_function=getattr(utils, self.hparams.bin_function),
            initialize_response_=getattr(
                nn.init, self.hparams.initialize_response + "_"
            ),
            initialize_selection_logits_=getattr(
                nn.init, self.hparams.initialize_selection_logits + "_"
            ),
            threshold_init_beta=self.hparams.threshold_init_beta,
            threshold_init_cutoff=self.hparams.threshold_init_cutoff,
        )
        # average first n channels of every tree, where n is the number of output targets for regression
        # and number of classes for classification
        self.output_response = utils.Lambda(
            lambda x: x[..., : self.hparams.output_dim].mean(dim=-2)
        )

    def forward(self, x: Dict):
        # unpacking into a tuple
        x = x["continuous"], x["categorical"]
        # eliminating None in case there is no categorical or continuous columns
        x = (item for item in x if len(item)>0)
        x = torch.cat(tuple(x), dim=1)
        x = self.dense_block(x)
        x = self.output_response(x)
        return x
