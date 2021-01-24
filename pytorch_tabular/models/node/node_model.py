# Pytorch Tabular
# Author: Manu Joseph <manujoseph@gmail.com>
# For license information, see LICENSE.TXT
"""Tabular Model"""
import logging
from typing import Dict

import torch
import torch.nn as nn
from omegaconf import DictConfig

from ..base_model import BaseModel
from . import utils as utils
from .architecture_blocks import DenseODSTBlock

logger = logging.getLogger(__name__)


class NODEModel(BaseModel):
    def __init__(self, config: DictConfig, **kwargs):
        super().__init__(config, **kwargs)

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
        
        def subset(x):
            return x[..., : self.hparams.output_dim].mean(dim=-2)
        self.output_response = utils.Lambda(subset)

    def unpack_input(self, x: Dict):
        # unpacking into a tuple
        x = x["categorical"], x["continuous"]
        # eliminating None in case there is no categorical or continuous columns
        x = (item for item in x if len(item) > 0)
        x = torch.cat(tuple(x), dim=1)
        return x

    def forward(self, x: Dict):
        x = self.unpack_input(x)
        x = self.dense_block(x)
        x = self.output_response(x)
        if (
            (self.hparams.task == "regression")
            and (self.hparams.target_range is not None)
        ):
            for i in range(self.hparams.output_dim):
                y_min, y_max = self.hparams.target_range[i]
                x[:, i] = y_min + nn.Sigmoid()(x[:, i]) * (y_max - y_min)
        return x


class CategoryEmbeddingNODEModel(BaseModel):
    def __init__(self, config: DictConfig, **kwargs):
        self.embedding_cat_dim = sum([y for x, y in config.embedding_dims])
        super().__init__(config, **kwargs)

    def _build_network(self):
        self.embedding_layers = nn.ModuleList(
            [nn.Embedding(x, y) for x, y in self.hparams.embedding_dims]
        )
        if self.hparams.embedding_dropout != 0 and self.embedding_cat_dim != 0:
            self.embedding_dropout = nn.Dropout(self.hparams.embedding_dropout)
        self.dense_block = DenseODSTBlock(
            input_dim=self.hparams.continuous_dim + self.embedding_cat_dim,
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

        def subset(x):
            return x[..., : self.hparams.output_dim].mean(dim=-2)
        self.output_response = utils.Lambda(subset)

    def unpack_input(self, x):
        # unpacking into a tuple
        continuous_data, categorical_data = x["continuous"], x["categorical"]
        if self.embedding_cat_dim != 0:
            # x = []
            # for i, embedding_layer in enumerate(self.embedding_layers):
            #     x.append(embedding_layer(categorical_data[:, i]))
            x = [
                embedding_layer(categorical_data[:, i])
                for i, embedding_layer in enumerate(self.embedding_layers)
            ]
            x = torch.cat(x, 1)

        if self.hparams.continuous_dim != 0:
            if self.embedding_cat_dim != 0:
                x = torch.cat([x, continuous_data], 1)
            else:
                x = continuous_data
        return x

    def forward(self, x: Dict):
        x = self.unpack_input(x)
        if self.hparams.embedding_dropout != 0 and self.embedding_cat_dim != 0:
            x = self.embedding_dropout(x)
        x = self.dense_block(x)
        x = self.output_response(x)
        if (
            (self.hparams.task == "regression")
            and (self.hparams.target_range is not None)
        ):
            for i in range(self.hparams.output_dim):
                y_min, y_max = self.hparams.target_range[i]
                x[:, i] = y_min + nn.Sigmoid()(x[:, i]) * (y_max - y_min)
        return x
