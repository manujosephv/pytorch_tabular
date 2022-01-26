# Pytorch Tabular
# Author: Manu Joseph <manujoseph@gmail.com>
# For license information, see LICENSE.TXT
"""Tabular Model"""
import logging
from typing import Dict

import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig

from ..base_model import BaseModel
from . import utils as utils
from .architecture_blocks import DenseODSTBlock

logger = logging.getLogger(__name__)


class NODEBackbone(pl.LightningModule):
    def __init__(self, config: DictConfig, **kwargs):
        super().__init__()
        if config.embed_categorical:
            self.embedding_cat_dim = sum([y for x, y in config.embedding_dims])
        self.save_hyperparameters(config)
        self._build_network()

    def _build_network(self):
        if self.hparams.embed_categorical:
            self.embedding_layers = nn.ModuleList(
                [nn.Embedding(x, y) for x, y in self.hparams.embedding_dims]
            )
            if self.hparams.embedding_dropout != 0 and self.embedding_cat_dim != 0:
                self.embedding_dropout = nn.Dropout(self.hparams.embedding_dropout)
            self.hparams.node_input_dim = (
                self.hparams.continuous_dim + self.embedding_cat_dim
            )
        else:
            self.hparams.node_input_dim = (
                self.hparams.continuous_dim + self.hparams.categorical_dim
            )
        self.dense_block = DenseODSTBlock(
            input_dim=self.hparams.node_input_dim,
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
        self.output_dim = (
            self.hparams.output_dim + self.hparams.additional_tree_output_dim
        )

    def unpack_input(self, x: Dict):
        if self.hparams.embed_categorical:
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
        else:
            # unpacking into a tuple
            x = x["categorical"], x["continuous"]
            # eliminating None in case there is no categorical or continuous columns
            x = (item for item in x if len(item) > 0)
            x = torch.cat(tuple(x), dim=1)
        return x

    def forward(self, x):
        x = self.unpack_input(x)
        if self.hparams.embed_categorical:
            if self.hparams.embedding_dropout != 0 and self.embedding_cat_dim != 0:
                x = self.embedding_dropout(x)
        x = self.dense_block(x)
        return x


class NODEModel(BaseModel):
    def __init__(self, config: DictConfig, **kwargs):
        super().__init__(config, **kwargs)

    def subset(self, x):
        return x[..., : self.hparams.output_dim].mean(dim=-2)

    def data_aware_initialization(self, datamodule):
        """Performs data-aware initialization for NODE"""
        logger.info("Data Aware Initialization....")
        # Need a big batch to initialize properly
        alt_loader = datamodule.train_dataloader(batch_size=2000)
        batch = next(iter(alt_loader))
        for k, v in batch.items():
            if isinstance(v, list) and (len(v) == 0):
                # Skipping empty list
                continue
            # batch[k] = v.to("cpu" if self.config.gpu == 0 else "cuda")
            batch[k] = v.to(self.device)

        # single forward pass to initialize the ODST
        with torch.no_grad():
            self(batch)

    def _build_network(self):
        self.backbone = NODEBackbone(self.hparams)
        # average first n channels of every tree, where n is the number of output targets for regression
        # and number of classes for classification

        self.head = utils.Lambda(self.subset)

    def extract_embedding(self):
        if self.hparams.embed_categorical:
            if self.backbone.embedding_cat_dim != 0:
                return self.backbone.embedding_layers
        else:
            raise ValueError(
                "Model has been trained with no categorical feature and therefore can't be used as a Categorical Encoder"
            )
