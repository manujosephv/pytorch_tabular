# Pytorch Tabular
# Author: Manu Joseph <manujoseph@gmail.com>
# For license information, see LICENSE.TXT
"""Tabular Model"""
import warnings

import torch
import torch.nn as nn
from omegaconf import DictConfig

from pytorch_tabular.models.common.layers import Embedding1dLayer, PreEncoded1dLayer
from pytorch_tabular.utils import get_logger

from ..base_model import BaseModel
from ..common import activations
from ..common.layers import Lambda
from .architecture_blocks import DenseODSTBlock

logger = get_logger(__name__)


class NODEBackbone(nn.Module):
    def __init__(self, config: DictConfig, **kwargs):
        super().__init__()
        self.hparams = config
        # self.hparams.output_dim = (0 if self.hparams.output_dim is None else self.hparams.output_dim)  # For SSL cases where output_dim will be None
        self._build_network()

    def _build_network(self):
        if self.hparams.embed_categorical:
            self.hparams.node_input_dim = self.hparams.continuous_dim + self.hparams.embedded_cat_dim
        else:
            self.hparams.node_input_dim = self.hparams.continuous_dim + self.hparams.categorical_dim
        self.dense_block = DenseODSTBlock(
            input_dim=self.hparams.node_input_dim,
            num_trees=self.hparams.num_trees,
            num_layers=self.hparams.num_layers,
            tree_output_dim=self.hparams.output_dim + self.hparams.additional_tree_output_dim,
            max_features=self.hparams.max_features,
            input_dropout=self.hparams.input_dropout,
            depth=self.hparams.depth,
            choice_function=getattr(activations, self.hparams.choice_function),
            bin_function=getattr(activations, self.hparams.bin_function),
            initialize_response_=getattr(nn.init, self.hparams.initialize_response + "_"),
            initialize_selection_logits_=getattr(nn.init, self.hparams.initialize_selection_logits + "_"),
            threshold_init_beta=self.hparams.threshold_init_beta,
            threshold_init_cutoff=self.hparams.threshold_init_cutoff,
        )
        self.output_dim = self.hparams.output_dim + self.hparams.additional_tree_output_dim

    def _build_embedding_layer(self):
        if self.hparams.embed_categorical:
            embedding = Embedding1dLayer(
                continuous_dim=self.hparams.continuous_dim,
                categorical_embedding_dims=self.hparams.embedding_dims,
                embedding_dropout=self.hparams.embedding_dropout,
                batch_norm_continuous_input=self.hparams.batch_norm_continuous_input,
            )
        else:
            embedding = PreEncoded1dLayer(
                continuous_dim=self.hparams.continuous_dim,
                categorical_dim=self.hparams.categorical_dim,
                batch_norm_continuous_input=self.hparams.batch_norm_continuous_input,
                embedding_dropout=self.hparams.embedding_dropout,
            )
        return embedding

    def forward(self, x: torch.Tensor):  # TODO factor out target encoding option.
        x = self.dense_block(x)
        return x


class NODEModel(BaseModel):
    def __init__(self, config: DictConfig, **kwargs):
        super().__init__(config, **kwargs)

    def subset(self, x):
        return x[..., : self.hparams.output_dim].mean(dim=-2)

    def data_aware_initialization(self, datamodule):
        """Performs data-aware initialization for NODE"""
        logger.info("Data Aware Initialization of NODE using a forward pass with 2000 batch size....")
        # Need a big batch to initialize properly
        alt_loader = datamodule.train_dataloader(batch_size=self.hparams.data_aware_init_batch_size)
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
        self._backbone = NODEBackbone(self.hparams)
        # Embedding Layer
        self._embedding_layer = self._backbone._build_embedding_layer()
        # average first n channels of every tree, where n is the number of output targets for regression
        # and number of classes for classification
        # Not using config head because NODE has a specific head
        warnings.warn("Ignoring head config because NODE has a specific head which subsets the tree outputs")
        self._head = Lambda(self.subset)

    # def extract_embedding(self):
    #     if self.hparams.embed_categorical:
    #         if self.hparams.embedded_cat_dim != 0:
    #             return self.embedding_layer.cat_embedding_layers
    #     else:
    #         raise ValueError(
    #             "Model has been trained with no categorical feature and therefore can't be used as a Categorical Encoder"
    #         )
