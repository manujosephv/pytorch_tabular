# Pytorch Tabular
# Author: Manu Joseph <manujoseph@gmail.com>
# For license information, see LICENSE.TXT
# Inspired by implementation https://github.com/ryancheunggit/tabular_dae
"""DenoisingAutoEncoder Model"""
import logging
from collections import OrderedDict, namedtuple
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig

from ..base_model import SSLBaseModel
from ..common.heads import MultiTaskHead
from ..common.noise_generators import SwapNoiseCorrupter
from ..common.utils import OneHot

logger = logging.getLogger(__name__)


class MixedEmbedding1dLayer(nn.Module):
    """
    Enables different values in a categorical features to have different embeddings
    """

    def __init__(
        self,
        continuous_dim: int,
        categorical_embedding_dims: Tuple[int, int],
        max_onehot_cardinality: int = 4,
        embedding_dropout: float = 0.0,
        batch_norm_continuous_input: bool = False,
    ):
        super(MixedEmbedding1dLayer, self).__init__()
        self.continuous_dim = continuous_dim
        self.categorical_embedding_dims = categorical_embedding_dims
        self.categorical_dim = len(categorical_embedding_dims)
        self.batch_norm_continuous_input = batch_norm_continuous_input

        binary_feat_idx = []
        onehot_feat_idx = []
        embedding_feat_idx = []
        embd_layers = {}
        one_hot_layers = {}
        for i, (cardinality, embed_dim) in enumerate(categorical_embedding_dims):
            # conditions based on enhanced cardinality (including missing/new value placeholder)
            if cardinality == 2:
                binary_feat_idx.append(i)
            elif cardinality <= max_onehot_cardinality:
                onehot_feat_idx.append(i)
                one_hot_layers[str(i)]=OneHot(cardinality)
            else:
                embedding_feat_idx.append(i)
                embd_layers[str(i)]=nn.Embedding(cardinality, embed_dim)

        if self.categorical_dim > 0:
            # Embedding layers
            self.embedding_layers = nn.ModuleDict(embd_layers)
            self.one_hot_layers = nn.ModuleDict(one_hot_layers)
        self._onehot_feat_idx = onehot_feat_idx
        self._binary_feat_idx = binary_feat_idx
        self._embedding_feat_idx = embedding_feat_idx

        if embedding_dropout > 0:
            self.embd_dropout = nn.Dropout(embedding_dropout)
        else:
            self.embd_dropout = None
        # Continuous Layers
        if batch_norm_continuous_input:
            self.normalizing_batch_norm = nn.BatchNorm1d(continuous_dim)

    @property
    def embedded_cat_dim(self):
        return sum([embd_dim for i, (_, embd_dim) in enumerate(self.categorical_embedding_dims) if i in self._embedding_feat_idx])

    def forward(self, x: Dict[str, Any]) -> torch.Tensor:
        assert (
            "continuous" in x or "categorical" in x
        ), "x must contain either continuous and categorical features"
        # (B, N)
        continuous_data, categorical_data = x.get(
            "continuous", torch.empty(0, 0)
        ), x.get("categorical", torch.empty(0, 0))
        assert categorical_data.shape[1] == len(
            self._onehot_feat_idx + self._binary_feat_idx + self._embedding_feat_idx
        ), "categorical_data must have same number of columns as categorical embedding layers"
        assert (
            continuous_data.shape[1] == self.continuous_dim
        ), "continuous_data must have same number of columns as continuous dim"
        # embed = None
        if continuous_data.shape[1] > 0:
            if self.batch_norm_continuous_input:
                continuous_data = self.normalizing_batch_norm(continuous_data)
            # (B, N, C)
        if categorical_data.shape[1] > 0:
            x_cat = []
            x_cat_orig = []
            x_binary = []
            x_embed = []
            for i in range(self.categorical_dim):
                if i in self._binary_feat_idx:
                    x_binary.append(categorical_data[:, i : i + 1])
                elif i in self._onehot_feat_idx:
                    x_cat.append(self.one_hot_layers[str(i)](categorical_data[:, i]))
                    x_cat_orig.append(categorical_data[:, i:i+1])
                else:
                    x_embed.append(self.embedding_layers[str(i)](categorical_data[:, i]))
            # (B, N, E)
            x_cat = torch.cat(x_cat, 1) if len(x_cat) > 0 else None
            x_cat_orig = torch.cat(x_cat_orig, 1) if len(x_cat_orig) > 0 else None
            x_binary = torch.cat(x_binary, 1) if len(x_binary) > 0 else None
            x_embed = torch.cat(x_embed, 1) if len(x_embed) > 0 else None
            all_none = (x_cat is None) and (x_binary is None) and (x_embed is None)
            assert not all_none, "All inputs can't be none!"
            if self.embd_dropout is not None:
                x_embed = self.embd_dropout(x_embed)
        else:
            x_cat = None
            x_cat_orig = None
            x_binary = None
            x_embed = None
        return OrderedDict(
            binary=x_binary,
            categorical=x_cat,
            _categorical_orig=x_cat_orig,
            continuous=continuous_data,
            embedding=x_embed,
        )


class DenoisingAutoEncoderFeaturizer(nn.Module):
    output_tuple = namedtuple("output_tuple", ["features", "mask"])

    def __init__(self, encoder, config: DictConfig, **kwargs):
        super().__init__()
        self.encoder = encoder
        self.config = config
        self._build_network()

    def _get_noise_probability(self, name):
        return self.config.noise_probabilities.get(
            name, self.config.default_noise_probability
        )

    def _build_embedding_layer(self):
        return MixedEmbedding1dLayer(
            continuous_dim=self.config.continuous_dim,
            categorical_embedding_dims=self.config.embedding_dims,
            max_onehot_cardinality=self.config.max_onehot_cardinality,
            embedding_dropout=self.config.embedding_dropout,
            batch_norm_continuous_input=self.config.batch_norm_continuous_input,
        )

    def _build_network(self):
        swap_probabilities = []
        # binary_feat_idx = []
        # onehot_feat_idx = []
        # embedding_feat_idx = []
        # embd_layers = []
        # one_hot_layers = []
        # for i, (name, (cardinality, embed_dim)) in enumerate(
        #     zip(self.config.categorical_cols, self.config.embedding_dims)
        # ):  # conditions based on real cardinality (excluding missing value placeholder)
        #     if cardinality == 2:
        #         binary_feat_idx.append(i)
        #         swap_probabilities += [self._get_noise_probability(name)]
        #     elif cardinality <= self.config.max_onehot_cardinality:
        #         onehot_feat_idx.append(i)
        #         swap_probabilities += [self._get_noise_probability(name)] * cardinality
        #         one_hot_layers.append(OneHot(cardinality))
        #     else:
        #         embedding_feat_idx.append(i)
        #         swap_probabilities += [self._get_noise_probability(name)] * embed_dim
        #         embd_layers.append(nn.Embedding(cardinality, embed_dim))
        # for name in self.config.continuous_cols:
        #     swap_probabilities.append(self._get_noise_probability(name))

        swap_probabilities = []
        for i, (name, (cardinality, embed_dim)) in enumerate(
            zip(self.config.categorical_cols, self.config.embedding_dims)
        ):  # conditions based on real cardinality (excluding missing value placeholder)
            if cardinality == 2:
                swap_probabilities += [self._get_noise_probability(name)]
            elif cardinality <= self.config.max_onehot_cardinality:
                swap_probabilities += [self._get_noise_probability(name)] * cardinality
            else:
                swap_probabilities += [self._get_noise_probability(name)] * embed_dim
        for name in self.config.continuous_cols:
            swap_probabilities.append(self._get_noise_probability(name))

        # if self.config.categorical_dim > 0:
        #     # Embedding layers
        #     self.embedding_layers = nn.ModuleList(embd_layers)
        #     self.onehot_layers = nn.ModuleList(one_hot_layers)
        # self._onehot_feat_idx = onehot_feat_idx
        # self._binary_feat_idx = binary_feat_idx
        # self._embedding_feat_idx = embedding_feat_idx
        self._swap_probabilities = swap_probabilities
        self.swap_noise = SwapNoiseCorrupter(swap_probabilities)
        # self.reconstruction = MultiTaskHead(
        #     self.decoder.output_dim,
        #     n_binary=len(binary_feat_idx),
        #     n_categorical=len(onehot_feat_idx),
        #     n_numerical=len(embedding_feat_idx) + len(self.config.continuous_cols),
        # )
        # self.mask_reconstruction = nn.Linear(
        #     self.decoder.output_dim, len(swap_probabilities)
        # )

    # def _embed_input(self, x: Dict):
    #     # (B, N)
    #     continuous_data, categorical_data = x["continuous"], x["categorical"]
    #     if self.config.categorical_dim > 0:
    #         x_cat = []
    #         x_binary = []
    #         x_embed = []
    #         for i in range(len(self.config.categorical_cols)):
    #             if i in self._binary_feat_idx:
    #                 x_binary.append(categorical_data[:, i : i + 1])
    #             elif i in self._onehot_feat_idx:
    #                 x_cat.append(self.one_hot_layers[i](categorical_data[:, i]))
    #             else:
    #                 x_embed.append(self.embedding_layers[i](categorical_data[:, i]))
    #         # (B, N, E)
    #         x_cat = torch.cat(x_cat, 1) if len(x_cat) > 0 else None
    #         x_binary = torch.cat(x_binary, 1) if len(x_binary) > 0 else None
    #         x_embed = torch.cat(x_embed, 1) if len(x_embed) > 0 else None
    #         all_none = (x_cat is None) and (x_binary is None) and (x_embed is None)
    #         assert not all_none, "All inputs can't be none!"
    #     else:
    #         x_cat = None
    #         x_binary = None
    #         x_embed = None
    #     return OrderedDict(
    #         binary=x_binary,
    #         categorical=x_cat,
    #         continuous=continuous_data,
    #         embedding=x_embed,
    #     )

    def forward(self, x: Dict, perturb: bool = True):
        # (B, N, E)
        # x = self._embed_input(x)
        pick_keys = ['binary','categorical','continuous','embedding']
        x = torch.cat([x[key] for key in pick_keys if x[key] is not None], 1)
        # x = torch.cat([item for item in x.values() if item is not None], 1)
        mask = None
        if perturb:
            # swap noise
            with torch.no_grad():
                x, mask = self.swap_noise(x)
        # encoder
        z = self.encoder(x)
        return self.output_tuple(z, mask)


class DenoisingAutoEncoderModel(SSLBaseModel):
    output_tuple = namedtuple("output_tuple", ["original", "reconstructed"])

    def __init__(self, config: DictConfig, **kwargs):
        encoded_cat_dims = 0
        inferred_config = kwargs.get("inferred_config")
        encoder_config = kwargs.get("encoder_config")
        for card, embd_dim in inferred_config.embedding_dims:
            if card == 2:
                encoded_cat_dims += 1
            elif card <= config.max_onehot_cardinality:
                encoded_cat_dims+= card
            else:
                encoded_cat_dims+=embd_dim
        config.encoder_config._backbone_input_dim = encoded_cat_dims + len(config.continuous_cols)
        super().__init__(config, **kwargs)

    def _get_noise_probability(self, name):
        return self.hparams.noise_probabilities.get(
            name, self.hparams.default_noise_probability
        )

    def _build_network(self):
        self.featurizer = DenoisingAutoEncoderFeaturizer(self.encoder, self.hparams)
        self.embedding = self.featurizer._build_embedding_layer()
        self.reconstruction = MultiTaskHead(
            self.decoder.output_dim,
            n_binary=len(self.embedding._binary_feat_idx),
            n_categorical=len(self.embedding._onehot_feat_idx),
            n_numerical=self.embedding.embedded_cat_dim
            + len(self.hparams.continuous_cols),
            cardinality=[self.embedding.categorical_embedding_dims[i][0] for i in self.embedding._onehot_feat_idx],
        )
        self.mask_reconstruction = nn.Linear(
            self.decoder.output_dim, len(self.featurizer.swap_noise.probas)
        )

    def _setup_loss(self):
        self.losses = {
            "binary": nn.BCEWithLogitsLoss(),
            "categorical": nn.CrossEntropyLoss(),
            "continuous": nn.MSELoss(),
            "mask": nn.BCEWithLogitsLoss(),
        }

    def _setup_metrics(self):
        return None

    def forward(self, x: Dict):
        x = self.embedding(x)
        # (B, N, E)
        features = self.featurizer(x, perturb=True)
        z, mask = features.features, features.mask
        # decoder
        z_hat = self.decoder(z)
        # reconstruction
        reconstructed_in = self.reconstruction(z_hat)
        # mask reconstruction
        reconstructed_mask = self.mask_reconstruction(z_hat)
        output_dict = dict(mask=self.output_tuple(mask, reconstructed_mask))
        if "continuous" in reconstructed_in.keys():
            output_dict["continuous"] = self.output_tuple(
                torch.cat([x["continuous"], x["embedding"]], 1),
                reconstructed_in["continuous"],
            )
        if "categorical" in reconstructed_in.keys():
            output_dict["categorical"] = self.output_tuple(
                x["_categorical_orig"], reconstructed_in["categorical"]
            )
        if "binary" in reconstructed_in.keys():
            output_dict["binary"] = self.output_tuple(
                x["binary"], reconstructed_in["binary"]
            )
        return output_dict
        # return dict(
        #     continuous=self.output_tuple(
        #         torch.cat([x["continuous"], x["embedding"]], 1),
        #         reconstructed_in["continuous"],
        #     ),
        #     categorical=self.output_tuple(
        #         x["categorical"], reconstructed_in["categorical"]
        #     ),
        #     binary=self.output_tuple(x["binary"], reconstructed_in["binary"]),
        #     mask=self.output_tuple(mask, reconstructed_mask),
        # )

    def calculate_loss(self, output, tag):
        total_loss = 0
        # TODO include weights for different types
        for type, out in output.items():
            #TODO special treatment for categorical. input is one-hot encoded, output is list of logits
            if type=="categorical":
                loss = 0
                for i in range(out.original.size(-1)):
                    loss += self.losses[type](out.reconstructed[i], out.original[:,i])
            else:
                loss = self.losses[type]( out.reconstructed, out.original)
            self.log(
                f"{tag}_{type}_loss",
                loss.item(),
                on_epoch=True,
                on_step=False,
                logger=True,
                prog_bar=False,
            )
            total_loss += loss
        self.log(
            f"{tag}_loss",
            total_loss,
            on_epoch=(tag == "valid") or (tag == "test"),
            on_step=(tag == "train"),
            # on_step=False,
            logger=True,
            prog_bar=True,
        )
        return total_loss

    def calculate_metrics(self, output, tag):
        pass

    def featurize(self, x: Dict):
        x = self.embedding(x)
        return self.featurizer(x, perturb=False).features
