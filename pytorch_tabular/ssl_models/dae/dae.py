# Pytorch Tabular
# Author: Manu Joseph <manujoseph@gmail.com>
# For license information, see LICENSE.TXT
# Inspired by implementation https://github.com/ryancheunggit/tabular_dae
"""DenoisingAutoEncoder Model"""
import logging
from collections import OrderedDict, namedtuple
from typing import Dict

import torch
import torch.nn as nn
from omegaconf import DictConfig

from ..base_model import SSLBaseModel
from ..common.heads import MultiTaskHead
from ..common.noise_generators import SwapNoiseCorrupter
from ..common.utils import OneHot

logger = logging.getLogger(__name__)


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

    def _build_network(self):
        swap_probabilities = []
        binary_feat_idx = []
        onehot_feat_idx = []
        embedding_feat_idx = []
        embd_layers = []
        one_hot_layers = []
        for i, (name, (cardinality, embed_dim)) in enumerate(
            zip(self.config.categorical_cols, self.config.embedding_dims)
        ):  # conditions based on real cardinality (excluding missing value placeholder)
            if cardinality == 2:
                binary_feat_idx.append(i)
                swap_probabilities += [self._get_noise_probability(name)]
            elif cardinality <= self.config.max_onehot_cardinality:
                onehot_feat_idx.append(i)
                swap_probabilities += [self._get_noise_probability(name)] * cardinality
                one_hot_layers.append(OneHot(cardinality))
            else:
                embedding_feat_idx.append(i)
                swap_probabilities += [self._get_noise_probability(name)] * embed_dim
                embd_layers.append(nn.Embedding(cardinality, embed_dim))
        for name in self.config.continuous_cols:
            swap_probabilities.append(self._get_noise_probability(name))

        if self.config.categorical_dim > 0:
            # Embedding layers
            self.embedding_layers = nn.ModuleList(embd_layers)
            self.onehot_layers = nn.ModuleList(one_hot_layers)
        self._onehot_feat_idx = onehot_feat_idx
        self._binary_feat_idx = binary_feat_idx
        self._embedding_feat_idx = embedding_feat_idx
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

    def _embed_input(self, x: Dict):
        # (B, N)
        continuous_data, categorical_data = x["continuous"], x["categorical"]
        if self.config.categorical_dim > 0:
            x_cat = []
            x_binary = []
            x_embed = []
            for i in range(len(self.config.categorical_cols)):
                if i in self._binary_feat_idx:
                    x_binary.append(categorical_data[:, i : i + 1])
                elif i in self._onehot_feat_idx:
                    x_cat.append(self.one_hot_layers[i](categorical_data[:, i]))
                else:
                    x_embed.append(self.embedding_layers[i](categorical_data[:, i]))
            # (B, N, E)
            x_cat = torch.cat(x_cat, 1) if len(x_cat) > 0 else None
            x_binary = torch.cat(x_binary, 1) if len(x_binary) > 0 else None
            x_embed = torch.cat(x_embed, 1) if len(x_embed) > 0 else None
        all_none = (x_cat is None) and (x_binary is None) and (x_embed is None)
        assert not all_none, "All inputs can't be none!"
        return OrderedDict(
            binary=x_binary,
            categorical=x_cat,
            continuous=continuous_data,
            embedding=x_embed,
        )

    def forward(self, x: Dict, perturb: bool = True):
        # (B, N, E)
        x = self._embed_input(x)
        x = torch.cat([item for item in x.values() if item is not None], 1)
        mask = None
        if perturb:
            # swap noise
            with torch.no_grad():
                x, mask = self.swap_noise(x)
        # encoder
        z = self.encoder(dict(continuous=x)) #TODO Need to separate embedding to resuse backbones
        return self.output_tuple(z, mask)


class DenoisingAutoEncoderModel(SSLBaseModel):
    output_tuple = namedtuple("output_tuple", ["original", "reconstructed"])

    def __init__(self, config: DictConfig, **kwargs):
        super().__init__(config, **kwargs)

    def _get_noise_probability(self, name):
        return self.hparams.noise_probabilities.get(
            name, self.hparams.default_noise_probability
        )

    def _build_network(self):
        self.featurizer = DenoisingAutoEncoderFeaturizer(self.encoder, self.hparams)
        self.reconstruction = MultiTaskHead(
            self.decoder.output_dim,
            n_binary=len(self.featurizer._binary_feat_idx),
            n_categorical=len(self.featurizer._onehot_feat_idx),
            n_numerical=len(self.featurizer._embedding_feat_idx)
            + len(self.hparams.continuous_cols),
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
        # (B, N, E)
        features = self.featurizer(x, perturb=True)
        z, mask = features.features, features.mask
        # decoder
        z_hat = self.decoder(z)
        # reconstruction
        reconstructed_in = self.reconstruction(z_hat)
        # mask reconstruction
        reconstructed_mask = self.mask_reconstruction(z_hat)
        return dict(
            continuous=self.output_tuple(
                torch.cat([x["continuous"], x["embedding"]], 1),
                reconstructed_in["continuous"],
            ),
            categorical=self.output_tuple(
                x["categorical"], reconstructed_in["categorical"]
            ),
            binary=self.output_tuple(x["binary"], reconstructed_in["binary"]),
            mask=self.output_tuple(mask, reconstructed_mask),
        )

    def calculate_loss(self, output, tag):
        total_loss = 0
        for type, out in output.items():
            loss = self.losses[type](out.original, out.reconstructed)
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
        return self.featurizer(x, perturb=False).features
