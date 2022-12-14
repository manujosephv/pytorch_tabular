# Pytorch Tabular
# Author: Manu Joseph <manujoseph@gmail.com>
# For license information, see LICENSE.TXT
# Inspired by implementation https://github.com/ryancheunggit/tabular_dae
"""DenoisingAutoEncoder Model"""
import logging
from collections import namedtuple
from typing import Dict

import torch
import torch.nn as nn
from omegaconf import DictConfig

from ..base_model import SSLBaseModel
from ..common.heads import MultiTaskHead
from ..common.layers import MixedEmbedding1dLayer
from ..common.noise_generators import SwapNoiseCorrupter

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

        self._swap_probabilities = swap_probabilities
        self.swap_noise = SwapNoiseCorrupter(swap_probabilities)

    def forward(self, x: Dict, perturb: bool = True):
        # (B, N, E)
        # x = self._embed_input(x)
        pick_keys = ["binary", "categorical", "continuous", "embedding"]
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
    ALLOWED_MODELS = ["CategoryEmbeddingModelConfig"]

    def __init__(self, config: DictConfig, **kwargs):
        encoded_cat_dims = 0
        inferred_config = kwargs.get("inferred_config")
        for card, embd_dim in inferred_config.embedding_dims:
            if card == 2:
                encoded_cat_dims += 1
            elif card <= config.max_onehot_cardinality:
                encoded_cat_dims += card
            else:
                encoded_cat_dims += embd_dim
        config.encoder_config._backbone_input_dim = encoded_cat_dims + len(
            config.continuous_cols
        )
        assert (
            config.encoder_config._config_name in self.ALLOWED_MODELS
        ), "Encoder must be one of the following: " + ", ".join(self.ALLOWED_MODELS)
        if config.decoder_config is not None:
            assert (
                config.decoder_config._config_name in self.ALLOWED_MODELS
            ), "Decoder must be one of the following: " + ", ".join(self.ALLOWED_MODELS)
            if "-" in config.encoder_config.layers:
                config.decoder_config._backbone_input_dim = int(
                    config.encoder_config.layers.split("-")[-1]
                )
            else:
                config.decoder_config._backbone_input_dim = int(
                    config.encoder_config.layers
                )
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
            cardinality=[
                self.embedding.categorical_embedding_dims[i][0]
                for i in self.embedding._onehot_feat_idx
            ],
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
        if self.mode == "pretrain":
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
        else:  # self.mode == "finetune"
            return features.features

    def calculate_loss(self, output, tag):
        total_loss = 0
        # TODO include weights for different types
        for type, out in output.items():
            # TODO special treatment for categorical. input is one-hot encoded, output is list of logits
            if type == "categorical":
                loss = 0
                for i in range(out.original.size(-1)):
                    loss += self.losses[type](out.reconstructed[i], out.original[:, i])
            else:
                loss = self.losses[type](out.reconstructed, out.original)
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

    @property
    def output_dim(self):
        return self.featurizer.encoder.output_dim
