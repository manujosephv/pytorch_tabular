# Pytorch Tabular
# Author: Manu Joseph <manujoseph@gmail.com>
# For license information, see LICENSE.TXT
# Inspired by implementation https://github.com/ryancheunggit/tabular_dae
"""DenoisingAutoEncoder Model"""
import logging
from collections import OrderedDict, namedtuple
from typing import Dict
import omegaconf

# import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
# from einops import rearrange
from omegaconf import DictConfig
import numpy as np

# from pytorch_tabular.models.common.utils import to_one_hot

# from pytorch_tabular.utils import _initialize_layers, _linear_dropout_bn
# import pytorch_tabular.models as models
from ..base_model import SSLBaseModel

logger = logging.getLogger(__name__)


class OneHot(nn.Module):
    def __init__(self, cardinality):
        super().__init__()
        self.cardinality = cardinality

    def forward(self, x):
        return F.one_hot(x, self.cardinality)


class SwapNoiseCorrupter(nn.Module):
    """
    Apply swap noise on the input data.
    Each data point has specified chance be replaced by a random value from the same column.
    """

    def __init__(self, probas):
        super().__init__()
        self.probas = torch.from_numpy(np.array(probas))

    def forward(self, x):
        should_swap = torch.bernoulli(
            self.probas.to(x.device) * torch.ones((x.shape)).to(x.device)
        )
        corrupted_x = torch.where(should_swap == 1, x[torch.randperm(x.shape[0])], x)
        mask = (corrupted_x != x).float()
        return corrupted_x, mask


class MultiTaskHead(nn.Module):
    """
    Simple Linear transformation to take last hidden representation to reconstruct inputs.
    Output is dictionary of variable type to tensor mapping.
    """

    def __init__(
        self, in_features, n_binary=0, n_categorical=0, n_numerical=0, cardinality=[]
    ):
        super().__init__()
        assert n_categorical == len(
            cardinality
        ), "require cardinalities for each categorical variable"
        assert n_binary + n_categorical + n_numerical, "need some targets"
        self.n_binary = n_binary
        self.n_categorical = n_categorical
        self.n_numerical = n_numerical

        self.binary_linear = nn.Linear(in_features, n_binary) if n_binary else None
        self.categorical_linears = nn.ModuleList(
            [nn.Linear(in_features, card) for card in cardinality]
        )
        self.numerical_linear = (
            nn.Linear(in_features, n_numerical) if n_numerical else None
        )

    def forward(self, features):
        outputs = dict()

        if self.binary_linear:
            outputs["binary"] = self.binary_linear(features)

        if self.categorical_linears:
            outputs["categorical"] = [
                linear(features) for linear in self.categorical_linears
            ]

        if self.numerical_linear:
            outputs["continuous"] = self.numerical_linear(features)

        return outputs


class DenoisingAutoEncoderModel(SSLBaseModel):
    output_tuple = namedtuple("output_tuple", ["original", "reconstructed"])

    def __init__(self, encoder, decoder, config: DictConfig, **kwargs):
        super().__init__(encoder, decoder, config, **kwargs)

    def _get_noise_probability(self, name):
        return self.hparams.noise_probabilities.get(name, self.hparams.default_noise_probability)

    def _build_network(self):
        swap_probabilities = []
        binary_feat_idx = []
        onehot_feat_idx = []
        embedding_feat_idx = []
        embd_layers = []
        one_hot_layers = []
        for i, (name, (cardinality, embed_dim)) in enumerate(
            zip(self.hparams.categorical_cols, self.hparams.embedding_dims)
        ):  # conditions based on real cardinality (excluding missing value placeholder)
            if cardinality == 2:
                binary_feat_idx.append(i)
                swap_probabilities += [self._get_noise_probability(name)]
            elif cardinality <= self.hparams.max_onehot_cardinality:
                onehot_feat_idx.append(i)
                swap_probabilities += [
                    self._get_noise_probability(name)
                ] * cardinality
                one_hot_layers.append(OneHot(cardinality))
            else:
                embedding_feat_idx.append(i)
                swap_probabilities += [
                    self._get_noise_probability(name)
                ] * embed_dim
                embd_layers.append(nn.Embedding(cardinality, embed_dim))
        for name in self.hparams.continuous_cols:
            swap_probabilities.append(self._get_noise_probability(name))

        if self.hparams.categorical_dim > 0:
            # Embedding layers
            self.embedding_layers = nn.ModuleList(embd_layers)
            self.onehot_layers = nn.ModuleList(one_hot_layers)
            self._onehot_feat_idx = onehot_feat_idx
            self._binary_feat_idx = binary_feat_idx
            self._embedding_feat_idx = embedding_feat_idx
        self._swap_probabilities = swap_probabilities
        self.swap_noise = SwapNoiseCorrupter(swap_probabilities)
        self.reconstruction = MultiTaskHead(
            self.decoder.output_dim,
            n_binary=len(binary_feat_idx),
            n_categorical=len(onehot_feat_idx),
            n_numerical=len(embedding_feat_idx) + len(self.hparams.continuous_cols),
        )
        self.mask_reconstruction = nn.Linear(
            self.decoder.output_dim, len(swap_probabilities)
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

    def _embed_input(self, x: Dict):
        # (B, N)
        continuous_data, categorical_data = x["continuous"], x["categorical"]
        if self.hparams.categorical_dim > 0:
            x_cat = []
            x_binary = []
            x_embed = []
            for i in range(len(self.hparams.categorical_cols)):
                if i in self._binary_feat_idx:
                    x_binary.append(categorical_data[:, i:i+1])
                elif i in self._onehot_feat_idx:
                    x_cat.append(self.one_hot_layers[i](categorical_data[:, i]))
                else:
                    x_embed.append(self.embedding_layers[i](categorical_data[:, i]))
            # (B, N, E)
            x_cat = torch.cat(x_cat, 1) if len(x_cat)>0 else None
            x_binary = torch.cat(x_binary, 1) if len(x_binary)>0 else None
            x_embed = torch.cat(x_embed, 1) if len(x_embed)>0 else None
        all_none =  (x_cat is None) and (x_binary is None) and (x_embed is None)
        assert not all_none, "All inputs can't be none!"
        return OrderedDict(
            binary=x_binary, categorical=x_cat, continuous=continuous_data, embedding=x_embed
        )

    def forward(self, x: Dict):
        # (B, N, E)
        x = self._embed_input(x)
        x_embed = torch.cat([item for item in x.values() if item is not None], 1)
        # swap noise
        with torch.no_grad():
            corrupted_x, mask = self.swap_noise(x_embed)
        # encoder
        z = self.encoder(corrupted_x) #TODO need to change encoder to separate embedding flow
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

    def extract_embedding(self):
        if self.hparams.categorical_dim > 0:
            return self.backbone.cat_embedding_layers
        else:
            raise ValueError(
                "Model has been trained with no categorical feature and therefore can't be used as a Categorical Encoder"
            )

    