# noqa W605
from collections import OrderedDict
from typing import Any, Dict, Tuple

import torch
from torch import nn

from pytorch_tabular.ssl_models.common.utils import OneHot


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
                one_hot_layers[str(i)] = OneHot(cardinality)
            else:
                embedding_feat_idx.append(i)
                embd_layers[str(i)] = nn.Embedding(cardinality, embed_dim)

        if self.categorical_dim > 0:
            # Embedding layers
            self.embedding_layer = nn.ModuleDict(embd_layers)
            self.one_hot_layers = nn.ModuleDict(one_hot_layers)
        self._onehot_feat_idx = onehot_feat_idx
        self._binary_feat_idx = binary_feat_idx
        self._embedding_feat_idx = embedding_feat_idx

        if embedding_dropout > 0 and len(embedding_feat_idx) > 0:
            self.embd_dropout = nn.Dropout(embedding_dropout)
        else:
            self.embd_dropout = None
        # Continuous Layers
        if batch_norm_continuous_input:
            self.normalizing_batch_norm = nn.BatchNorm1d(continuous_dim)

    @property
    def embedded_cat_dim(self):
        return sum(
            [
                embd_dim
                for i, (_, embd_dim) in enumerate(self.categorical_embedding_dims)
                if i in self._embedding_feat_idx
            ]
        )

    def forward(self, x: Dict[str, Any]) -> torch.Tensor:
        assert "continuous" in x or "categorical" in x, "x must contain either continuous and categorical features"
        # (B, N)
        continuous_data, categorical_data = x.get("continuous", torch.empty(0, 0)), x.get(
            "categorical", torch.empty(0, 0)
        )
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
                    x_binary.append(categorical_data[:, i : i + 1])  # noqa: E203
                elif i in self._onehot_feat_idx:
                    x_cat.append(self.one_hot_layers[str(i)](categorical_data[:, i]))
                    x_cat_orig.append(categorical_data[:, i : i + 1])  # noqa: E203
                else:
                    x_embed.append(self.embedding_layer[str(i)](categorical_data[:, i]))
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
