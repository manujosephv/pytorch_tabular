# Pytorch Tabular
# Author: Manu Joseph <manujoseph@gmail.com>
# For license information, see LICENSE.TXT
import random
from typing import Callable

import torch
import torch.nn as nn

from pytorch_tabular.models.common.activations import entmax15


class NeuralDecisionStump(nn.Module):
    def __init__(
        self,
        n_features: int,
        binning_activation: Callable = entmax15,
        feature_mask_function: Callable = entmax15,
    ):
        super().__init__()
        self._num_cutpoints = 1
        self._num_leaf = 2
        self.n_features = n_features
        self.binning_activation = binning_activation
        self.feature_mask_function = feature_mask_function
        self._build_network()

    def _build_network(self):
        if self.feature_mask_function is not None:
            # sampling a random beta distribution
            # random distribution helps with diversity in trees and feature splits
            alpha = random.uniform(0.5, 10.0)
            beta = random.uniform(0.5, 10.0)
            # with torch.no_grad():
            feature_mask = (
                torch.distributions.Beta(torch.tensor([alpha]), torch.tensor([beta]))
                .sample((self.n_features,))
                .squeeze(-1)
            )
            self.feature_mask = nn.Parameter(feature_mask, requires_grad=True)
        W = torch.linspace(
            1.0,
            self._num_cutpoints + 1.0,
            self._num_cutpoints + 1,
            requires_grad=False,
        ).reshape(1, 1, -1)
        self.register_buffer("W", W)

        cutpoints = torch.rand([self.n_features, self._num_cutpoints])
        # Append zeros to the beginning of each row
        cutpoints = torch.cat([torch.zeros([self.n_features, 1], device=cutpoints.device), cutpoints], 1)
        self.cut_points = nn.Parameter(cutpoints, requires_grad=True)
        self.leaf_responses = nn.Parameter(torch.rand(self.n_features, self._num_leaf), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.feature_mask_function is not None:
            feature_mask = self.feature_mask_function(self.feature_mask)
        # Repeat W for each batch size using broadcasting
        W = torch.ones(x.size(0), 1, 1, device=x.device) * self.W
        # Binning features
        x = torch.bmm(x.unsqueeze(-1), W) - self.cut_points.unsqueeze(0)
        x = self.binning_activation(x)  # , dim=-1)
        x = x * self.leaf_responses.unsqueeze(0)
        x = (x * feature_mask.reshape(1, -1, 1)).sum(dim=1)
        return x, feature_mask


class NeuralDecisionTree(nn.Module):
    def __init__(
        self,
        depth: int,
        n_features: int,
        dropout: float = 0,
        binning_activation: Callable = entmax15,
        feature_mask_function: Callable = entmax15,
    ):
        super().__init__()
        self.depth = depth
        self._num_cutpoints = 1
        self.n_features = n_features
        self._dropout = dropout
        self.binning_activation = binning_activation
        self.feature_mask_function = feature_mask_function
        self._build_network()

    def _build_network(self):
        for d in range(self.depth):
            for n in range(max(2 ** (d), 1)):
                self.add_module(
                    "decision_stump_{}_{}".format(d, n),
                    NeuralDecisionStump(
                        self.n_features + (2 ** (d) if d > 0 else 0),
                        self.binning_activation,
                        self.feature_mask_function,
                    ),
                )
        self.dropout = nn.Dropout(self._dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tree_input = x
        feature_masks = []
        for d in range(self.depth):
            layer_nodes = []
            layer_feature_masks = []
            for n in range(max(2 ** (d), 1)):
                leaf_nodes, feature_mask = self._modules["decision_stump_{}_{}".format(d, n)](tree_input)
                layer_nodes.append(leaf_nodes)
                layer_feature_masks.append(feature_mask)
            layer_nodes = torch.cat(layer_nodes, dim=1)
            tree_input = torch.cat([x, layer_nodes], dim=1)
            feature_masks.append(layer_feature_masks)
        return self.dropout(layer_nodes), feature_masks


class GatedFeatureLearningUnit(nn.Module):
    def __init__(
        self,
        n_features_in: int,
        n_stages: int,
        feature_mask_function: Callable = entmax15,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_features_in = n_features_in
        self.n_features_out = n_features_in
        self.feature_mask_function = feature_mask_function
        self._dropout = dropout
        self.n_stages = n_stages
        self._build_network()

    def _create_feature_mask(self):
        feature_masks = torch.cat(
            [
                torch.distributions.Beta(
                    torch.tensor([random.uniform(0.5, 10.0)]),
                    torch.tensor([random.uniform(0.5, 10.0)]),
                )
                .sample((self.n_features_in,))
                .squeeze(-1)
                for _ in range(self.n_stages)
            ]
        ).reshape(self.n_stages, self.n_features_in)
        return nn.Parameter(
            feature_masks,
            requires_grad=True,
        )

    def _build_network(self):
        self.W_in = nn.ModuleList(
            [nn.Linear(2 * self.n_features_in, 2 * self.n_features_in) for _ in range(self.n_stages)]
        )
        self.W_out = nn.ModuleList(
            [nn.Linear(2 * self.n_features_in, self.n_features_in) for _ in range(self.n_stages)]
        )

        self.feature_masks = self._create_feature_mask()
        self.dropout = nn.Dropout(self._dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for d in range(self.n_stages):
            feature = self.feature_mask_function(self.feature_masks[d]) * x
            h_in = self.W_in[d](torch.cat([feature, h], dim=-1))
            z = torch.sigmoid(h_in[:, : self.n_features_in])
            r = torch.sigmoid(h_in[:, self.n_features_in :])  # noqa: E203
            h_out = torch.tanh(self.W_out[d](torch.cat([r * h, x], dim=-1)))
            h = self.dropout((1 - z) * h + z * h_out)
        return h
