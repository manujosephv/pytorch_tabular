# Pytorch Tabular
# Author: Manu Joseph <manujoseph@gmail.com>
# For license information, see LICENSE.TXT
"""SSL Heads"""
import torch.nn as nn


class MultiTaskHead(nn.Module):
    """
    Simple Linear transformation to take last hidden representation to reconstruct inputs.
    Output is dictionary of variable type to tensor mapping.
    """

    def __init__(self, in_features, n_binary=0, n_categorical=0, n_numerical=0, cardinality=[]):
        super().__init__()
        assert n_categorical == len(cardinality), "require cardinalities for each categorical variable"
        assert n_binary + n_categorical + n_numerical, "need some targets"
        self.n_binary = n_binary
        self.n_categorical = n_categorical
        self.n_numerical = n_numerical

        self.binary_linear = nn.Linear(in_features, n_binary) if n_binary else None
        self.categorical_linears = nn.ModuleList([nn.Linear(in_features, card) for card in cardinality])
        self.numerical_linear = nn.Linear(in_features, n_numerical) if n_numerical else None

    def forward(self, features):
        outputs = dict()

        if self.binary_linear:
            outputs["binary"] = self.binary_linear(features)

        if self.categorical_linears:
            outputs["categorical"] = [linear(features) for linear in self.categorical_linears]

        if self.numerical_linear:
            outputs["continuous"] = self.numerical_linear(features)

        return outputs
