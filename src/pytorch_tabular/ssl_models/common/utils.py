# Pytorch Tabular
# Author: Manu Joseph <manujoseph@gmail.com>
# For license information, see LICENSE.TXT
"""Utilities"""
import torch.nn as nn
import torch.nn.functional as F


class OneHot(nn.Module):
    def __init__(self, cardinality):
        super().__init__()
        self.cardinality = cardinality

    def forward(self, x):
        return F.one_hot(x, self.cardinality)
