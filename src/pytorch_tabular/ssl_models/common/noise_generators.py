# Pytorch Tabular
# Author: Manu Joseph <manujoseph@gmail.com>
# For license information, see LICENSE.TXT
# Inspired by implementation https://github.com/ryancheunggit/tabular_dae
"""DenoisingAutoEncoder Model"""
import numpy as np
import torch
import torch.nn as nn


class SwapNoiseCorrupter(nn.Module):
    """
    Apply swap noise on the input data.
    Each data point has specified chance be replaced by a random value from the same column.
    """

    def __init__(self, probas):
        super().__init__()
        self.probas = torch.from_numpy(np.array(probas))

    def forward(self, x):
        should_swap = torch.bernoulli(self.probas.to(x.device) * torch.ones((x.shape)).to(x.device))
        corrupted_x = torch.where(should_swap == 1, x[torch.randperm(x.shape[0])], x)
        mask = (corrupted_x != x).float()
        return corrupted_x, mask
