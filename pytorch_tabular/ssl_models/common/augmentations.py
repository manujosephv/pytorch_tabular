from typing import Dict

import numpy as np
import torch


def mixup(batch: Dict, lam: float = 0.5) -> Dict:
    """It apply mixup augmentation, making a weighted average between a tensor
    and some random element of the tensor taking random rows

    :param batch: Tensor on which apply the mixup augmentation
    :param lam: weight in the linear combination between the original values
        and the random permutation
    """
    result = {}
    for key, value in batch.items():
        random_index = _get_random_index(value)
        result[key] = lam * value + (1 - lam) * value[random_index, :]
        result[key] = result[key].to(dtype=value.dtype)
    return result


def cutmix(batch: Dict, lam: float = 0.1) -> Dict:
    """Define how apply cutmix to a tensor

    :param batch: Tensor on which apply the cutmix augmentation
    :param lam: probability values have 0 in a binary random mask,
        so it means probability original values will
    be updated
    """
    result = {}
    for key, value in batch.items():
        random_index = _get_random_index(value)
        x_binary_mask = torch.from_numpy(np.random.choice(2, size=value.shape, p=[lam, 1 - lam]))
        x_random = value[random_index, :]
        x_noised = value.clone().detach()
        x_noised[x_binary_mask == 0] = x_random[x_binary_mask == 0]
        result[key] = x_noised
    return result


def _get_random_index(x: torch.Tensor) -> torch.Tensor:
    """Given a tensor it compute random indices between 0 and the number of the first dimension

    :param x: Tensor used to get the number of rows
    """
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    return index
