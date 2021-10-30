from pytorch_tabular.augmentations import get_random_index, mixup, cutmix
import numpy as np
import torch


def test_get_random_index():
    torch.manual_seed(0)
    x = torch.Tensor([1, 2, 3])
    expected = np.array([2, 0, 1])
    actual = get_random_index(x).numpy()
    np.testing.assert_array_equal(actual, expected)


def test_mixup():
    x = torch.Tensor([[1, 1], [2, 2], [3, 3]])
    lam = 0.5
    random_index = torch.from_numpy(np.array([2, 0, 1]))
    expected = torch.Tensor([[2.0, 2.0], [1.5, 1.5], [2.5, 2.5]]).numpy()
    actual = mixup(x=x, random_index=random_index, lam=lam).numpy()
    np.testing.assert_array_equal(actual, expected)


def test_cutmix():
    np.random.seed(0)
    x = torch.Tensor([[1, 1], [2, 2], [3, 3]])
    lam = 0.5
    random_index = torch.from_numpy(np.array([2, 0, 1]))
    expected = torch.Tensor([[1.0, 1.0], [2.0, 2.0], [2.0, 3.0]]).numpy()
    actual = cutmix(x=x, random_index=random_index, lam=lam).numpy()
    np.testing.assert_array_equal(actual, expected)
