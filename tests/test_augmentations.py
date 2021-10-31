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
    torch.manual_seed(0)
    np.random.seed(0)
    x = torch.Tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    lam = 0.5
    expected = torch.Tensor([[2.0, 2.0], [1.5, 1.5], [2.5, 2.5]]).numpy()
    actual = mixup(batch={"x": x}, lam=lam)
    np.testing.assert_array_equal(actual["x"].numpy(), expected)


def test_cutmix():
    torch.manual_seed(0)
    np.random.seed(0)
    x = torch.Tensor([[1, 1], [2, 2], [3, 3]])
    lam = 0.5
    expected = torch.Tensor([[1.0, 1.0], [2.0, 2.0], [2.0, 3.0]]).numpy()
    actual = cutmix(batch={"x": x}, lam=lam)
    np.testing.assert_array_equal(actual["x"].numpy(), expected)
