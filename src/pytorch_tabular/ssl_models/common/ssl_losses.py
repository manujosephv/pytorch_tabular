import torch.nn as nn


def loss_contrastive(y_hat, y):
    return -nn.functional.cosine_similarity(y_hat, y).add_(-1).sum()
