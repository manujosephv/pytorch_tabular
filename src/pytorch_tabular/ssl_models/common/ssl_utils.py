import pytorch_lightning as pl
from torch import Tensor

from pytorch_tabular.models.common import PositionWiseFeedForward


class Denoising(pl.LightningModule):
    def __init__(self, input_dim: int):
        super().__init__()
        self.mlp = PositionWiseFeedForward(d_model=input_dim, d_ff=2 * input_dim)

    def forward(self, x: Tensor):
        return {"logits": self.mlp(x)}


class Contrastive(pl.LightningModule):
    def __init__(self, input_dim: int):
        super().__init__()
        self.mlp = PositionWiseFeedForward(d_model=input_dim, d_ff=2 * input_dim)

    def forward(self, x: Tensor):
        x = x / x.norm(dim=-1, keepdim=True)
        return {"logits": self.mlp(x)}
