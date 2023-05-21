# W605
import torch
import torch.nn.functional as F
from entmax import entmax15, sparsemax
from torch import nn
from torch.autograd import Function
from torch.jit import script

from pytorch_tabular.models.common.layers import PositionWiseFeedForward


# GLU Variants Improve Transformer https://arxiv.org/pdf/2002.05202.pdf
class GEGLU(nn.Module):
    """Gated Exponential Linear Unit (GEGLU)"""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        Args:
            d_model: dimension of the model
            d_ff: dimension of the feedforward layer
            dropout: dropout probability
        """
        super().__init__()
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout, nn.GELU(), True, False, False, False)

    def forward(self, x: torch.Tensor):
        return self.ffn(x)


class ReGLU(nn.Module):
    """ReGLU."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        Args:
            d_model: dimension of the model
            d_ff: dimension of the feedforward layer
            dropout: dropout probability
        """
        super().__init__()
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout, nn.ReLU(), True, False, False, False)

    def forward(self, x: torch.Tensor):
        return self.ffn(x)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        Args:
            d_model: dimension of the model
            d_ff: dimension of the feedforward layer
            dropout: dropout probability
        """
        super().__init__()
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout, nn.SiLU(), True, False, False, False)

    def forward(self, x: torch.Tensor):
        return self.ffn(x)


class Entmoid15(Function):
    """A highly optimized equivalent of labda x: Entmax15([x, 0])"""

    @staticmethod
    def forward(ctx, input):
        output = Entmoid15._forward(input)
        ctx.save_for_backward(output)
        return output

    @staticmethod
    @script
    def _forward(input):
        input, is_pos = abs(input), input >= 0
        tau = (input + torch.sqrt(F.relu(8 - input**2))) / 2
        tau.masked_fill_(tau <= input, 2.0)
        y_neg = 0.25 * F.relu(tau - input, inplace=True) ** 2
        return torch.where(is_pos, 1 - y_neg, y_neg)

    @staticmethod
    def backward(ctx, grad_output):
        return Entmoid15._backward(ctx.saved_tensors[0], grad_output)

    @staticmethod
    @script
    def _backward(output, grad_output):
        gppr0, gppr1 = output.sqrt(), (1 - output).sqrt()
        grad_input = grad_output * gppr0
        q = grad_input / (gppr0 + gppr1)
        grad_input -= q * gppr0
        return grad_input


def sparsemoid(input):
    return (0.5 * input + 0.5).clamp_(0, 1)


entmoid15 = Entmoid15.apply
entmax15 = entmax15
sparsemax = sparsemax
