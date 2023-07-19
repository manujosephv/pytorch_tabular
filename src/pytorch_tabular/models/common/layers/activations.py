# W605
import torch
import torch.nn.functional as F
from entmax import entmax15, sparsemax
from torch import Tensor
from torch.autograd import Function
from torch.jit import script


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


def t_softmax(input: Tensor, t: Tensor = None, dim: int = -1) -> Tensor:
    if t is None:
        t = torch.tensor(0.5, device=input.device)
    assert (t >= 0.0).all()
    maxes = torch.max(input, dim=dim, keepdim=True).values
    input_minus_maxes = input - maxes

    w = torch.relu(input_minus_maxes + t) + 1e-8
    return torch.softmax(input_minus_maxes + torch.log(w), dim=dim)


class TSoftmax(torch.nn.Module):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, input: Tensor, t: Tensor) -> Tensor:
        return t_softmax(input, t, self.dim)


class RSoftmax(torch.nn.Module):
    def __init__(self, dim: int = -1, eps: float = 1e-8):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.tsoftmax = TSoftmax(dim=dim)

    @classmethod
    def calculate_t(cls, input: Tensor, r: Tensor, dim: int = -1, eps: float = 1e-8):
        # r represents what is the fraction of zero values that we want to have
        assert ((0.0 <= r) & (r <= 1.0)).all()

        maxes = torch.max(input, dim=dim, keepdim=True).values
        input_minus_maxes = input - maxes

        zeros_mask = torch.exp(input_minus_maxes) == 0.0
        zeros_frac = zeros_mask.sum(dim=dim, keepdim=True).float() / input_minus_maxes.shape[dim]

        q = torch.clamp((r - zeros_frac) / (1 - zeros_frac), min=0.0, max=1.0)
        x_minus_maxes = input_minus_maxes * (~zeros_mask).float()
        if q.ndim > 1:
            t = -torch.quantile(x_minus_maxes, q.view(-1), dim=dim, keepdim=True).detach()
            t = t.squeeze(dim).diagonal(dim1=-2, dim2=-1).unsqueeze(-1) + eps
        else:
            t = -torch.quantile(x_minus_maxes, q, dim=dim).detach() + eps
        return t

    def forward(self, input: Tensor, r: Tensor):
        t = RSoftmax.calculate_t(input, r, self.dim, self.eps)
        return self.tsoftmax(input, t)
