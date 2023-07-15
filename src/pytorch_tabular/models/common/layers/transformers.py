# W605
import math
from typing import Optional

import torch
from einops import rearrange
from torch import einsum, nn

from pytorch_tabular.utils import _initialize_kaiming

# from . import activations


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
        self.ffn = PositionWiseFeedForward(
            d_model, d_ff, dropout, nn.GELU(), True, False, False, False
        )

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
        self.ffn = PositionWiseFeedForward(
            d_model, d_ff, dropout, nn.ReLU(), True, False, False, False
        )

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
        self.ffn = PositionWiseFeedForward(
            d_model, d_ff, dropout, nn.SiLU(), True, False, False, False
        )

    def forward(self, x: torch.Tensor):
        return self.ffn(x)


GATED_UNITS = {"GEGLU": GEGLU, "ReGLU": ReGLU, "SwiGLU": SwiGLU}


# https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/transformers/feed_forward.py
class PositionWiseFeedForward(nn.Module):
    r"""
    title: Position-wise Feed-Forward Network (FFN)
    summary: Documented reusable implementation of the position wise feedforward network.

    # Position-wise Feed-Forward Network (FFN)
    This is a [PyTorch](https://pytorch.org)  implementation
    of position-wise feedforward network used in transformer.
    FFN consists of two fully connected layers.
    Number of dimensions in the hidden layer $d_{ff}$, is generally set to around
    four times that of the token embedding $d_{model}$.
    So it is sometime also called the expand-and-contract network.
    There is an activation at the hidden layer, which is
    usually set to ReLU (Rectified Linear Unit) activation, $$\\max(0, x)$$
    That is, the FFN function is,
    $$FFN(x, W_1, W_2, b_1, b_2) = \\max(0, x W_1 + b_1) W_2 + b_2$$
    where $W_1$, $W_2$, $b_1$ and $b_2$ are learnable parameters.
    Sometimes the
    GELU (Gaussian Error Linear Unit) activation is also used instead of ReLU.
    $$x \\Phi(x)$$ where $\\Phi(x) = P(X \\le x), X \\sim \\mathcal{N}(0,1)$
    ### Gated Linear Units
    This is a generic implementation that supports different variants including
    [Gated Linear Units](https://arxiv.org/abs/2002.05202) (GLU).
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation=nn.ReLU(),
        is_gated: bool = False,
        bias1: bool = True,
        bias2: bool = True,
        bias_gate: bool = True,
    ):
        """
        * `d_model` is the number of features in a token embedding
        * `d_ff` is the number of features in the hidden layer of the FFN
        * `dropout` is dropout probability for the hidden layer
        * `is_gated` specifies whether the hidden layer is gated
        * `bias1` specified whether the first fully connected layer should have a learnable bias
        * `bias2` specified whether the second fully connected layer should have a learnable bias
        * `bias_gate` specified whether the fully connected layer for the gate should have a learnable bias
        """
        super().__init__()
        # Layer one parameterized by weight $W_1$ and bias $b_1$
        self.layer1 = nn.Linear(d_model, d_ff, bias=bias1)
        # Layer one parameterized by weight $W_1$ and bias $b_1$
        self.layer2 = nn.Linear(d_ff, d_model, bias=bias2)
        # Hidden layer dropout
        self.dropout = nn.Dropout(dropout)
        # Activation function $f$
        self.activation = activation
        # Whether there is a gate
        self.is_gated = is_gated
        if is_gated:
            # If there is a gate the linear layer to transform inputs to
            # be multiplied by the gate, parameterized by weight $V$ and bias $c$
            self.linear_v = nn.Linear(d_model, d_ff, bias=bias_gate)

    def forward(self, x: torch.Tensor):
        # $f(x W_1 + b_1)$
        g = self.activation(self.layer1(x))
        # If gated, $f(x W_1 + b_1) \otimes (x V + b) $
        if self.is_gated:
            x = g * self.linear_v(x)
        # Otherwise
        else:
            x = g
        # Apply dropout
        x = self.dropout(x)
        # $(f(x W_1 + b_1) \otimes (x V + b)) W_2 + b_2$ or $f(x W_1 + b_1) W_2 + b_2$
        # depending on whether it is gated
        return self.layer2(x)


# Inspired by implementations
# 1. lucidrains - https://github.com/lucidrains/tab-transformer-pytorch/
# If you are interested in Transformers, you should definitely check out his repositories.
# 2. PyTorch Wide and Deep - https://github.com/jrzaurin/pytorch-widedeep/
# It is another library for tabular data, which supports multi modal problems.
# Check out the library if you haven't already.
# 3. AutoGluon - https://github.com/awslabs/autogluon
# AutoGluon is an AuttoML library which supports Tabular data as well. it is from Amazon Research and is in MXNet
# 4. LabML Annotated Deep Learning Papers - The position-wise FF was shamelessly copied from
# https://github.com/labmlai/annotated_deep_learning_paper_implementations/tree/master/labml_nn/transformers


class AddNorm(nn.Module):
    """Applies LayerNorm, Dropout and adds to input.

    Standard AddNorm operations in Transformers
    """

    def __init__(self, input_dim: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(input_dim)

    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        return self.ln(self.dropout(Y) + X)


class MultiHeadedAttention(nn.Module):
    """Multi Headed Attention Block in Transformers."""

    def __init__(
        self,
        input_dim: int,
        num_heads: int = 8,
        head_dim: int = 16,
        dropout: int = 0.1,
        keep_attn: bool = True,
    ):
        super().__init__()
        assert (
            input_dim % num_heads == 0
        ), "'input_dim' must be multiples of 'num_heads'"
        inner_dim = head_dim * num_heads
        self.n_heads = num_heads
        self.scale = head_dim**-0.5
        self.keep_attn = keep_attn

        self.to_qkv = nn.Linear(input_dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, input_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.n_heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))
        sim = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)
        if self.keep_attn:
            self.attn_weights = attn
        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)", h=h)
        return self.to_out(out)


class TransformerEncoderBlock(nn.Module):
    """A single Transformer Encoder Block."""

    def __init__(
        self,
        input_embed_dim: int,
        num_heads: int = 8,
        ff_hidden_multiplier: int = 4,
        ff_activation: str = "GEGLU",
        attn_dropout: float = 0.1,
        keep_attn: bool = True,
        ff_dropout: float = 0.1,
        add_norm_dropout: float = 0.1,
        transformer_head_dim: Optional[int] = None,
    ):
        """
        Args:
            input_embed_dim: The input embedding dimension
            num_heads: The number of attention heads
            ff_hidden_multiplier: The hidden dimension multiplier for the position-wise feed-forward layer
            ff_activation: The activation function for the position-wise feed-forward layer
            attn_dropout: The dropout probability for the attention layer
            keep_attn: Whether to keep the attention weights
            ff_dropout: The dropout probability for the position-wise feed-forward layer
            add_norm_dropout: The dropout probability for the residual connections
            transformer_head_dim: The dimension of the attention heads. If None, will default to input_embed_dim
        """
        super().__init__()
        self.mha = MultiHeadedAttention(
            input_embed_dim,
            num_heads,
            head_dim=input_embed_dim
            if transformer_head_dim is None
            else transformer_head_dim,
            dropout=attn_dropout,
            keep_attn=keep_attn,
        )

        try:
            self.pos_wise_ff = GATED_UNITS[ff_activation](
                d_model=input_embed_dim,
                d_ff=input_embed_dim * ff_hidden_multiplier,
                dropout=ff_dropout,
            )
        except AttributeError:
            self.pos_wise_ff = PositionWiseFeedForward(
                d_model=input_embed_dim,
                d_ff=input_embed_dim * ff_hidden_multiplier,
                dropout=ff_dropout,
                activation=getattr(nn, self.hparams.ff_activation),
            )
        self.attn_add_norm = AddNorm(input_embed_dim, add_norm_dropout)
        self.ff_add_norm = AddNorm(input_embed_dim, add_norm_dropout)

    def forward(self, x):
        y = self.mha(x)
        x = self.attn_add_norm(x, y)
        y = self.pos_wise_ff(y)
        return self.ff_add_norm(x, y)


class AppendCLSToken(nn.Module):
    """Appends the [CLS] token for BERT-like inference."""

    def __init__(self, d_token: int, initialization: str) -> None:
        """Initialize self."""
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(d_token))
        d_sqrt_inv = 1 / math.sqrt(d_token)
        _initialize_kaiming(self.weight, initialization, d_sqrt_inv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform the forward pass."""
        assert x.ndim == 3
        return torch.cat([x, self.weight.view(1, 1, -1).repeat(len(x), 1, 1)], dim=1)

