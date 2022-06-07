# noqa W605
from typing import Any, Dict, Optional, Tuple

import torch
from einops import rearrange
from torch import einsum, nn

from pytorch_tabular.models.common import activations


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


# https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/transformers/feed_forward.py
class PositionWiseFeedForward(nn.Module):
    """
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
    usually set to ReLU (Rectified Linear Unit) activation, $$\max(0, x)$$
    That is, the FFN function is,
    $$FFN(x, W_1, W_2, b_1, b_2) = \max(0, x W_1 + b_1) W_2 + b_2$$
    where $W_1$, $W_2$, $b_1$ and $b_2$ are learnable parameters.
    Sometimes the
    GELU (Gaussian Error Linear Unit) activation is also used instead of ReLU.
    $$x \Phi(x)$$ where $\Phi(x) = P(X \le x), X \sim \mathcal{N}(0,1)$
    ### Gated Linear Units
    This is a generic implementation that supports different variants including
    [Gated Linear Units](https://arxiv.org/abs/2002.05202) (GLU).
    We have also implemented experiments on these:
    * [experiment that uses `labml.configs`](glu_variants/experiment.html)
    * [simpler version from scratch](glu_variants/simple.html)
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
    """
    Applies LayerNorm, Dropout and adds to input. Standard AddNorm operations in Transformers
    """

    def __init__(self, input_dim: int, dropout: float):
        super(AddNorm, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(input_dim)

    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        return self.ln(self.dropout(Y) + X)


class MultiHeadedAttention(nn.Module):
    """
    Multi Headed Attention Block in Transformers
    """

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


# Slight adaptation from https://github.com/jrzaurin/pytorch-widedeep which in turn adapted from AutoGluon
class SharedEmbeddings(nn.Module):
    """
    Enables different values in a categorical feature to share some embeddings across
    """

    def __init__(
        self,
        num_embed: int,
        embed_dim: int,
        add_shared_embed: bool = False,
        frac_shared_embed: float = 0.25,
    ):
        super(SharedEmbeddings, self).__init__()
        assert frac_shared_embed < 1, "'frac_shared_embed' must be less than 1"

        self.add_shared_embed = add_shared_embed
        self.embed = nn.Embedding(num_embed, embed_dim, padding_idx=0)
        self.embed.weight.data.clamp_(-2, 2)
        if add_shared_embed:
            col_embed_dim = embed_dim
        else:
            col_embed_dim = int(embed_dim * frac_shared_embed)
        self.shared_embed = nn.Parameter(torch.empty(1, col_embed_dim).uniform_(-1, 1))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        out = self.embed(X)
        shared_embed = self.shared_embed.expand(out.shape[0], -1)
        if self.add_shared_embed:
            out += shared_embed
        else:
            out[:, : shared_embed.shape[1]] = shared_embed
        return out

    @property
    def weight(self):
        w = self.embed.weight.detach()
        if self.add_shared_embed:
            w += self.shared_embed
        else:
            w[:, : self.shared_embed.shape[1]] = self.shared_embed
        return w


class EmbeddingLayer(nn.Module):
    """
    Enables different values in a categorical feature to have different embeddings
    """

    def __init__(
        self,
        continuous_dim: int,
        categorical_embedding_dims: Tuple[int, int],
        embedding_dropout: float = 0.0,
        batch_norm_continuous_input: bool = False,
    ):
        super(EmbeddingLayer, self).__init__()
        self.continuous_dim = continuous_dim
        self.categorical_embedding_dims = categorical_embedding_dims
        self.batch_norm_continuous_input = batch_norm_continuous_input

        # Embedding layers
        self.cat_embedding_layers = nn.ModuleList(
            [nn.Embedding(x, y) for x, y in categorical_embedding_dims]
        )
        if embedding_dropout > 0:
            self.embd_dropout = nn.Dropout(embedding_dropout)
        else:
            self.embd_dropout = None
        # Continuous Layers
        if batch_norm_continuous_input:
            self.normalizing_batch_norm = nn.BatchNorm1d(continuous_dim)

    def forward(self, x: Dict[str, Any]) -> torch.Tensor:
        assert "continuous" in x and "categorical" in x, "x must contain continuous and categorical features"
        # (B, N)
        continuous_data, categorical_data = x["continuous"], x["categorical"]
        assert categorical_data.shape[1] == len(
            self.cat_embedding_layers
        ), "categorical_data must have same number of columns as categorical embedding layers"
        embed = None
        if continuous_data.shape[1] > 0:
            if self.batch_norm_continuous_input:
                embed = self.normalizing_batch_norm(continuous_data)
            else:
                embed = continuous_data
            # (B, N, C)
        if categorical_data.shape[1] > 0:
            categorical_embed = torch.cat(
                [
                    embedding_layer(categorical_data[:, i])
                    for i, embedding_layer in enumerate(self.cat_embedding_layers)
                ], dim=1
            )
        # (B, N, C + C)
            if embed is None:
                embed = categorical_embed
            else:
                embed = torch.cat([embed, categorical_embed], dim=1)
        if self.embd_dropout is not None:
            embed = self.embd_dropout(embed)
        return embed


class TransformerEncoderBlock(nn.Module):
    """A single Transformer Encoder Block"""

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
            self.pos_wise_ff = getattr(activations, ff_activation)(
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


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, *args, **kwargs):
        return self.func(*args, **kwargs)


class ModuleWithInit(nn.Module):
    """Base class for pytorch module with data-aware initializer on first batch"""

    def __init__(self):
        super().__init__()
        self._is_initialized_tensor = nn.Parameter(
            torch.tensor(0, dtype=torch.uint8), requires_grad=False
        )
        self._is_initialized_bool = None
        # Note: this module uses a separate flag self._is_initialized so as to achieve both
        # * persistence: is_initialized is saved alongside model in state_dict
        # * speed: model doesn't need to cache
        # please DO NOT use these flags in child modules

    def initialize(self, *args, **kwargs):
        """initialize module tensors using first batch of data"""
        raise NotImplementedError("Please implement ")

    def __call__(self, *args, **kwargs):
        if self._is_initialized_bool is None:
            self._is_initialized_bool = bool(self._is_initialized_tensor.item())
        if not self._is_initialized_bool:
            self.initialize(*args, **kwargs)
            self._is_initialized_tensor.data[...] = 1
            self._is_initialized_bool = True
        return super().__call__(*args, **kwargs)
