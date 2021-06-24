from typing import Optional
import torch
import torch.nn.functional as F
from torch import nn, einsum
from pytorch_tabular.models import common #import PositionWiseFeedForward, GEGLU, ReGLU, SwiGLU
from einops import rearrange


class AddNorm(nn.Module):
    def __init__(self, input_dim: int, dropout: float):
        super(AddNorm, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(input_dim)

    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        return self.ln(self.dropout(Y) + X)


class MultiHeadedAttention(nn.Module):
    def __init__(
        self, input_dim: int, num_heads: int = 8, head_dim: int = 16, dropout: int = 0.1
    ):
        super().__init__()
        assert input_dim % num_heads == 0, "'input_dim' must be multiples of 'num_heads'"
        inner_dim = head_dim * num_heads
        self.n_heads = num_heads
        self.scale = head_dim ** -0.5

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

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)", h=h)
        return self.to_out(out)

#Shamelessly copied with slight adaptation from https://github.com/jrzaurin/pytorch-widedeep/blob/b487b06721c5abe56ac68c8a38580b95e0897fd4/pytorch_widedeep/models/tab_transformer.py
class SharedEmbeddings(nn.Module):
    def __init__(
        self,
        num_embed: int,
        embed_dim: int,
        add_shared_embed: bool = False,
        frac_shared_embed: float=0.25,
    ):
        super(SharedEmbeddings, self).__init__()
        assert (
            frac_shared_embed < 1
        ), "'frac_shared_embed' must be less than 1"

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

class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        input_embed_dim: int,
        num_heads: int = 8,
        ff_hidden_multiplier: int = 4,
        ff_activation: str = "GEGLU",
        attn_dropout: float = 0.1,
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
        )
        
        try:
            self.pos_wise_ff = getattr(common, ff_activation)(d_model=input_embed_dim,
            d_ff=input_embed_dim * ff_hidden_multiplier,
            dropout=ff_dropout)
        except AttributeError:
            self.pos_wise_ff = getattr(common, "PositionWiseFeedForward")(
                d_model=input_embed_dim,
                d_ff=input_embed_dim * ff_hidden_multiplier,
                dropout=ff_dropout,
                activation = getattr(nn, self.hparams.ff_activation)
            )
        self.attn_add_norm = AddNorm(input_embed_dim, add_norm_dropout)
        self.ff_add_norm = AddNorm(input_embed_dim, add_norm_dropout)

    def forward(self, x):
        y = self.mha(x)
        x = self.attn_add_norm(x, y)
        y = self.pos_wise_ff(y)
        return self.ff_add_norm(x, y)


# class MLP(nn.Module):
#     def __init__(self, dims, act=None):
#         super().__init__()
#         dims_pairs = list(zip(dims[:-1], dims[1:]))
#         layers = []
#         for ind, (dim_in, dim_out) in enumerate(dims_pairs):
#             is_last = ind >= (len(dims) - 1)
#             linear = nn.Linear(dim_in, dim_out)
#             layers.append(linear)

#             if is_last:
#                 continue

#             act = default(act, nn.ReLU())
#             layers.append(act)

#         self.mlp = nn.Sequential(*layers)

#     def forward(self, x):
#         return self.mlp(x)
