# import torch
# from torch import nn
# from einops import rearrange, repeat
# from einops.layers.torch import Rearrange

# # helpers

# def pair(t):
#     return t if isinstance(t, tuple) else (t, t)

# # classes

# class PreNorm(nn.Module):
#     def __init__(self, dim, fn):
#         super().__init__()
#         self.norm = nn.LayerNorm(dim)
#         self.fn = fn
#     def forward(self, x, **kwargs):
#         return self.fn(self.norm(x), **kwargs)

# class FeedForward(nn.Module):
#     def __init__(self, dim, hidden_dim, dropout = 0.):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(dim, hidden_dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, dim),
#             nn.Dropout(dropout)
#         )
#     def forward(self, x):
#         return self.net(x)

# class Attention(nn.Module):
#     def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
#         super().__init__()
#         inner_dim = dim_head *  heads
#         project_out = not (heads == 1 and dim_head == dim)

#         self.heads = heads
#         self.scale = dim_head ** -0.5

#         self.attend = nn.Softmax(dim = -1)
#         self.dropout = nn.Dropout(dropout)

#         self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

#         self.to_out = nn.Sequential(
#             nn.Linear(inner_dim, dim),
#             nn.Dropout(dropout)
#         ) if project_out else nn.Identity()

#     def forward(self, x):
#         qkv = self.to_qkv(x).chunk(3, dim = -1)
#         q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

#         dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

#         attn = self.attend(dots)
#         attn = self.dropout(attn)

#         out = torch.matmul(attn, v)
#         out = rearrange(out, 'b h n d -> b n (h d)')
#         return self.to_out(out)

# class Transformer(nn.Module):
#     def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
#         super().__init__()
#         self.layers = []
#         for _ in range(depth):
#             self.layers.append(nn.Sequential(
#                 PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
#                 PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
#             ))
            
#         self.layers = nn.Sequential(*self.layers)
        
#     def forward(self, x):
#         for attn, ff in self.layers:
#             x = attn(x) + x
#             x = ff(x) + x
#         return x


# class ViT(nn.Module):
#     def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., state_include= False):
#         super().__init__()
#         image_height, image_width = pair(image_size)
#         patch_height, patch_width = pair(patch_size)

#         assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

#         num_patches = (image_height // patch_height) * (image_width // patch_width)
#         patch_dim = channels * patch_height * patch_width
#         assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

#         self.to_patch_embedding = nn.Sequential(
#             Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
#             nn.LayerNorm(patch_dim),
#             nn.Linear(patch_dim, dim),
#             nn.LayerNorm(dim),
#         )

#         self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
#         self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
#         self.dropout = nn.Dropout(emb_dropout)

#         self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

#         self.pool = pool
#         self.to_latent = nn.Identity()
        
    
#         self.mlp_head = nn.Sequential(
#             nn.LayerNorm(dim),
#             nn.Linear(dim, 512)
#         )

#         # self.apply(self._init_weights)

#     def forward(self, img, var = None):
#         x = self.to_patch_embedding(img)
#         b, n, _ = x.shape

#         cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
#         x = torch.cat((cls_tokens, x), dim=1)
#         x += self.pos_embedding[:, :(n + 1)]
#         x = self.dropout(x)

#         x = self.transformer(x)

#         x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

#         x = self.to_latent(x)

#         return self.mlp_head(x)
    
#     def _init_weights(self, module):
#         """Initialize the weights

#         Args:
#             module (_type_): _description_
#         """
#         if isinstance(module, (nn.Linear, nn.Embedding)):
#             module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
#         elif isinstance(module, nn.LayerNorm):
#             module.bias.data.zero_()
#             module.weight.data.fill_(1.0)
        
#         if isinstance(module, nn.Linear) and module.bias is not None:
#             module.bias.data.zero_()





# -----------------------------------------------------------------------------
#  vision_transformer_revised.py  ·  May 2025
#  Minimal, shape‑safe ViT compatible with the revised ForceEstimator.
# -----------------------------------------------------------------------------
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


# -----------------------------------------------------------------------------
#  Helper utils
# -----------------------------------------------------------------------------

def pair(x) -> Tuple[int, int]:
    """Ensure *x* is a pair (h, w)."""
    return x if isinstance(x, tuple) else (x, x)


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:  # noqa: D401,E501
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401,E501
        return self.net(x)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401,E501
        B, N, _ = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # each is (B, N, inner_dim)
        q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=self.heads) for t in qkv)

        dots = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, h, N, N)
        attn = self.softmax(dots)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)  # (B, h, N, d)
        out = rearrange(out, "b h n d -> b n (h d)")  # (B, N, inner_dim)
        return self.proj(out)


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.attn = PreNorm(dim, Attention(dim, heads, dim_head, dropout))
        self.ff = PreNorm(dim, FeedForward(dim, mlp_dim, dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401,E501
        x = self.attn(x) + x
        x = self.ff(x) + x
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(dim, heads, dim_head, mlp_dim, dropout)
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401,E501
        for layer in self.layers:
            x = layer(x)
        return x


# -----------------------------------------------------------------------------
#  Vision Transformer (ViT)
# -----------------------------------------------------------------------------
class ViT(nn.Module):
    def __init__(
        self,
        *,
        image_size: int | tuple,
        patch_size: int | tuple,
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        pool: str = "cls",
        channels: int = 3,
        dim_head: int = 64,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert pool in {"cls", "mean"}, "pool must be 'cls' or 'mean'"

        ih, iw = pair(image_size)
        ph, pw = pair(patch_size)
        assert ih % ph == 0 and iw % pw == 0, "image dims must be multiples of patch size"

        num_patches = (ih // ph) * (iw // pw)
        patch_dim = channels * ph * pw

        # Patch → token embedding
        self.to_patch = Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=ph, p2=pw)
        self.patch_emb = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        # Positional + CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.emb_dropout = nn.Dropout(emb_dropout)

        # Transformer encoder
        self.transformer = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
        )

        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))

        self._init_weights()

    # ------------------------------------------------------------------  forward
    def forward(self, img: torch.Tensor) -> torch.Tensor:  # noqa: D401,E501
        # img: (B, C, H, W)
        B = img.size(0)
        x = self.to_patch(img)            # (B, N, patch_dim)
        x = self.patch_emb(x)             # (B, N, dim)

        cls = repeat(self.cls_token, "1 1 d -> b 1 d", b=B)
        x = torch.cat((cls, x), dim=1)    # (B, N+1, dim)
        x = x + self.pos_embed[:, : x.size(1)]
        x = self.emb_dropout(x)

        x = self.transformer(x)           # (B, N+1, dim)
        x = x[:, 0] if self.pool == "cls" else x.mean(dim=1)  # (B, dim)
        x = self.to_latent(x)
        return self.mlp_head(x)

    # ------------------------------------------------------------------  init  --
    def _init_weights(self) -> None:
        """Xavier init for linear layers; normal init for positional encodings."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)
