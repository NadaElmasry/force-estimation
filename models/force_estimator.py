# import torch
# import torch.nn as nn
# from models.resnet import ResnetEncoder
# from models.transformer import ViT
# from models.recurrency import RecurrencyBlock
# from models.utils import FcBlock
# from einops import repeat

# class ForceEstimator(nn.Module):

#     def __init__(self, architecture: str, recurrency: bool, pretrained: bool,
#                  att_type: str = None, embed_dim: int = 512,
#                  state_size: int = 0) -> None:
#         super(ForceEstimator, self).__init__()

#         assert architecture in ["cnn", "vit", "fc"], "The resnet encoder must be either a cnn or a vision transformer"
        
#         self.architecture = architecture
#         self.recurrency = recurrency

#         if self.architecture != "fc":
#             if self.architecture == "cnn":
#                 encoder = ResnetEncoder(num_layers=50,
#                                             pretrained=pretrained,
#                                             att_type=att_type)
                
#                 self.encoder = nn.Sequential(
#                     encoder.encoder.conv1,
#                     encoder.encoder.bn1,
#                     encoder.encoder.relu,
#                     encoder.encoder.maxpool,
#                     encoder.encoder.layer1,
#                     encoder.encoder.layer2,
#                     encoder.encoder.layer3,
#                     encoder.encoder.layer4[:2],
#                     encoder.encoder.layer4[2].conv1,
#                     encoder.encoder.layer4[2].bn1,
#                     encoder.encoder.layer4[2].conv2,
#                     encoder.encoder.layer4[2].bn2,
#                     encoder.encoder.layer4[2].conv3
#                 )

#                 self.splitted = nn.Sequential(
#                     encoder.encoder.layer4[2].bn3,
#                     encoder.encoder.layer4[2].relu,
#                     # encoder.encoder.avgpool,
#                     # encoder.encoder.fc
#                 )

#             elif self.architecture == "vit":
#                 encoder = ViT(image_size=256,
#                                 patch_size=16,
#                                 dim=1024,
#                                 depth=6,
#                                 heads=16,
#                                 mlp_dim=2048,
#                                 dropout=0.1,
#                                 emb_dropout=0.1,
#                                 channels=3,
#                                 )
                
#                 self.cls_token = nn.Parameter(torch.randn(1, 1, 1024))
#                 self.embeding = encoder.to_patch_embedding
#                 self.pos_embed = encoder.pos_embedding
#                 self.dropout = encoder.dropout

#                 self.encoder = nn.Sequential(
#                     encoder.transformer.layers[:5]
#                 )

#                 self.last_layer = nn.Sequential(
#                     encoder.transformer.layers[5]
#                 )

#                 self.splitted = nn.Sequential(
#                     encoder.to_latent,
#                     encoder.mlp_head
#                 )
                
#             final_ch = 512 if self.architecture=="vit" else (2048 * 8 * 8)
            
#             if not self.recurrency:
#                 self.embed_dim = embed_dim + state_size
#             else:
#                 self.embed_dim = embed_dim

#             self.embed_block = FcBlock(final_ch, embed_dim)
            
#             if recurrency:
#                 self.recurrency = RecurrencyBlock(embed_dim=self.embed_dim)
#             else:
#                 if state_size != 0:
#                     self.final = nn.Sequential(
#                         FcBlock(self.embed_dim, 84),
#                         FcBlock(84, 180),
#                         FcBlock(180, 50),
#                         nn.Linear(50, 3)
#                     )
#                 else:
#                     self.final = nn.Linear(self.embed_dim, 3)
                
#         else:
#             self.encoder = nn.Sequential(
#                 FcBlock(state_size, 500),
#                 FcBlock(500, 1000),
#                 FcBlock(1000, 1000),
#                 FcBlock(1000, 500),
#                 FcBlock(500, 50),
#                 nn.Linear(50, 3)
#             )
    
#     def forward(self, x, rs = None) -> torch.Tensor:

#         if self.architecture != "fc":
#             if self.recurrency:
#                 batch_size = x[0].shape[0]
#                 rec_size = len(x)

#                 features = torch.zeros(batch_size, rec_size, self.embed_dim).cuda().float()

#                 for i in range(batch_size):
#                     inp = torch.cat([img[i].unsqueeze(0) for img in x], dim=0)
#                     if self.architecture == "vit":
#                         inp = self.embeding(inp)
#                         b, n, _ = inp.shape

#                         cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
#                         inp = torch.cat((cls_tokens, inp), dim=1)
#                         inp += self.pos_embed[:, :(n + 1)]
#                         inp = self.dropout(inp)

#                     out = self.encoder(inp)
#                     # register a hook
#                     if out.requires_grad:
#                         h = out.register_hook(self.activations_hook)
#                     if self.architecture == "vit":
#                         out = self.last_layer(out)
#                         out = out[:, 0]
                    
#                     out = self.splitted(out)
#                     out = out.view(rec_size, -1)
#                     features[i] = self.embed_block(out)
                
#                 pred = self.recurrency(features, rs)
            
#             else:

#                 if self.architecture == "vit":
#                     x = self.embeding(x)
#                     b, n, _ = x.shape

#                     cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
#                     x = torch.cat((cls_tokens, x), dim=1)
#                     x += self.pos_embed[:, :(n + 1)]
#                     x = self.dropout(x)

#                 out = self.encoder(x)
                
#                 # register a hook
#                 if out.requires_grad:
#                     h = out.register_hook(self.activations_hook)
#                 if self.architecture == "vit":
#                     out = self.last_layer(out)
#                     out = out[:, 0]
                    
#                 out = self.splitted(out)
#                 out_flatten = out.view(out.shape[0], -1)
#                 out = self.embed_block(out_flatten)
                
#                 if rs is not None:
#                     out = torch.cat([out, rs], dim=1)
                
#                 pred = self.final(out)
                
#         else:
#             pred = self.encoder(x)

#         return pred
    

#     def activations_hook(self, grad):
#         self.gradients = grad
    

#     def get_activations_gradient(self):
#         return self.gradients
    

#     def get_activations(self, x):

#         if self.recurrency:
#             batch_size = x[0].shape[0]

#             for i in range(batch_size):
#                 inp = torch.cat([img[i].unsqueeze(0) for img in x], dim=0)
#                 if self.architecture == "vit":
#                     inp = self.embeding(inp)
#                     b, n, _ = inp.shape

#                     cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
#                     inp = torch.cat((cls_tokens, inp), dim=1)
#                     inp += self.pos_embed[:, :(n + 1)]
#                     inp = self.dropout(inp)

#                 inp = self.encoder(inp)

#                 if i == 0:
#                     if self.architecture == "vit":
#                         out = torch.zeros(batch_size, inp.shape[0], inp.shape[1], inp.shape[2]).cuda().float()
#                     else:
#                         out = torch.zeros(batch_size, inp.shape[0], inp.shape[1], inp.shape[2], inp.shape[3]).cuda().float()
#                 out[i] = inp
            
#             out = torch.mean(out, dim=1)
        
#         else:
#             if self.architecture == "vit":
#                 x = self.embeding(x)
#                 b, n, _ = x.shape

#                 cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
#                 x = torch.cat((cls_tokens, x), dim=1)
#                 x += self.pos_embed[:, :(n + 1)]
#                 x = self.dropout(x)
            
#             out = self.encoder(x)

#         return out



# -----------------------------------------------------------------------------
#  force_estimator_revised.py  ·  May 2025
#  Clean, shape‑safe implementation of ForceEstimator supporting three modes:
#    • vision‑only  (cnn | vit)
#    • vision + state (cnn | vit)
#    • state‑only    (fc)
#  Optional LSTM recurrency over a sequence of length T (>=1).
# -----------------------------------------------------------------------------
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange, repeat

from models.resnet import ResnetEncoder
from models.transformer import ViT
from models.recurrency import RecurrencyBlock  # expects (B, T, F) [+ state]
from models.utils import FcBlock

# -----------------------------------------------------------------------------
class ForceEstimator(nn.Module):
    """Flexible force‑regression backbone.

    Parameters
    ----------
    architecture : {'cnn', 'vit', 'fc'}
        cnn / vit  → image encoder  ·  fc → state‑only branch.
    recurrency   : bool
        If *True*, adds an LSTM block that consumes a sequence of length `seq_length`.
    pretrained   : bool
        Passed to ResNet encoder (ignored for ViT).
    att_type     : str | None
        Optional attention plug‑in for ResNet.
    embed_dim    : int
        Size of per‑frame feature after the embed_block.
    state_size   : int
        Dimensionality of robot‑state vector. 0 ⇒ no state input.
    seq_length   : int ≥ 1
        Temporal window. 1 ⇒ no real sequence (recurrency should be False).
    """

    def __init__(
        self,
        architecture: str,
        recurrency: bool,
        pretrained: bool = False,
        *,
        att_type: Optional[str] = None,
        embed_dim: int = 512,
        state_size: int = 0,
        seq_length: int = 1,
    ) -> None:
        super().__init__()
        assert architecture in {"cnn", "vit", "fc"}, "architecture must be cnn, vit or fc"
        self.architecture = architecture
        self.recurrency = recurrency
        self.seq_length = seq_length
        self.state_size = state_size
        self.embed_dim = embed_dim  # per‑frame latent dim fed into LSTM or FC head

        # -------------------------------------------------  IMAGE ENCODERS  ----
        if architecture == "cnn":
            # ResNet‑50 backbone (truncated)
            resnet = ResnetEncoder(num_layers=50, pretrained=pretrained, att_type=att_type)
            enc = resnet.encoder  # alias
            self.cnn_stem = nn.Sequential(
                enc.conv1, enc.bn1, enc.relu, enc.maxpool,
                enc.layer1, enc.layer2, enc.layer3, *enc.layer4  # full Layer4
            )
            self.cnn_pool = nn.AdaptiveAvgPool2d((1, 1))
            in_ch = 2048

        elif architecture == "vit":
            self.vit = ViT(
                image_size=256,
                patch_size=16,
                dim=1024,
                depth=6,
                heads=16,
                mlp_dim=2048,
                dropout=0.1,
                emb_dropout=0.1,
                channels=3,
            )
            in_ch = 1024  # CLS token dim

            # helper buffer so we can call vit.to_patch_embedding directly
            self.register_buffer("cls_token", self.vit.cls_token.clone())

        else:  # state‑only branch (fc)
            in_ch = state_size  # directly the state vector

        # -------------------------------------------------  COMMON PROJECTION  --
        if architecture != "fc":
            # project per‑frame image feature → embed_dim
            self.img_proj = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_ch, embed_dim),
                nn.ReLU(inplace=True),
            )

        # -------------------------------------------------  TEMPORAL / FINAL  ---
        if recurrency:
            self.lstm = RecurrencyBlock(embed_dim=embed_dim, hidden_size=embed_dim)
            # output of RecurrencyBlock is (B, 3) already
            # (the class must implement its own regression head)
        else:
            # no LSTM → concatenate state (if any) and regress directly
            head_in = embed_dim + state_size
            self.head = nn.Sequential(
                FcBlock(head_in, 256),
                FcBlock(256, 64),
                nn.Linear(64, 3),
            )

        # -------------------------------------------------  STATE‑ONLY PATH  ----
        if architecture == "fc":
            if recurrency:
                self.state2embed = nn.Linear(state_size, embed_dim)
            else:
                self.fc_encoder = nn.Sequential(
                    FcBlock(state_size, 500),
                    FcBlock(500, 1000),
                    FcBlock(1000, 500),
                    FcBlock(500, 64),
                    nn.Linear(64, 3),
                )

    # =====================================================================  FORWARD  ==
    def forward(self, img: torch.Tensor, state: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        img   :
            • cnn / vit  :  (B, C, H, W) **or** (B, T, C, H, W)
            • fc branch  :  ignored → pass `img=None`
        state :
            Robot‑state tensor or *None*.
            Shape (B, S) for non‑recurrent, (B, T, S) for recurrent.
        """
        if self.architecture == "fc":
            return self._forward_state_only(state)

        # ---------------  Gather shapes  ---------------
        if img.dim() == 5:  # (B, T, C, H, W)
            B, T, C, H, W = img.shape
            assert T == self.seq_length, "img sequence length mismatch"
            img = rearrange(img, "b t c h w -> (b t) c h w")  # flatten time
        else:  # (B, C, H, W)
            B, C, H, W = img.shape
            T = 1
            assert self.seq_length == 1, "Provided single frame but seq_length > 1"

        # ---------------  Per‑frame feature extraction  ---------------
        if self.architecture == "cnn":
            feats = self.cnn_pool(self.cnn_stem(img))  # (B*T, 2048, 1, 1)
            feats = feats.view(feats.size(0), -1)      # (B*T, 2048)
        else:  # vit
            vit_out = self.vit.to_patch_embedding(img)   # (B*T, N, 1024)
            cls = repeat(self.cls_token, "1 1 d -> b 1 d", b=vit_out.size(0))
            vit_in = torch.cat((cls, vit_out), dim=1)
            vit_out = self.vit.transformer(vit_in)       # (B*T, N+1, 1024)
            feats = vit_out[:, 0]                        # CLS token

        feats = self.img_proj(feats)  # (B*T, embed_dim)

        if self.recurrency:
            feats = feats.view(B, T, -1)  # (B, T, embed_dim)
            if state is not None:
                state = state.view(B, T, -1)
            pred = self.lstm(feats, state)  # RecurrencyBlock handles state concat
        else:
            if T > 1:
                # no recurrency ⇒ treat each frame as separate sample ⇒ average predictions
                feats = feats.view(B, T, -1).mean(dim=1)  # simple aggregation
                if state is not None:
                    state = state.mean(dim=1)
            if state is not None:
                feats = torch.cat([feats, state], dim=-1)
            pred = self.head(feats)

        return pred

    # ------------------------------------------------------------------  STATE‑ONLY  --
    def _forward_state_only(self, state: torch.Tensor) -> torch.Tensor:
        if not self.recurrency:
            if state.dim() == 3:  # (B, T, S) but no LSTM → average over time
                state = state.mean(dim=1)
            return self.fc_encoder(state)

        # recurrency=True
        if state.dim() == 2:
            # expand singular state along fake time axis
            state = state.unsqueeze(1)  # (B, 1, S)
        B, T, S = state.shape
        state_flat = state.view(B * T, S)
        em = self.state2embed(state_flat).view(B, T, -1)  # (B, T, embed_dim)
        return self.lstm(em, None)
