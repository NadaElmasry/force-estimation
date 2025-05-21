# -----------------------------------------------------------------------------
#  recurrency_block_revised.py  ·  May 2025
#  Clean re‑implementation of RecurrencyBlock used by ForceEstimator.
# -----------------------------------------------------------------------------
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional


class RecurrencyBlock(nn.Module):
    """Fuse per‑frame features (and optional robot‑state) with a 2‑layer LSTM.

    Expected shapes
    ---------------
    features : (B, T, F)
        Output of the image (or state) embed projection; F ≡ embed_dim.
    state    : (B, T, S) or *None*
        Raw robot‑state vector per time‑step. Will be linearly projected to F and
        concatenated along the feature axis. If *None*, a zeros‑tensor is used.

    Output
    ------
    force    : (B, 3)
        Predicted 3‑D wrench for the **last** time‑step T‑1. (Change to mean or
        full‑sequence regression if needed.)
    """

    def __init__(
        self,
        *,
        embed_dim: int = 512,
        hidden_size: Optional[int] = None,
        num_layers: int = 2,
        state_size: int = 0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size or embed_dim  # sensible default
        self.num_layers = num_layers
        self.state_size = state_size

        if state_size > 0:
            self.state2embed = nn.Linear(state_size, embed_dim)
        else:
            # register a dummy parameter so .to(device) still works
            self.register_buffer("_dummy_state", torch.empty(0))

        # LSTM takes concatenated [feat, state_emb] → dim = 2*embed_dim
        self.lstm = nn.LSTM(
            input_size=embed_dim * 2,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(self.hidden_size, 3)

    # ------------------------------------------------------------------  forward
    def forward(
        self,
        features: torch.Tensor,  # (B, T, F)
        state: Optional[torch.Tensor] = None,  # (B, T, S) or None
    ) -> torch.Tensor:
        B, T, F = features.shape
        assert F == self.embed_dim, "feature dim mismatch"

        # ----- state processing ------------------------------------------------
        if self.state_size > 0 and state is not None:
            B2, T_state, S = state.shape
            assert B2 == B, "batch mismatch between features and state"
            assert S == self.state_size, "state dim mismatch"
            if T_state != T:
                # simple time‑axis alignment: pad or truncate
                if T_state < T:
                    pad = state.new_zeros(B, T - T_state, S)
                    state = torch.cat([state, pad], dim=1)
                else:
                    state = state[:, :T, :]
            state_emb = self.state2embed(state.view(B * T, S)).view(B, T, F)
        else:
            # no state provided → zeros
            state_emb = features.new_zeros(B, T, F)

        # ----- LSTM ------------------------------------------------------------
        lstm_in = torch.cat([features, state_emb], dim=2)   # (B, T, 2F)
        h0 = torch.zeros(self.num_layers, B, self.hidden_size, device=features.device)
        c0 = torch.zeros_like(h0)
        out, _ = self.lstm(lstm_in, (h0, c0))               # (B, T, H)

        # take the last time‑step
        last = out[:, -1, :]                                # (B, H)
        return self.fc(last)                                # (B, 3)
