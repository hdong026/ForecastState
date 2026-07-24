"""Stage residual predictors F_s operating in resolution space Y_ωs.

Paper: Eq. (24)–(27). Propagated object is forecast residual, not latent state.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph_operator import StageGraphOperator


class SharedHistoryEncoder(nn.Module):
    """Light shared temporal encoder over history [B,P,N,C_in] -> [B,P,N,D]."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        self.net = nn.Sequential(*layers)
        self.out_dim = hidden_dim

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        # history: B P N C
        return self.net(history)


class StageResidualPredictor(nn.Module):
    """F_s(X, Z_bar; A_s) -> R_s in [B, h, M, C_y]."""

    def __init__(
        self,
        h: int,
        num_regions: int,
        history_len: int,
        history_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 1,
        dropout: float = 0.1,
        use_graph: bool = True,
        zero_init_head: bool = True,
    ):
        super().__init__()
        self.h = int(h)
        self.num_regions = int(num_regions)
        self.history_len = int(history_len)
        self.output_dim = int(output_dim)
        self.use_graph = bool(use_graph)

        # project history temporal axis to h and mix channels
        self.hist_proj = nn.Sequential(
            nn.Linear(history_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.temp_mix = nn.Sequential(
            nn.Conv2d(hidden_dim + output_dim, hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.ReLU(),
        )
        self.graph_ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.head = nn.Conv2d(hidden_dim, output_dim, kernel_size=1)
        if zero_init_head:
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def _align_history(self, hist_enc: torch.Tensor) -> torch.Tensor:
        """[B,P,M,D] -> [B,h,M,D] via adaptive pool on time."""
        b, p, m, d = hist_enc.shape
        if p == self.h:
            return hist_enc
        x = hist_enc.permute(0, 2, 3, 1).reshape(b * m, d, p)
        x = F.adaptive_avg_pool1d(x, self.h)
        return x.reshape(b, m, d, self.h).permute(0, 3, 1, 2)

    def forward(
        self,
        hist_region: torch.Tensor,
        z_bar: torch.Tensor,
        graph_op: StageGraphOperator | None,
    ) -> torch.Tensor:
        """
        hist_region: [B,P,M,D_hist]
        z_bar: [B,h,M,C_y]
        """
        h_feat = self.hist_proj(hist_region)
        h_feat = self._align_history(h_feat)
        x = torch.cat([h_feat, z_bar], dim=-1)  # B h M D+Cy
        x = self.temp_mix(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)  # B h M H
        if self.use_graph and graph_op is not None:
            x_g = graph_op.propagate(x)
            x = x + self.graph_ff(x_g)
        residual = self.head(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return residual
