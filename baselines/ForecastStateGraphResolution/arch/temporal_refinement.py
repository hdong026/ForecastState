"""Temporal helpers: initial forecast and optional KASA reuse notes.

Paper allows Ŷ^(0)=0; we also support last_value / linear_head for stability.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class InitialForecast(nn.Module):
    def __init__(
        self,
        mode: str,
        input_len: int,
        pred_len: int,
        num_nodes: int,
        output_dim: int = 1,
    ):
        super().__init__()
        self.mode = str(mode).lower()
        self.input_len = int(input_len)
        self.pred_len = int(pred_len)
        self.num_nodes = int(num_nodes)
        self.output_dim = int(output_dim)
        if self.mode not in {"zero", "last_value", "linear_head"}:
            raise ValueError(f"Unsupported initial_forecast: {self.mode}")
        if self.mode == "linear_head":
            self.head = nn.Conv2d(self.input_len, self.pred_len, kernel_size=1)
            nn.init.xavier_uniform_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def forward(self, history_data: torch.Tensor) -> torch.Tensor:
        """history_data: [B,P,N,C] -> Y0 [B,H,N,C_y]."""
        b, p, n, _ = history_data.shape
        flow = history_data[..., : self.output_dim]
        if self.mode == "zero":
            return torch.zeros(b, self.pred_len, n, self.output_dim, device=history_data.device, dtype=history_data.dtype)
        if self.mode == "last_value":
            last = flow[:, -1:, :, :]
            return last.expand(-1, self.pred_len, -1, -1).contiguous()
        # linear_head on channel-0..output_dim-1
        # Conv2d over time: input [B, P, N, C] -> permute to [B, P, N*C] messy; do per-channel
        outs = []
        for c in range(self.output_dim):
            x = flow[..., c : c + 1].permute(0, 1, 2, 3)  # B P N 1
            y = self.head(x)  # B H N 1
            outs.append(y)
        return torch.cat(outs, dim=-1)
