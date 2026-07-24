"""Minimal forecast-state chain in target space (T3 -> T6 -> T12)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def upsample_forecast(forecast: torch.Tensor, target_len: int) -> torch.Tensor:
    """Linearly upsample [B, H, N, C] to [B, target_len, N, C]."""
    batch_size, horizon, num_nodes, channels = forecast.shape
    x = forecast.permute(0, 2, 3, 1).reshape(batch_size * num_nodes, channels, horizon)
    x = F.interpolate(x, size=target_len, mode="linear", align_corners=False)
    return x.reshape(batch_size, num_nodes, channels, target_len).permute(0, 3, 1, 2)


class ForecastBlock(nn.Module):
    """Simple target-space block: temporal mean pool + node embedding + MLP."""

    def __init__(
        self,
        input_dim: int,
        num_nodes: int,
        node_emb_dim: int,
        hidden_dim: int,
        horizon: int,
        output_dim: int = 1,
    ):
        super().__init__()
        self.horizon = horizon
        self.output_dim = output_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim + node_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, horizon * output_dim),
        )

    def forward(self, x: torch.Tensor, node_emb: torch.Tensor) -> torch.Tensor:
        # x: [B, T, N, C]
        batch_size, _, num_nodes, _ = x.shape
        pooled = x.mean(dim=1)
        node_feat = node_emb.unsqueeze(0).expand(batch_size, num_nodes, -1)
        h = torch.cat([pooled, node_feat], dim=-1)
        out = self.mlp(h)
        return out.view(batch_size, self.horizon, num_nodes, self.output_dim)


class ForecastSpaceLite(nn.Module):
    """Forecast-space chain: X -> Y3 -> Y6 -> Y12 with upsampled forecast conditioning."""

    def __init__(self, **model_args):
        super().__init__()
        self.seq_len = int(model_args["seq_len"])
        self.pred_len = int(model_args["pred_len"])
        self.num_nodes = int(model_args["num_nodes"])
        self.node_emb_dim = int(model_args.get("node_emb_dim", 16))
        self.hidden_dim = int(model_args.get("hidden_dim", 64))
        self.output_dim = int(model_args.get("output_dim", 1))
        horizons = model_args.get("chain_horizons", [3, 6, 12])
        self.h3, self.h6, self.h12 = int(horizons[0]), int(horizons[1]), int(horizons[2])

        self.node_emb = nn.Parameter(torch.randn(self.num_nodes, self.node_emb_dim) * 0.02)

        self.block1 = ForecastBlock(
            input_dim=1,
            num_nodes=self.num_nodes,
            node_emb_dim=self.node_emb_dim,
            hidden_dim=self.hidden_dim,
            horizon=self.h3,
            output_dim=self.output_dim,
        )
        self.block2 = ForecastBlock(
            input_dim=2,
            num_nodes=self.num_nodes,
            node_emb_dim=self.node_emb_dim,
            hidden_dim=self.hidden_dim,
            horizon=self.h6,
            output_dim=self.output_dim,
        )
        self.block3 = ForecastBlock(
            input_dim=2,
            num_nodes=self.num_nodes,
            node_emb_dim=self.node_emb_dim,
            hidden_dim=self.hidden_dim,
            horizon=self.h12,
            output_dim=self.output_dim,
        )
        self.final_proj = nn.Linear(self.output_dim, self.output_dim)

    def forward(
        self,
        history_data: torch.Tensor,
        future_data: torch.Tensor = None,
        batch_seen: int = 0,
        epoch: int = 0,
        train: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        del future_data, batch_seen, epoch, train, kwargs
        x_value = history_data[..., :1]

        y3 = self.block1(x_value, self.node_emb)
        y3_up = upsample_forecast(y3, self.seq_len)
        y6 = self.block2(torch.cat([x_value, y3_up], dim=-1), self.node_emb)

        y6_up = upsample_forecast(y6, self.seq_len)
        y12 = self.block3(torch.cat([x_value, y6_up], dim=-1), self.node_emb)
        y12 = self.final_proj(y12)
        return y12
