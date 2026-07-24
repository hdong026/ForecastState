"""Temporal projection and lifting operators (document Parts 3–4)."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def temporal_project(
    y: torch.Tensor,
    target_length: int,
) -> torch.Tensor:
    """Project [B, H, N, C] to [B, target_length, N, C]."""
    if y.ndim != 4:
        raise ValueError(
            f"Expected [B, H, N, C], got {tuple(y.shape)}."
        )

    batch_size, horizon, num_nodes, channels = y.shape

    if not 1 <= target_length <= horizon:
        raise ValueError(
            f"target_length must be in [1, {horizon}], got {target_length}."
        )

    if target_length == horizon:
        return y

    if horizon % target_length == 0:
        block_size = horizon // target_length
        return y.reshape(
            batch_size,
            target_length,
            block_size,
            num_nodes,
            channels,
        ).mean(dim=2)

    y_pool = y.permute(0, 2, 3, 1).reshape(
        batch_size * num_nodes,
        channels,
        horizon,
    )
    y_pool = F.adaptive_avg_pool1d(y_pool, target_length)

    return y_pool.reshape(
        batch_size,
        num_nodes,
        channels,
        target_length,
    ).permute(0, 3, 1, 2).contiguous()


def temporal_lift(
    z: torch.Tensor,
    full_horizon: int,
) -> torch.Tensor:
    """Lift [B, h, N, C] to [B, H, N, C]."""
    if z.ndim != 4:
        raise ValueError(
            f"Expected [B, h, N, C], got {tuple(z.shape)}."
        )

    batch_size, current_length, num_nodes, channels = z.shape

    if current_length > full_horizon:
        raise ValueError(
            f"Cannot lift length {current_length} to {full_horizon}."
        )

    if current_length == full_horizon:
        return z

    if full_horizon % current_length == 0:
        repeat_factor = full_horizon // current_length
        return z.repeat_interleave(repeat_factor, dim=1)

    z_interp = z.permute(0, 2, 3, 1).reshape(
        batch_size * num_nodes,
        channels,
        current_length,
    )
    z_interp = F.interpolate(
        z_interp,
        size=full_horizon,
        mode="linear",
        align_corners=False,
    )

    return z_interp.reshape(
        batch_size,
        num_nodes,
        channels,
        full_horizon,
    ).permute(0, 3, 1, 2).contiguous()
