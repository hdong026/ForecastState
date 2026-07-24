"""Temporal projection and lifting for HyperD Forecast-State Chain.

Tensor contract: [B, H, N] (3D), as required by the HyperDChain guide.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def temporal_project(
    x: torch.Tensor,
    target_len: int,
) -> torch.Tensor:
    """
    Args:
        x: [B, H, N]
        target_len: target temporal resolution

    Returns:
        [B, target_len, N]
    """
    if x.ndim != 3:
        raise ValueError(
            f"Expected [B, H, N], got {tuple(x.shape)}."
        )

    batch_size, horizon, num_nodes = x.shape

    if target_len <= 0 or target_len > horizon:
        raise ValueError(
            f"target_len must be in [1, {horizon}], got {target_len}."
        )

    if target_len == horizon:
        return x

    if horizon % target_len == 0:
        block_size = horizon // target_len
        return x.reshape(
            batch_size,
            target_len,
            block_size,
            num_nodes,
        ).mean(dim=2)

    pooled = x.transpose(1, 2)
    pooled = F.adaptive_avg_pool1d(
        pooled,
        output_size=target_len,
    )
    return pooled.transpose(1, 2).contiguous()


def temporal_lift(
    x: torch.Tensor,
    full_horizon: int,
) -> torch.Tensor:
    """
    Args:
        x: [B, h, N]
        full_horizon: H

    Returns:
        [B, H, N]
    """
    if x.ndim != 3:
        raise ValueError(
            f"Expected [B, h, N], got {tuple(x.shape)}."
        )

    _, current_len, _ = x.shape

    if current_len <= 0 or current_len > full_horizon:
        raise ValueError(
            f"Cannot lift length {current_len} to {full_horizon}."
        )

    if current_len == full_horizon:
        return x

    if full_horizon % current_len == 0:
        repeat_factor = full_horizon // current_len
        return x.repeat_interleave(
            repeat_factor,
            dim=1,
        )

    interpolated = F.interpolate(
        x.transpose(1, 2),
        size=full_horizon,
        mode="linear",
        align_corners=False,
    )
    return interpolated.transpose(1, 2).contiguous()
