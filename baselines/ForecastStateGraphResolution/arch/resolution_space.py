"""Temporal / graph / unified resolution projection and lifting.

Paper: Eq. (4)–(9), (43). Tensor convention: [B, T, N, C].
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def temporal_project(y: torch.Tensor, target_h: int) -> torch.Tensor:
    """D_h: [B, H, N, C] -> [B, h, N, C]. Non-overlapping mean or adaptive pool."""
    if y.dim() != 4:
        raise ValueError(f"Expected [B,T,N,C], got {tuple(y.shape)}")
    b, full_h, n, c = y.shape
    target_h = int(target_h)
    if target_h <= 0:
        raise ValueError(f"target_h must be positive, got {target_h}")
    if target_h == full_h:
        return y
    if full_h % target_h == 0:
        group = full_h // target_h
        return y.reshape(b, target_h, group, n, c).mean(dim=2)
    x = y.permute(0, 2, 3, 1).reshape(b * n, c, full_h)
    x = F.adaptive_avg_pool1d(x, target_h)
    return x.reshape(b, n, c, target_h).permute(0, 3, 1, 2)


def temporal_lift(
    z: torch.Tensor,
    full_h: int,
    mode: str = "linear",
) -> torch.Tensor:
    """U_h: [B, h, N, C] -> [B, H, N, C]. Identity when h == H."""
    if z.dim() != 4:
        raise ValueError(f"Expected [B,T,N,C], got {tuple(z.shape)}")
    b, cur_h, n, c = z.shape
    full_h = int(full_h)
    if cur_h == full_h:
        return z
    mode = str(mode).lower()
    if mode == "block_repeat":
        if full_h % cur_h != 0:
            # fall back to linear if not divisible
            mode = "linear"
        else:
            rep = full_h // cur_h
            return z.repeat_interleave(rep, dim=1)
    if mode != "linear":
        raise ValueError(f"Unsupported temporal lift mode: {mode}")
    x = z.permute(0, 2, 3, 1).reshape(b * n, c, cur_h)
    x = F.interpolate(x, size=full_h, mode="linear", align_corners=False)
    return x.reshape(b, n, c, full_h).permute(0, 3, 1, 2)


def graph_project(y: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """P_s Y: [B,T,N,C] x [M,N] -> [B,T,M,C] (cluster mean)."""
    # y: B T N C, p: M N
    return torch.einsum("mn,btnc->btmc", p, y)


def graph_lift(z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """C_s Z: [B,T,M,C] x [N,M] -> [B,T,N,C] (broadcast cluster value)."""
    return torch.einsum("nm,btmc->btnc", c, z)


def validate_resolution_schedule(
    schedule: list[dict],
    pred_len: int,
) -> list[dict]:
    """Validate Ω: h nondecreasing, capacity nonincreasing, last=(H,1)."""
    if not schedule:
        raise ValueError("resolution_schedule must be non-empty")
    cleaned = []
    prev_h, prev_cap = None, None
    for i, item in enumerate(schedule):
        if "h" not in item or "capacity" not in item:
            raise ValueError(f"stage {i} must contain 'h' and 'capacity': {item}")
        h = int(item["h"])
        cap = int(item["capacity"])
        if h <= 0 or cap <= 0:
            raise ValueError(f"stage {i}: h and capacity must be positive")
        if prev_h is not None and h < prev_h:
            raise ValueError(f"h must be nondecreasing: stage {i} h={h} < prev={prev_h}")
        if prev_cap is not None and cap > prev_cap:
            raise ValueError(
                f"capacity must be nonincreasing: stage {i} capacity={cap} > prev={prev_cap}"
            )
        cleaned.append({"h": h, "capacity": cap})
        prev_h, prev_cap = h, cap
    if cleaned[-1]["h"] != int(pred_len):
        raise ValueError(
            f"last stage h={cleaned[-1]['h']} must equal pred_len={pred_len}"
        )
    if cleaned[-1]["capacity"] != 1:
        raise ValueError(
            f"last stage capacity must be 1, got {cleaned[-1]['capacity']}"
        )
    return cleaned


class ResolutionSpace(nn.Module):
    """Unified Π_s = graph_project ∘ temporal_project, Λ_s = temporal_lift ∘ graph_lift."""

    def __init__(
        self,
        h: int,
        capacity: int,
        full_h: int,
        num_nodes: int,
        c_assign: torch.Tensor,
        p_proj: torch.Tensor,
        temporal_lift_mode: str = "linear",
    ):
        super().__init__()
        self.h = int(h)
        self.capacity = int(capacity)
        self.full_h = int(full_h)
        self.num_nodes = int(num_nodes)
        self.num_regions = int(c_assign.shape[1])
        self.temporal_lift_mode = temporal_lift_mode
        if c_assign.shape != (num_nodes, self.num_regions):
            raise ValueError(
                f"C shape {tuple(c_assign.shape)} != ({num_nodes}, {self.num_regions})"
            )
        if p_proj.shape != (self.num_regions, num_nodes):
            raise ValueError(
                f"P shape {tuple(p_proj.shape)} != ({self.num_regions}, {num_nodes})"
            )
        self.register_buffer("C", c_assign.float())
        self.register_buffer("P", p_proj.float())

    def project(self, y: torch.Tensor) -> torch.Tensor:
        """Π_s(y): [B,H,N,C] -> [B,h,M,C]."""
        z_t = temporal_project(y, self.h)
        return graph_project(z_t, self.P)

    def lift(self, z: torch.Tensor) -> torch.Tensor:
        """Λ_s(z): [B,h,M,C] -> [B,H,N,C]."""
        y_g = graph_lift(z, self.C)
        return temporal_lift(y_g, self.full_h, mode=self.temporal_lift_mode)

    def diagnostics(self) -> dict:
        return {
            "h": self.h,
            "capacity": self.capacity,
            "num_regions": self.num_regions,
            "num_nodes": self.num_nodes,
            "C_shape": list(self.C.shape),
            "P_shape": list(self.P.shape),
            "is_graph_identity": self.capacity == 1,
            "is_temporal_identity": self.h == self.full_h,
        }
