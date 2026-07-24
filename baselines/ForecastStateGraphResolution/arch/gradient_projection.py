"""Final-primary gradient projection utilities.

Paper: Eq. (33)–(38).
"""
from __future__ import annotations

from typing import Iterable, Optional

import torch


def flatten_grads(
    parameters: Iterable[torch.nn.Parameter],
    grads: list[Optional[torch.Tensor]],
) -> torch.Tensor:
    chunks = []
    for p, g in zip(parameters, grads):
        if g is None:
            chunks.append(torch.zeros(p.numel(), device=p.device, dtype=p.dtype))
        else:
            chunks.append(g.reshape(-1))
    if not chunks:
        return torch.tensor(0.0)
    return torch.cat(chunks)


def unflatten_and_set_grads(
    parameters: list[torch.nn.Parameter],
    flat: torch.Tensor,
) -> None:
    offset = 0
    for p in parameters:
        numel = p.numel()
        g = flat[offset : offset + numel].view_as(p)
        offset += numel
        p.grad = g.clone()


def project_auxiliary_grad(
    g_s: torch.Tensor,
    g0: torch.Tensor,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, dict]:
    """Remove conflicting component of g_s against g0 (Eq. 35)."""
    dot = torch.dot(g_s, g0)
    g0_norm_sq = torch.dot(g0, g0) + eps
    g0_norm = torch.sqrt(g0_norm_sq)
    gs_norm = torch.linalg.norm(g_s) + eps
    cos = (dot / (g0_norm * gs_norm)).item()
    triggered = bool(dot.item() < 0)
    if triggered:
        g_proj = g_s - (dot / g0_norm_sq) * g0
    else:
        g_proj = g_s
    stats = {
        "dot": float(dot.item()),
        "cosine": float(cos),
        "projection_triggered": triggered,
        "norm_before": float(torch.linalg.norm(g_s).item()),
        "norm_after": float(torch.linalg.norm(g_proj).item()),
    }
    return g_proj, stats


def aggregate_and_cap_aux(
    projected: list[torch.Tensor],
    g0: torch.Tensor,
    rho: float = 1.0,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, dict]:
    if not projected:
        zero = torch.zeros_like(g0)
        return zero, {"cap_triggered": False, "aux_norm": 0.0, "g0_norm": float(torch.linalg.norm(g0).item())}
    g_aux = torch.stack(projected, dim=0).mean(dim=0)
    g0_norm = torch.linalg.norm(g0)
    aux_norm = torch.linalg.norm(g_aux)
    max_allowed = float(rho) * g0_norm
    cap_triggered = bool(aux_norm.item() > max_allowed.item() + 1e-12)
    if cap_triggered:
        g_aux = g_aux * (max_allowed / (aux_norm + eps))
    stats = {
        "cap_triggered": cap_triggered,
        "aux_norm": float(torch.linalg.norm(g_aux).item()),
        "aux_norm_before_cap": float(aux_norm.item()),
        "g0_norm": float(g0_norm.item()),
        "rho": float(rho),
    }
    return g_aux, stats


def compute_param_grads(
    loss: torch.Tensor,
    parameters: list[torch.nn.Parameter],
    retain_graph: bool = False,
) -> list[Optional[torch.Tensor]]:
    grads = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        create_graph=False,
        allow_unused=True,
    )
    return list(grads)
