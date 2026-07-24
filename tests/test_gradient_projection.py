"""Unit tests for final-primary gradient projection."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines.ForecastStateGraphResolution.arch.gradient_projection import (
    aggregate_and_cap_aux,
    project_auxiliary_grad,
)


def test_conflicting_gradient_projected():
    g0 = torch.tensor([1.0, 0.0])
    gs = torch.tensor([-2.0, 1.0])  # conflicts on dim0
    g_proj, stats = project_auxiliary_grad(gs, g0)
    assert stats["projection_triggered"] is True
    # after projection, dot with g0 should be ~0 (removed negative component)
    assert torch.dot(g_proj, g0).item() >= -1e-6
    assert abs(g_proj[0].item()) < 1e-6
    assert abs(g_proj[1].item() - 1.0) < 1e-6


def test_nonconflicting_unchanged():
    g0 = torch.tensor([1.0, 0.0])
    gs = torch.tensor([2.0, 3.0])
    g_proj, stats = project_auxiliary_grad(gs, g0)
    assert stats["projection_triggered"] is False
    assert torch.allclose(g_proj, gs)


def test_norm_cap():
    g0 = torch.tensor([1.0, 0.0])
    g_aux_big = torch.tensor([0.0, 10.0])
    capped, stats = aggregate_and_cap_aux([g_aux_big], g0, rho=0.5)
    assert stats["cap_triggered"] is True
    assert abs(torch.linalg.norm(capped).item() - 0.5) < 1e-5


if __name__ == "__main__":
    test_conflicting_gradient_projected()
    test_nonconflicting_unchanged()
    test_norm_cap()
    print("test_gradient_projection: OK")
