"""Unit tests for temporal/graph resolution spaces."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines.ForecastStateGraphResolution.arch.resolution_space import (
    ResolutionSpace,
    temporal_lift,
    temporal_project,
    validate_resolution_schedule,
)


def test_temporal_identity():
    y = torch.randn(2, 12, 5, 1)
    z = temporal_project(y, 12)
    assert torch.allclose(z, y)
    y2 = temporal_lift(z, 12)
    assert torch.allclose(y2, y)


def test_temporal_shapes_and_constant():
    y = torch.ones(2, 12, 4, 1) * 3.0
    z = temporal_project(y, 3)
    assert z.shape == (2, 3, 4, 1)
    assert torch.allclose(z, torch.ones_like(z) * 3.0)
    y_hat = temporal_lift(z, 12, mode="linear")
    assert y_hat.shape == (2, 12, 4, 1)
    assert torch.allclose(y_hat, torch.ones_like(y_hat) * 3.0, atol=1e-5)
    assert torch.isfinite(y_hat).all()


def test_temporal_grad():
    y = torch.randn(2, 12, 3, 1, requires_grad=True)
    z = temporal_project(y, 6)
    loss = z.sum()
    loss.backward()
    assert y.grad is not None
    assert torch.isfinite(y.grad).all()


def test_graph_identity():
    n = 7
    c = torch.eye(n)
    p = torch.eye(n)
    space = ResolutionSpace(h=12, capacity=1, full_h=12, num_nodes=n, c_assign=c, p_proj=p)
    y = torch.randn(2, 12, n, 1)
    z = space.project(y)
    assert torch.allclose(z, y)
    y2 = space.lift(z)
    assert torch.allclose(y2, y)


def test_schedule_validation():
    ok = validate_resolution_schedule(
        [{"h": 3, "capacity": 4}, {"h": 6, "capacity": 2}, {"h": 12, "capacity": 1}],
        pred_len=12,
    )
    assert ok[-1]["capacity"] == 1
    try:
        validate_resolution_schedule([{"h": 12, "capacity": 2}], pred_len=12)
        raise AssertionError("should fail last capacity")
    except ValueError:
        pass
    try:
        validate_resolution_schedule(
            [{"h": 6, "capacity": 1}, {"h": 3, "capacity": 1}],
            pred_len=3,
        )
        raise AssertionError("should fail decreasing h")
    except ValueError:
        pass


if __name__ == "__main__":
    test_temporal_identity()
    test_temporal_shapes_and_constant()
    test_temporal_grad()
    test_graph_identity()
    test_schedule_validation()
    print("test_resolution_space: OK")
