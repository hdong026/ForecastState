"""Diagnostics helpers for stage-wise forecast-state logging."""
from __future__ import annotations

from typing import Any

import torch


def stage_shape_report(
    stage_idx: int,
    h: int,
    capacity: int,
    pred: torch.Tensor,
    target: torch.Tensor,
    mae: float | None = None,
) -> dict[str, Any]:
    return {
        "stage": stage_idx,
        "forecast_resolution": {"h": h, "capacity": capacity},
        "prediction_shape": list(pred.shape),
        "target_shape": list(target.shape),
        "MAE": mae,
    }


def assert_same_resolution(pred: torch.Tensor, target: torch.Tensor, stage_idx: int) -> None:
    if pred.shape != target.shape:
        raise AssertionError(
            f"Scale-matched supervision failed at stage {stage_idx}: "
            f"pred {tuple(pred.shape)} vs target {tuple(target.shape)}"
        )


def summarize_alphas(alphas: list[torch.Tensor]) -> list[float]:
    out = []
    for a in alphas:
        if a is None:
            out.append(float("nan"))
        else:
            out.append(float(a.detach().float().mean().item()))
    return out
