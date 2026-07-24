"""Scale-matched chain loss + DVA for HyperD Forecast-State Chain."""
from __future__ import annotations

import torch

from .temporal_ops import temporal_project


def _rescale_pair(
    scaler,
    pred: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if scaler is not None and getattr(
        scaler,
        "rescale",
        False,
    ):
        pred = scaler.inverse_transform(pred.clone())
        target = scaler.inverse_transform(target.clone())

    return pred, target


def _raw_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    scaler,
    metric_forward,
    loss_fn,
) -> torch.Tensor:
    pred, target = _rescale_pair(
        scaler,
        pred,
        target,
    )

    return metric_forward(
        loss_fn,
        {
            "prediction": pred,
            "target": target,
        },
    )


def compute_hyperd_chain_loss(
    out: dict,
    real_value: torch.Tensor,
    chain_lengths: list[int],
    chain_loss_weights: list[float],
    scaler,
    metric_forward,
    loss_fn,
) -> torch.Tensor:
    if len(chain_lengths) != len(chain_loss_weights):
        raise ValueError(
            "chain_lengths and chain_loss_weights "
            "must have the same length."
        )

    stage_states = out["stage_forecast_states"]

    if len(stage_states) != len(chain_lengths):
        raise ValueError(
            "Number of returned stage states does not "
            "match chain_lengths."
        )

    total_loss = real_value.new_zeros(())

    for stage_len, weight, stage_prediction in zip(
        chain_lengths,
        chain_loss_weights,
        stage_states,
    ):
        weight = float(weight)

        if weight == 0.0:
            continue

        target_state = temporal_project(
            real_value[..., 0],
            target_len=stage_len,
        ).unsqueeze(-1)

        total_loss = total_loss + weight * _raw_loss(
            stage_prediction,
            target_state,
            scaler,
            metric_forward,
            loss_fn,
        )

    total_loss = total_loss + out["dual_view_loss"]

    return total_loss
