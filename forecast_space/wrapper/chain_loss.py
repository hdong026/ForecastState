"""Chain loss ported from KASA ChainForecastingRunner._legacy_loss."""

from __future__ import annotations

import torch
from basicts.metrics import masked_mae

from forecast_space.models.kasa_g1 import ChainForecasting


def _rescale_flow(
    tensor: torch.Tensor,
    mean: float,
    std: float,
) -> torch.Tensor:
    return tensor * std + mean


def _raw_mae(
    pred: torch.Tensor,
    target: torch.Tensor,
    mean: float,
    std: float,
    null_val: float = 0.0,
) -> torch.Tensor:
    pred_rescaled = _rescale_flow(pred, mean, std)
    target_rescaled = _rescale_flow(target, mean, std)
    if torch.isnan(torch.tensor(null_val)):
        mask = ~torch.isnan(target_rescaled)
        targets_mask = mask
    else:
        eps = 5e-5
        targets_mask = ~torch.isclose(
            target_rescaled,
            torch.tensor(null_val, device=target_rescaled.device),
            atol=eps,
            rtol=0.0,
        )
    return masked_mae(pred_rescaled, target_rescaled, targets_mask)


def _weighted_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: float,
    mean: float,
    std: float,
    null_val: float,
) -> torch.Tensor:
    if float(weight) == 0.0:
        return torch.tensor(0.0, device=target.device)
    return float(weight) * _raw_mae(pred, target, mean, std, null_val)


def compute_legacy_chain_loss(
    out: dict,
    real_value: torch.Tensor,
    chain_lengths: list[int],
    chain_loss_weights: list[float],
    spatial_stage_loss_weights: list[float],
    spatial_graph_loss_weights: list[float],
    flow_mean: float,
    flow_std: float,
    null_val: float = 0.0,
) -> torch.Tensor:
    preds = out["chain_preds"]
    targets = [ChainForecasting.pool_target(real_value, k) for k in chain_lengths]
    final_target = targets[-1]

    loss = torch.tensor(0.0, device=real_value.device)
    if len(chain_loss_weights) > 1:
        for weight, pred, target in zip(
            chain_loss_weights[:-1], preds[:-1], targets[:-1]
        ):
            loss = loss + _weighted_loss(
                pred, target, weight, flow_mean, flow_std, null_val
            )

    loss = loss + _weighted_loss(
        out["pred"],
        final_target,
        chain_loss_weights[-1],
        flow_mean,
        flow_std,
        null_val,
    )

    spatial_stage_preds = out.get("spatial_stage_preds") or []
    final_pred = out["pred"]
    if spatial_stage_preds and any(float(w) != 0.0 for w in spatial_stage_loss_weights):
        weights = list(spatial_stage_loss_weights)
        if len(weights) < len(spatial_stage_preds):
            weights = weights + [weights[-1]] * (len(spatial_stage_preds) - len(weights))
        weights = weights[: len(spatial_stage_preds)]
        for pred, weight in zip(spatial_stage_preds, weights):
            if pred is final_pred:
                continue
            loss = loss + _weighted_loss(
                pred, final_target, weight, flow_mean, flow_std, null_val
            )

    graph_stage_preds = []
    graph_diag = out.get("graph_resolution_diagnostics") or {}
    if graph_diag:
        graph_stage_preds = graph_diag.get("node_stage_preds") or []
    if graph_stage_preds and any(float(w) != 0.0 for w in spatial_graph_loss_weights):
        g_weights = list(spatial_graph_loss_weights)
        if len(g_weights) < len(graph_stage_preds):
            g_weights = g_weights + [g_weights[-1]] * (len(graph_stage_preds) - len(g_weights))
        g_weights = g_weights[: len(graph_stage_preds)]
        for pred, weight in zip(graph_stage_preds, g_weights):
            if len(graph_stage_preds) > 1 and pred is final_pred:
                continue
            loss = loss + _weighted_loss(
                pred, final_target, weight, flow_mean, flow_std, null_val
            )
    return loss


def kasa_g1_chain_loss(
    prediction: torch.Tensor,
    targets: torch.Tensor,
    targets_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fallback MAE when chain loss is not precomputed (validation/test)."""
    del targets_mask
    return masked_mae(prediction, targets)
