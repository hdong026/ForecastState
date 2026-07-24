"""Chain loss for ForecastSpace (ported from KASA ChainForecastingRunner._legacy_loss)."""

from __future__ import annotations

import torch

from .forecast_state_chain import ForecastStateChain


def _rescale_pair(scaler, pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if scaler is not None and getattr(scaler, "rescale", False):
        pred = scaler.inverse_transform(pred.clone())
        target = scaler.inverse_transform(target.clone())
    return pred, target


def _raw_loss(pred: torch.Tensor, target: torch.Tensor, scaler, metric_forward, loss_fn) -> torch.Tensor:
    pred_rescaled, target_rescaled = _rescale_pair(scaler, pred, target)
    return metric_forward(loss_fn, {"prediction": pred_rescaled, "target": target_rescaled})


def _weighted_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: float,
    scaler,
    metric_forward,
    loss_fn,
) -> torch.Tensor:
    if float(weight) == 0.0:
        return torch.tensor(0.0, device=target.device)
    return float(weight) * _raw_loss(pred, target, scaler, metric_forward, loss_fn)


def compute_legacy_chain_loss(
    out: dict,
    real_value: torch.Tensor,
    chain_lengths: list[int],
    chain_loss_weights: list[float],
    scaler,
    metric_forward,
    loss_fn,
    spatial_stage_loss_weights: list[float] | None = None,
    spatial_graph_loss_weights: list[float] | None = None,
) -> torch.Tensor:
    spatial_stage_loss_weights = spatial_stage_loss_weights or [0.0, 0.0, 1.0]
    spatial_graph_loss_weights = spatial_graph_loss_weights or [0.0, 0.0, 0.0]

    preds = out["chain_preds"]
    targets = [ForecastStateChain.pool_target(real_value, k) for k in chain_lengths]
    final_target = targets[-1]

    loss = torch.tensor(0.0, device=real_value.device)
    if len(chain_loss_weights) > 1:
        for weight, pred, target in zip(chain_loss_weights[:-1], preds[:-1], targets[:-1]):
            loss = loss + _weighted_loss(pred, target, weight, scaler, metric_forward, loss_fn)

    loss = loss + _weighted_loss(
        out["pred"],
        final_target,
        chain_loss_weights[-1],
        scaler,
        metric_forward,
        loss_fn,
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
            loss = loss + _weighted_loss(pred, final_target, weight, scaler, metric_forward, loss_fn)

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
            loss = loss + _weighted_loss(pred, final_target, weight, scaler, metric_forward, loss_fn)
    return loss
