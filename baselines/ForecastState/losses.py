"""Scale-matched progressive temporal losses (document Part 7)."""
from __future__ import annotations

import torch

from .temporal_ops import temporal_project


def progressive_temporal_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    stage_predictions: list[torch.Tensor],
    temporal_resolutions: list[int],
    masked_mae_fn,
    null_val: float = 0.0,
    aux_weight: float = 0.0,
) -> torch.Tensor:
    """Final loss plus optional scale-matched auxiliary supervision.

    Never compares a short-horizon state directly with a full-horizon target.
    """
    final_loss = masked_mae_fn(
        prediction,
        target,
        null_val=null_val,
    )

    if aux_weight <= 0.0 or len(stage_predictions) <= 1:
        return final_loss

    auxiliary_losses = []

    for stage_prediction, resolution in zip(
        stage_predictions[:-1],
        temporal_resolutions[:-1],
    ):
        prediction_state = temporal_project(
            stage_prediction,
            target_length=resolution,
        )
        target_state = temporal_project(
            target,
            target_length=resolution,
        )
        if prediction_state.shape != target_state.shape:
            raise RuntimeError(
                "Scale-matched auxiliary supervision shape mismatch: "
                f"pred {tuple(prediction_state.shape)} vs "
                f"target {tuple(target_state.shape)} at resolution {resolution}."
            )
        auxiliary_losses.append(
            masked_mae_fn(
                prediction_state,
                target_state,
                null_val=null_val,
            )
        )

    return final_loss + aux_weight * torch.stack(auxiliary_losses).mean()
