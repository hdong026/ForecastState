"""Thin adapter around the verified KASA TemporalStep (document Part 5)."""
from __future__ import annotations

import torch
import torch.nn as nn


class KASATemporalStepAdapter(nn.Module):
    """Thin adapter around the verified KASA-ST TemporalStep.

    Maps the progressive-chain call signature onto the real KASA interface:

        residual = kasa_step(
            history_data=history_data,
            prev_forecast=prev_condition,
            spatial_codebook=spatial_codebook,
        )

    Each adapter wraps one KASATemporalStep whose ``output_len`` equals the
    stage resolution. ``target_length`` is validated against that fixed length.
    """

    def __init__(
        self,
        kasa_step: nn.Module,
        output_channels: int,
        spatial_codebook: nn.Parameter | None = None,
    ) -> None:
        super().__init__()
        self.kasa_step = kasa_step
        self.output_channels = int(output_channels)
        self.spatial_codebook = spatial_codebook
        if not hasattr(kasa_step, "output_len"):
            raise TypeError(
                "kasa_step must expose output_len matching the stage resolution."
            )

    def forward(
        self,
        history_data: torch.Tensor,
        prev_condition: torch.Tensor | None,
        target_length: int,
        batch_seen: int | None = None,
        epoch: int | None = None,
        train: bool = True,
    ) -> torch.Tensor:
        del batch_seen, epoch, train  # unused by KASA TemporalStep

        if int(target_length) != int(self.kasa_step.output_len):
            raise RuntimeError(
                f"Adapter target_length={target_length} does not match "
                f"KASATemporalStep.output_len={self.kasa_step.output_len}."
            )

        if history_data.ndim != 4:
            raise ValueError(
                f"history_data must be [B, P, N, C], got {tuple(history_data.shape)}."
            )
        if prev_condition is not None:
            if prev_condition.ndim != 4:
                raise ValueError(
                    f"prev_condition must be [B, h, N, C], got {tuple(prev_condition.shape)}."
                )
            if prev_condition.shape[1] != target_length:
                raise RuntimeError(
                    f"prev_condition temporal length {prev_condition.shape[1]} "
                    f"!= target_length {target_length}."
                )

        residual = self.kasa_step(
            history_data=history_data,
            prev_forecast=prev_condition,
            spatial_codebook=self.spatial_codebook,
        )

        expected_shape = (
            history_data.shape[0],
            target_length,
            history_data.shape[2],
            self.output_channels,
        )

        if residual.shape != expected_shape:
            raise RuntimeError(
                f"KASA TemporalStep returned {tuple(residual.shape)}, "
                f"expected {expected_shape}."
            )

        return residual
