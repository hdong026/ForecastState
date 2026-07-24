"""Temporal-only Forecast-State Chain (document Part 6)."""
from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from .temporal_ops import temporal_lift, temporal_project


class ProgressiveTemporalForecasting(nn.Module):
    """Temporal-only Forecast-State Chain."""

    def __init__(
        self,
        temporal_steps: Sequence[nn.Module],
        temporal_resolutions: Sequence[int],
        full_horizon: int,
        output_channels: int = 1,
        use_prev_condition: bool = True,
        learnable_stage_scale: bool = False,
    ) -> None:
        super().__init__()

        temporal_resolutions = list(temporal_resolutions)

        if not temporal_resolutions:
            raise ValueError("temporal_resolutions cannot be empty.")

        if len(temporal_steps) != len(temporal_resolutions):
            raise ValueError(
                "The number of temporal steps must equal the number "
                "of temporal resolutions."
            )

        if temporal_resolutions != sorted(temporal_resolutions):
            raise ValueError("Temporal resolutions must be non-decreasing.")

        if temporal_resolutions[-1] != full_horizon:
            raise ValueError(
                "The final temporal resolution must equal full_horizon."
            )

        self.temporal_steps = nn.ModuleList(temporal_steps)
        self.temporal_resolutions = temporal_resolutions
        self.full_horizon = full_horizon
        self.output_channels = output_channels
        self.use_prev_condition = use_prev_condition

        if learnable_stage_scale:
            self.stage_scales = nn.Parameter(
                torch.ones(len(temporal_resolutions))
            )
        else:
            self.register_buffer(
                "stage_scales",
                torch.ones(len(temporal_resolutions)),
                persistent=False,
            )

        self.latest_stage_residuals: list[torch.Tensor] = []
        self.latest_stage_predictions: list[torch.Tensor] = []
        self.latest_stage_states: list[torch.Tensor] = []

    def forward(
        self,
        history_data: torch.Tensor,
        future_data: torch.Tensor | None = None,
        batch_seen: int | None = None,
        epoch: int | None = None,
        train: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        del future_data, kwargs

        if history_data.ndim != 4:
            raise ValueError(
                "history_data must have shape [B, P, N, C_in], "
                f"got {tuple(history_data.shape)}."
            )

        batch_size, _, num_nodes, _ = history_data.shape

        # Ŷ^(0) = 0
        full_forecast = history_data.new_zeros(
            batch_size,
            self.full_horizon,
            num_nodes,
            self.output_channels,
        )

        stage_residuals = []
        stage_predictions = []
        stage_states = []

        for stage_index, (resolution, temporal_step) in enumerate(
            zip(self.temporal_resolutions, self.temporal_steps)
        ):
            # Z̄_s = D_{h_s}(Ŷ^(s-1)); disabled entirely when use_prev_condition=False
            if self.use_prev_condition:
                prev_condition = temporal_project(
                    full_forecast,
                    target_length=resolution,
                )
            else:
                prev_condition = None

            # R_s = F_s(X, Z̄_s)
            residual = temporal_step(
                history_data=history_data,
                prev_condition=prev_condition,
                target_length=resolution,
                batch_seen=batch_seen,
                epoch=epoch,
                train=train,
            )

            expected_shape = (
                batch_size,
                resolution,
                num_nodes,
                self.output_channels,
            )
            if residual.shape != expected_shape:
                raise RuntimeError(
                    f"Stage {stage_index} returned {tuple(residual.shape)}, "
                    f"expected {expected_shape}."
                )

            # ΔŶ^(s) = U_{h_s}(R_s)
            lifted_residual = temporal_lift(
                residual,
                full_horizon=self.full_horizon,
            )

            # Ŷ^(s) = Ŷ^(s-1) + α_s ΔŶ^(s), α_s = 1 by default
            full_forecast = (
                full_forecast
                + self.stage_scales[stage_index] * lifted_residual
            )

            # S_s = D_{h_s}(Ŷ^(s))
            forecast_state = temporal_project(
                full_forecast,
                target_length=resolution,
            )

            stage_residuals.append(residual)
            stage_predictions.append(full_forecast)
            stage_states.append(forecast_state)

        self.latest_stage_residuals = stage_residuals
        self.latest_stage_predictions = stage_predictions
        self.latest_stage_states = stage_states

        return full_forecast
