"""HyperD-style progressive residual head with zero-init condition adapters."""
from __future__ import annotations

import torch
import torch.nn as nn


class ProgressiveHyperDHead(nn.Module):
    """
    HyperD-style forecasting head plus a lightweight
    previous-state adapter.
    """

    def __init__(
        self,
        feature_dim: int,
        fc_hidden_size: int,
        chain_lengths: list[int],
        condition_hidden_size: int = 32,
        use_prev_condition: bool = True,
    ) -> None:
        super().__init__()

        self.chain_lengths = list(chain_lengths)
        self.use_prev_condition = use_prev_condition

        self.shared_hidden = nn.Sequential(
            nn.Linear(feature_dim, fc_hidden_size),
            nn.LeakyReLU(),
        )

        self.stage_output_heads = nn.ModuleDict()
        self.condition_adapters = nn.ModuleDict()

        for stage_len in self.chain_lengths:
            key = str(stage_len)

            self.stage_output_heads[key] = nn.Linear(
                fc_hidden_size,
                stage_len,
            )

            adapter = nn.Sequential(
                nn.Linear(
                    stage_len,
                    condition_hidden_size,
                ),
                nn.LeakyReLU(),
                nn.Linear(
                    condition_hidden_size,
                    stage_len,
                ),
            )

            nn.init.zeros_(adapter[-1].weight)
            nn.init.zeros_(adapter[-1].bias)

            self.condition_adapters[key] = adapter

    def forward(
        self,
        encoded_history: torch.Tensor,
        prev_residual_state: torch.Tensor,
        stage_len: int,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            encoded_history:
                [B, N, D]
            prev_residual_state:
                [B, stage_len, N]
            stage_len:
                current temporal resolution

        Returns:
            base_proposal:
                [B, stage_len, N]
            condition_delta:
                [B, stage_len, N]
            state_proposal:
                [B, stage_len, N]
            correction:
                [B, stage_len, N]
        """
        key = str(stage_len)

        if key not in self.stage_output_heads:
            raise KeyError(
                f"Unknown stage_len={stage_len}. "
                f"Available={self.chain_lengths}."
            )

        if prev_residual_state.ndim != 3:
            raise ValueError(
                "prev_residual_state must have shape [B, h, N], "
                f"got {tuple(prev_residual_state.shape)}."
            )

        hidden = self.shared_hidden(encoded_history)

        base_proposal = self.stage_output_heads[key](
            hidden
        ).transpose(1, 2)

        if self.use_prev_condition:
            condition_delta = self.condition_adapters[key](
                prev_residual_state.transpose(1, 2)
            ).transpose(1, 2)
        else:
            condition_delta = torch.zeros_like(base_proposal)

        state_proposal = base_proposal + condition_delta
        correction = state_proposal - prev_residual_state

        return {
            "base_proposal": base_proposal,
            "condition_delta": condition_delta,
            "state_proposal": state_proposal,
            "correction": correction,
        }
