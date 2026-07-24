"""HyperD backbone with temporal Forecast-State Chain."""
from __future__ import annotations

from argparse import Namespace

import torch
import torch.nn as nn

from baselines.HyperD.arch.hyperd_arch import (
    Hybrid_Periodic_Pattern,
)
from baselines.HyperD.arch.loss import dual_view_loss

from .progressive_head import ProgressiveHyperDHead
from .stfe_encoder import HyperDSTFEEncoder
from .temporal_ops import temporal_lift, temporal_project


class HyperDForecastStateChain(nn.Module):
    """
    HyperD backbone with temporal Forecast-State Chain.

    The periodic future is computed once. The chain progressively
    refines the residual forecast at resolutions 3, 6, and 12.
    """

    def __init__(self, **model_args) -> None:
        super().__init__()

        configs = Namespace(**model_args)

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.num_nodes = configs.num_nodes

        self.path_daily = configs.init_path_daily
        self.path_weekly = configs.init_path_weekly
        self.adj = configs.adj

        self.alpha = configs.alpha
        self.F_low = configs.F_low

        self.embed_size = configs.embed_size
        self.hidden_size = configs.hidden_size
        self.fc_hidden_size = configs.fc_hidden_size

        self.daily_len = configs.time_of_day_size
        self.weekly_len = (
            configs.day_of_week_size * self.daily_len
        )

        self.chain_lengths = list(
            getattr(configs, "chain_lengths", [3, 6, 12])
        )
        self.chain_loss_weights = list(
            getattr(
                configs,
                "chain_loss_weights",
                [0.0, 0.0, 1.0],
            )
        )
        self.use_prev_condition = bool(
            getattr(configs, "use_prev_condition", True)
        )
        self.condition_hidden_size = int(
            getattr(
                configs,
                "condition_hidden_size",
                32,
            )
        )
        self.use_dual_view_loss = bool(
            getattr(configs, "use_dual_view_loss", True)
        )
        self.dual_view_weight = float(
            getattr(configs, "dual_view_weight", 1.0)
        )
        self.debug_shapes = bool(
            getattr(configs, "debug_shapes", False)
        )
        self._debug_logged = False

        self._validate_configuration()

        self.daily_emb = Hybrid_Periodic_Pattern(
            period_len=self.daily_len,
            num_nodes=self.num_nodes,
            adj=self.adj,
            init_npy_path=self.path_daily,
        )

        self.weekly_emb = Hybrid_Periodic_Pattern(
            period_len=self.weekly_len,
            num_nodes=self.num_nodes,
            adj=self.adj,
            init_npy_path=self.path_weekly,
        )

        self.stfe_encoder = HyperDSTFEEncoder(
            num_nodes=self.num_nodes,
            seq_len=self.seq_len,
            embed_size=self.embed_size,
            hidden_size=self.hidden_size,
        )

        self.progressive_head = ProgressiveHyperDHead(
            feature_dim=self.stfe_encoder.output_dim,
            fc_hidden_size=self.fc_hidden_size,
            chain_lengths=self.chain_lengths,
            condition_hidden_size=self.condition_hidden_size,
            use_prev_condition=self.use_prev_condition,
        )

    def _validate_configuration(self) -> None:
        if not self.chain_lengths:
            raise ValueError(
                "chain_lengths must contain at least one stage."
            )

        if any(stage_len <= 0 for stage_len in self.chain_lengths):
            raise ValueError(
                f"Invalid chain_lengths={self.chain_lengths}."
            )

        if self.chain_lengths != sorted(self.chain_lengths):
            raise ValueError(
                "chain_lengths must be non-decreasing."
            )

        if self.chain_lengths[-1] != self.pred_len:
            raise ValueError(
                f"Last chain length {self.chain_lengths[-1]} "
                f"must equal pred_len {self.pred_len}."
            )

        if len(self.chain_lengths) != len(
            self.chain_loss_weights
        ):
            raise ValueError(
                "chain_lengths and chain_loss_weights "
                "must have equal length."
            )

    def _periodic_patterns(
        self,
        history_data: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        index_daily = (
            history_data[..., 1] * self.daily_len
        )
        index_daily = index_daily[:, -1, 0]

        index_weekly = (
            history_data[..., 1] * self.daily_len
            + history_data[..., 2] * self.weekly_len
        )
        index_weekly = index_weekly[:, -1, 0]

        S_D_in = self.daily_emb(
            index_daily,
            self.seq_len,
        )
        S_W_in = self.weekly_emb(
            index_weekly,
            self.seq_len,
        )
        S_in = S_D_in + S_W_in

        S_D_out = self.daily_emb(
            (index_daily + self.seq_len) % self.daily_len,
            self.pred_len,
        )
        S_W_out = self.weekly_emb(
            (index_weekly + self.seq_len) % self.weekly_len,
            self.pred_len,
        )
        S_out = S_D_out + S_W_out

        return S_in, S_out

    def _maybe_log_debug(
        self,
        S_out: torch.Tensor,
        encoded_history: torch.Tensor,
        stage_residual_proposals: list[torch.Tensor],
        stage_corrections: list[torch.Tensor],
        stage_residual_states: list[torch.Tensor],
        final_residual: torch.Tensor,
        final_prediction: torch.Tensor,
        dva_loss: torch.Tensor,
    ) -> None:
        if not self.debug_shapes or self._debug_logged:
            return
        self._debug_logged = True
        print(
            "[HyperDForecastStateChain debug]",
            f"chain_lengths={self.chain_lengths}",
            f"chain_loss_weights={self.chain_loss_weights}",
            f"use_prev_condition={self.use_prev_condition}",
            f"use_dual_view_loss={self.use_dual_view_loss}",
            f"periodic_out={tuple(S_out.shape)}",
            f"encoded_history={tuple(encoded_history.shape)}",
            f"proposals={[tuple(t.shape) for t in stage_residual_proposals]}",
            f"corr_norms={[float(t.abs().mean()) for t in stage_corrections]}",
            f"res_state_norms={[float(t.abs().mean()) for t in stage_residual_states]}",
            f"final_residual_norm={float(final_residual.abs().mean())}",
            f"prediction={tuple(final_prediction.shape)}",
            f"dual_view_loss={float(dva_loss.detach())}",
        )

    def forward(
        self,
        history_data: torch.Tensor,
        future_data: torch.Tensor | None = None,
        batch_seen: int | None = None,
        epoch: int | None = None,
        train: bool = False,
        return_all: bool = False,
        **kwargs,
    ):
        del future_data, batch_seen, epoch, train, kwargs

        if history_data.ndim != 4:
            raise ValueError(
                "history_data must have shape [B, P, N, C], "
                f"got {tuple(history_data.shape)}."
            )

        if history_data.shape[-1] < 3:
            raise ValueError(
                "HyperD requires flow, time-of-day, and "
                "day-of-week input features."
            )

        x = history_data[..., 0]

        S_in, S_out = self._periodic_patterns(
            history_data
        )

        residual_in = x - S_in

        encoded_history = self.stfe_encoder(
            residual_in
        )

        residual_full = torch.zeros_like(S_out)

        stage_residual_states = []
        stage_residual_proposals = []
        stage_corrections = []
        stage_full_residuals = []
        stage_full_predictions = []
        stage_forecast_states = []
        stage_base_proposals = []
        stage_condition_deltas = []

        for stage_len in self.chain_lengths:
            prev_residual_state = temporal_project(
                residual_full,
                target_len=stage_len,
            )

            head_out = self.progressive_head(
                encoded_history=encoded_history,
                prev_residual_state=prev_residual_state,
                stage_len=stage_len,
            )

            correction = head_out["correction"]

            residual_full = (
                residual_full
                + temporal_lift(
                    correction,
                    full_horizon=self.pred_len,
                )
            )

            full_prediction = S_out + residual_full

            residual_state = temporal_project(
                residual_full,
                target_len=stage_len,
            )
            forecast_state = temporal_project(
                full_prediction,
                target_len=stage_len,
            )

            stage_base_proposals.append(
                head_out["base_proposal"]
            )
            stage_condition_deltas.append(
                head_out["condition_delta"]
            )
            stage_residual_proposals.append(
                head_out["state_proposal"]
            )
            stage_corrections.append(correction)
            stage_residual_states.append(residual_state)
            stage_full_residuals.append(residual_full.clone())
            stage_full_predictions.append(full_prediction.clone())
            stage_forecast_states.append(forecast_state)

        final_prediction = stage_full_predictions[-1]
        final_residual = stage_full_residuals[-1]

        if self.use_dual_view_loss:
            loss_low, loss_high = dual_view_loss(
                final_prediction,
                S_out,
                final_residual,
                self.F_low,
            )
            dva_loss = (
                loss_low + loss_high
            ) * self.alpha * self.dual_view_weight
        else:
            dva_loss = final_prediction.new_zeros(())

        self._maybe_log_debug(
            S_out=S_out,
            encoded_history=encoded_history,
            stage_residual_proposals=stage_residual_proposals,
            stage_corrections=stage_corrections,
            stage_residual_states=stage_residual_states,
            final_residual=final_residual,
            final_prediction=final_prediction,
            dva_loss=dva_loss,
        )

        output = {
            "prediction": final_prediction.unsqueeze(-1),
            "pred": final_prediction.unsqueeze(-1),
            "periodic_in": S_in.unsqueeze(-1),
            "periodic_out": S_out.unsqueeze(-1),
            "residual_in": residual_in.unsqueeze(-1),
            "residual_final": final_residual.unsqueeze(-1),
            "dual_view_loss": dva_loss,
            "stage_base_proposals": [
                tensor.unsqueeze(-1)
                for tensor in stage_base_proposals
            ],
            "stage_condition_deltas": [
                tensor.unsqueeze(-1)
                for tensor in stage_condition_deltas
            ],
            "stage_residual_proposals": [
                tensor.unsqueeze(-1)
                for tensor in stage_residual_proposals
            ],
            "stage_corrections": [
                tensor.unsqueeze(-1)
                for tensor in stage_corrections
            ],
            "stage_residual_states": [
                tensor.unsqueeze(-1)
                for tensor in stage_residual_states
            ],
            "stage_full_residuals": [
                tensor.unsqueeze(-1)
                for tensor in stage_full_residuals
            ],
            "stage_full_predictions": [
                tensor.unsqueeze(-1)
                for tensor in stage_full_predictions
            ],
            "stage_forecast_states": [
                tensor.unsqueeze(-1)
                for tensor in stage_forecast_states
            ],
            "chain_preds": [
                tensor.unsqueeze(-1)
                for tensor in stage_forecast_states
            ],
            "chain_lengths": list(self.chain_lengths),
        }

        if return_all:
            return output

        return {
            "prediction": output["prediction"],
            "dual_view_loss": output["dual_view_loss"],
        }
