"""BasicTS entry model: builds shared KASA codebooks + progressive chain."""
from __future__ import annotations

import torch
from torch import nn

from .kasa_adapter import KASATemporalStepAdapter
from .kasa_temporal_step import KASATemporalStep
from .progressive_temporal import ProgressiveTemporalForecasting


class ForecastStateProgressive(nn.Module):
    """Temporal-only progressive forecasting for BasicTS.

    No graph convolution, clustering, pooling, or adaptive adjacency.
    """

    def __init__(self, **model_args):
        super().__init__()
        self.num_nodes = int(model_args.get("num_nodes", model_args.get("node_size")))
        self.input_len = int(model_args["input_len"])
        self.output_len = int(model_args.get("output_len", model_args.get("pred_len", 12)))
        self.output_channels = int(model_args.get("output_dim", model_args.get("output_channels", 1)))

        temporal_resolutions = list(
            model_args.get("temporal_resolutions", [3, 6, 12])
        )
        self.temporal_resolutions = [int(h) for h in temporal_resolutions]
        self.use_prev_condition = bool(model_args.get("use_prev_condition", True))
        self.learnable_stage_scale = bool(model_args.get("learnable_stage_scale", False))
        self.aux_loss_weight = float(model_args.get("aux_loss_weight", 0.0))

        patch_len = int(model_args.get("patch_len", 3))
        stride = int(model_args.get("stride", 4))
        td_size = int(model_args.get("td_size", 288))
        dw_size = int(model_args.get("dw_size", 7))
        d_td = int(model_args.get("d_td", 32))
        d_dw = int(model_args.get("d_dw", 32))
        d_d = int(model_args.get("d_d", 32))
        d_spa = int(model_args.get("d_spa", 32))
        num_layer = int(model_args.get("num_layer", 2))
        if_time_in_day = bool(model_args.get("if_time_in_day", True))
        if_day_in_week = bool(model_args.get("if_day_in_week", True))
        if_spatial = bool(model_args.get("if_spatial", True))

        # Shared KASA codebooks (exact KASA design)
        self.td_codebook = None
        self.dw_codebook = None
        self.spa_codebook = None
        if if_time_in_day:
            self.td_codebook = nn.Parameter(torch.empty(td_size, d_td))
            nn.init.xavier_uniform_(self.td_codebook)
        if if_day_in_week:
            self.dw_codebook = nn.Parameter(torch.empty(dw_size, d_dw))
            nn.init.xavier_uniform_(self.dw_codebook)
        if if_spatial:
            self.spa_codebook = nn.Parameter(torch.empty(self.num_nodes, d_spa))
            nn.init.xavier_uniform_(self.spa_codebook)

        step_kwargs = dict(
            input_len=self.input_len,
            patch_len=patch_len,
            stride=stride,
            td_size=td_size,
            dw_size=dw_size,
            td_codebook=self.td_codebook,
            dw_codebook=self.dw_codebook,
            spa_codebook=self.spa_codebook,
            if_time_in_day=if_time_in_day,
            if_day_in_week=if_day_in_week,
            if_spatial=if_spatial,
            d_d=d_d,
            d_td=d_td,
            d_dw=d_dw,
            d_spa=d_spa,
            num_layer=num_layer,
            use_patch_branch=bool(model_args.get("use_patch_branch", True)),
            use_downsample_branch=bool(model_args.get("use_downsample_branch", True)),
            use_linear_residual_branch=bool(
                model_args.get("use_linear_residual_branch", True)
            ),
            patch_data_input_mode=model_args.get("patch_data_input_mode", "all"),
            patch_embedding_mode=model_args.get("patch_embedding_mode", "serial_concat"),
            # Stage conditioning is controlled by the progressive chain /
            # adapter. Keep KASA's internal prev path enabled so the adapter
            # can inject Z̄_s when requested.
            use_prev_condition=True,
        )

        adapters = []
        for h in self.temporal_resolutions:
            kasa_step = KASATemporalStep(output_len=h, **step_kwargs)
            adapters.append(
                KASATemporalStepAdapter(
                    kasa_step=kasa_step,
                    output_channels=self.output_channels,
                    spatial_codebook=self.spa_codebook,
                )
            )

        self.core = ProgressiveTemporalForecasting(
            temporal_steps=adapters,
            temporal_resolutions=self.temporal_resolutions,
            full_horizon=self.output_len,
            output_channels=self.output_channels,
            use_prev_condition=self.use_prev_condition,
            learnable_stage_scale=self.learnable_stage_scale,
        )

    @property
    def latest_stage_residuals(self):
        return self.core.latest_stage_residuals

    @property
    def latest_stage_predictions(self):
        return self.core.latest_stage_predictions

    @property
    def latest_stage_states(self):
        return self.core.latest_stage_states

    def forward(
        self,
        history_data: torch.Tensor,
        future_data: torch.Tensor = None,
        batch_seen: int = None,
        epoch: int = None,
        train: bool = True,
        return_all: bool = False,
        **kwargs,
    ):
        pred = self.core(
            history_data=history_data,
            future_data=future_data,
            batch_seen=batch_seen,
            epoch=epoch,
            train=train,
            **kwargs,
        )
        if return_all:
            return {
                "prediction": pred,
                "stage_residuals": list(self.latest_stage_residuals),
                "stage_predictions": list(self.latest_stage_predictions),
                "stage_states": list(self.latest_stage_states),
                "temporal_resolutions": list(self.temporal_resolutions),
            }
        return pred
