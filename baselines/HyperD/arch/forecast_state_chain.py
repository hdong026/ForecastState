"""Forecast-state chain with final adaptive spatial refinement (G1)."""

import torch
import torch.nn.functional as F
from torch import nn

from .final_adaptive_graph import ABCDSpatialModule
from .temporal_step import KASATemporalStep, interpolate_forecast


class ForecastStateChain(nn.Module):
    """Forecast-state chain: X -> Y3 -> Y6 -> Y12 -> final adaptive spatial."""

    def __init__(self, **model_args):
        super().__init__()
        self.node_size = model_args.get("node_size", model_args.get("num_nodes"))
        self.input_len = model_args["input_len"]
        self.input_dim = model_args.get("input_dim", 3)
        self.output_len = model_args["output_len"]
        self.patch_len = model_args["patch_len"]
        self.stride = model_args.get("stride", model_args.get("patch_stride", 4))
        self.td_size = model_args["td_size"]
        self.dw_size = model_args["dw_size"]
        self.d_td = model_args.get("d_td", 32)
        self.d_dw = model_args.get("d_dw", 32)
        self.d_d = model_args.get("d_d", model_args.get("d_model", 32))
        self.d_spa = model_args.get("d_spa", 32)
        self.num_layer = model_args.get("num_layer", 2)

        self.if_time_in_day = model_args.get("if_time_in_day", True)
        self.if_day_in_week = model_args.get("if_day_in_week", True)
        self.if_spatial = model_args.get("if_spatial", True)
        self.spatial_scheme = str(model_args.get("spatial_scheme", "C")).upper()

        self.use_patch_branch = model_args.get("use_patch_branch", True)
        self.use_downsample_branch = model_args.get("use_downsample_branch", True)
        self.use_linear_residual_branch = model_args.get("use_linear_residual_branch", True)
        self.patch_data_input_mode = model_args.get("patch_data_input_mode", "all")
        self.patch_embedding_mode = model_args.get("patch_embedding_mode", "serial_concat")
        self.patch_feature_dim = model_args.get("patch_feature_dim", None)

        self.chain_lengths = list(model_args.get("chain_lengths", [3, 6, 12]))
        self.chain_loss_weights = list(model_args.get("chain_loss_weights", [0.2, 0.3, 1.0]))
        self.use_prev_condition = model_args.get("use_prev_condition", True)
        self.spatial_placement = str(model_args.get("spatial_placement", "final")).lower()
        self.post_spatial_mode = model_args.get("post_spatial_mode", "adaptive_only")
        self.use_pre_temporal_spatial_enhancement = model_args.get(
            "use_pre_temporal_spatial_enhancement", False
        )

        if self.chain_lengths[-1] != self.output_len:
            raise ValueError(
                f"Last chain length {self.chain_lengths[-1]} must equal output_len {self.output_len}"
            )
        if self.spatial_placement not in {"final", "none"}:
            raise ValueError(
                f"Unsupported spatial_placement for ForecastSpace G1: {self.spatial_placement}. "
                "Only 'final' and 'none' are supported in this integration."
            )

        self.td_codebook = None
        self.dw_codebook = None
        self.spa_codebook = None
        if self.if_time_in_day:
            self.td_codebook = nn.Parameter(torch.empty(self.td_size, self.d_td))
            nn.init.xavier_uniform_(self.td_codebook)
        if self.if_day_in_week:
            self.dw_codebook = nn.Parameter(torch.empty(self.dw_size, self.d_dw))
            nn.init.xavier_uniform_(self.dw_codebook)
        if self.if_spatial:
            self.spa_codebook = nn.Parameter(torch.empty(self.node_size, self.d_spa))
            nn.init.xavier_uniform_(self.spa_codebook)

        step_kwargs = dict(
            input_len=self.input_len,
            patch_len=self.patch_len,
            stride=self.stride,
            td_size=self.td_size,
            dw_size=self.dw_size,
            td_codebook=self.td_codebook,
            dw_codebook=self.dw_codebook,
            spa_codebook=self.spa_codebook,
            if_time_in_day=self.if_time_in_day,
            if_day_in_week=self.if_day_in_week,
            if_spatial=self.if_spatial,
            d_d=self.d_d,
            d_td=self.d_td,
            d_dw=self.d_dw,
            d_spa=self.d_spa,
            num_layer=self.num_layer,
            use_patch_branch=self.use_patch_branch,
            use_downsample_branch=self.use_downsample_branch,
            use_linear_residual_branch=self.use_linear_residual_branch,
            patch_data_input_mode=self.patch_data_input_mode,
            patch_embedding_mode=self.patch_embedding_mode,
            patch_feature_dim=self.patch_feature_dim,
            use_prev_condition=self.use_prev_condition,
        )

        self.temporal_steps = nn.ModuleList()
        for k in self.chain_lengths:
            self.temporal_steps.append(KASATemporalStep(output_len=k, **step_kwargs))

        self.spatial_module = ABCDSpatialModule(
            node_size=self.node_size,
            input_len=self.input_len,
            d_spa=self.d_spa,
            if_spatial=self.if_spatial,
            **self._abcd_spatial_kwargs(model_args),
        )

    @staticmethod
    def _abcd_spatial_kwargs(model_args: dict) -> dict:
        return {
            "spatial_scheme": model_args.get("spatial_scheme", "C"),
            "adj_mx_path": model_args.get("adj_mx_path"),
            "use_gcn": model_args.get("use_gcn", False),
            "gcn_hidden_dim": model_args.get("gcn_hidden_dim", 64),
            "use_dynamic_spatial": model_args.get("use_dynamic_spatial", False),
            "dyn_hidden_dim": model_args.get("dyn_hidden_dim", 64),
            "dyn_topk": model_args.get("dyn_topk", 20),
            "dyn_tau": model_args.get("dyn_tau", 0.5),
            "dyn_alpha": model_args.get("dyn_alpha", 0.15),
            "dyn_static_weight": model_args.get("dyn_static_weight", 0.2),
            "use_adaptive_adj": model_args.get("use_adaptive_adj", True),
            "adp_hidden_dim": model_args.get("adp_hidden_dim", 32),
            "adp_topk": model_args.get("adp_topk", 20),
            "adp_tau": model_args.get("adp_tau", 0.5),
            "adp_alpha": model_args.get("adp_alpha", 0.1),
            "use_hybrid_graph": model_args.get("use_hybrid_graph", False),
            "hybrid_alpha": model_args.get("hybrid_alpha", 0.2),
            "use_lightweight_spatial": model_args.get("use_lightweight_spatial", False),
            "light_alpha": model_args.get("light_alpha", 0.05),
            "post_spatial_mode": model_args.get("post_spatial_mode", "adaptive_only"),
            "adaptive_ms_topks": model_args.get("adaptive_ms_topks", [8, 16, 32]),
            "adaptive_ms_alpha": model_args.get("adaptive_ms_alpha", 0.10),
            "adaptive_ms_fusion": model_args.get("adaptive_ms_fusion", "softmax"),
            "adaptive_ms_share_logits": model_args.get("adaptive_ms_share_logits", True),
            "adaptive_ms_init": model_args.get("adaptive_ms_init", "favor_largest"),
        }

    @staticmethod
    def pool_target(future_target: torch.Tensor, target_len: int) -> torch.Tensor:
        target = future_target[..., :1]
        batch_size, future_len, num_nodes, _ = target.shape
        if future_len % target_len == 0:
            group = future_len // target_len
            return target.reshape(batch_size, target_len, group, num_nodes, 1).mean(dim=2)
        x = target.permute(0, 2, 3, 1).reshape(batch_size * num_nodes, 1, future_len)
        x = F.adaptive_avg_pool1d(x, target_len)
        return x.reshape(batch_size, num_nodes, 1, target_len).permute(0, 3, 1, 2)

    def _spatial_codebook(self):
        if self.use_pre_temporal_spatial_enhancement:
            return self.spatial_module.get_enhanced_spatial_embedding(self.spa_codebook)
        return self.spa_codebook

    def _apply_spatial_refine(self, forecast: torch.Tensor, history_data: torch.Tensor) -> torch.Tensor:
        history_flow = history_data[..., 0]
        return self.spatial_module.refine_prediction(forecast, history_flow)

    def _forward_chain(
        self, history_data: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        spatial_codebook = self._spatial_codebook()
        chain_preds: list[torch.Tensor] = []
        temporal_preds: list[torch.Tensor] = []
        prev_forecast = None

        for step_idx, step in enumerate(self.temporal_steps):
            target_len = self.chain_lengths[step_idx]
            prev_up = None
            if prev_forecast is not None and self.use_prev_condition:
                prev_up = interpolate_forecast(prev_forecast, target_len)

            t_k = step(history_data, prev_forecast=prev_up, spatial_codebook=spatial_codebook)
            temporal_preds.append(t_k)
            chain_preds.append(t_k)
            prev_forecast = t_k

        y_temporal_final = temporal_preds[-1]
        y_final = y_temporal_final
        if self.spatial_placement == "final":
            y_final = self._apply_spatial_refine(y_temporal_final, history_data)

        return y_final, chain_preds, temporal_preds

    def forward(
        self,
        history_data: torch.Tensor,
        future_data: torch.Tensor = None,
        batch_seen: int = 0,
        epoch: int = 0,
        train: bool = False,
        return_all: bool = False,
        **kwargs,
    ):
        del future_data, batch_seen, epoch, train, kwargs
        y_final, chain_preds, temporal_preds = self._forward_chain(history_data)
        if return_all:
            return {
                "pred": y_final,
                "chain_preds": chain_preds,
                "temporal_preds": temporal_preds,
                "temporal_stage_preds": temporal_preds,
                "spatial_preds": chain_preds,
                "spatial_stage_preds": [],
                "final_temporal_pred": temporal_preds[-1],
                "chain_lengths": list(self.chain_lengths),
                "spatial_organization_type": "final_only",
            }
        return y_final
