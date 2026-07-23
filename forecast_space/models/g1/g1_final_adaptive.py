"""G1_final_adaptive: forecast-state chain with final adaptive spatial refinement."""

from __future__ import annotations

import torch
from torch import nn

from forecast_space.models.g1.config import G1FinalAdaptiveConfig
from forecast_space.models.g1.final_adaptive_graph import FinalAdaptiveGraph
from forecast_space.models.g1.temporal_step import TemporalStep, interpolate_forecast


class G1FinalAdaptive(nn.Module):
    """Forecast-state chain: X -> Y3 -> Y6 -> Y12, then final adaptive spatial refine."""

    def __init__(self, config: G1FinalAdaptiveConfig):
        super().__init__()
        self.config = config
        self.node_size = config.num_nodes
        self.input_len = config.input_len
        self.output_len = config.output_len
        self.chain_lengths = list(config.chain_lengths)
        self.use_prev_condition = config.use_prev_condition

        if self.chain_lengths[-1] != self.output_len:
            raise ValueError(
                f"Last chain length {self.chain_lengths[-1]} must equal output_len {self.output_len}"
            )

        self.td_codebook = None
        self.dw_codebook = None
        self.spa_codebook = None
        if config.if_time_in_day:
            self.td_codebook = nn.Parameter(torch.empty(config.td_size, config.d_td))
            nn.init.xavier_uniform_(self.td_codebook)
        if config.if_day_in_week:
            self.dw_codebook = nn.Parameter(torch.empty(config.dw_size, config.d_dw))
            nn.init.xavier_uniform_(self.dw_codebook)
        if config.if_spatial:
            self.spa_codebook = nn.Parameter(torch.empty(config.num_nodes, config.d_spa))
            nn.init.xavier_uniform_(self.spa_codebook)

        step_kwargs = dict(
            input_len=config.input_len,
            patch_len=config.patch_len,
            stride=config.stride,
            td_size=config.td_size,
            dw_size=config.dw_size,
            td_codebook=self.td_codebook,
            dw_codebook=self.dw_codebook,
            spa_codebook=self.spa_codebook,
            if_time_in_day=config.if_time_in_day,
            if_day_in_week=config.if_day_in_week,
            if_spatial=config.if_spatial,
            d_d=config.d_d,
            d_td=config.d_td,
            d_dw=config.d_dw,
            d_spa=config.d_spa,
            num_layer=config.num_layer,
            use_patch_branch=config.use_patch_branch,
            use_downsample_branch=config.use_downsample_branch,
            use_linear_residual_branch=config.use_linear_residual_branch,
            patch_data_input_mode=config.patch_data_input_mode,
            patch_embedding_mode=config.patch_embedding_mode,
            use_prev_condition=config.use_prev_condition,
        )

        self.temporal_steps = nn.ModuleList(
            [
                TemporalStep(output_len=k, **step_kwargs)
                for k in self.chain_lengths
            ]
        )

        self.spatial_module = FinalAdaptiveGraph(
            node_size=config.num_nodes,
            input_len=config.input_len,
            d_spa=config.d_spa,
            if_spatial=config.if_spatial,
            spatial_scheme=config.spatial_scheme,
            adj_mx_path=config.adj_mx_path,
            use_gcn=config.use_gcn,
            gcn_hidden_dim=config.gcn_hidden_dim,
            use_dynamic_spatial=config.use_dynamic_spatial,
            dyn_hidden_dim=config.dyn_hidden_dim,
            dyn_topk=config.dyn_topk,
            dyn_tau=config.dyn_tau,
            dyn_static_weight=config.dyn_static_weight,
            use_adaptive_adj=config.use_adaptive_adj,
            adp_hidden_dim=config.adp_hidden_dim,
            adp_topk=config.adp_topk,
            adp_tau=config.adp_tau,
            use_hybrid_graph=config.use_hybrid_graph,
            hybrid_alpha=config.hybrid_alpha,
            post_spatial_mode=config.post_spatial_mode,
        )

    def _spatial_codebook(self):
        return self.spa_codebook

    def _apply_spatial_refine(self, forecast: torch.Tensor, history_data: torch.Tensor) -> torch.Tensor:
        history_flow = history_data[..., 0]
        return self.spatial_module.refine_prediction(forecast, history_flow)

    def _forward_chain(self, history_data: torch.Tensor) -> torch.Tensor:
        spatial_codebook = self._spatial_codebook()
        prev_forecast = None
        temporal_preds = []

        for step_idx, step in enumerate(self.temporal_steps):
            target_len = self.chain_lengths[step_idx]
            prev_up = None
            if prev_forecast is not None and self.use_prev_condition:
                prev_up = interpolate_forecast(prev_forecast, target_len)

            t_k = step(
                history_data,
                prev_forecast=prev_up,
                spatial_codebook=spatial_codebook,
            )
            temporal_preds.append(t_k)
            prev_forecast = t_k

        y_temporal_final = temporal_preds[-1]
        return self._apply_spatial_refine(y_temporal_final, history_data)

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Return prediction tensor [B, output_len, num_nodes, 1]."""
        del targets, kwargs
        return self._forward_chain(inputs)
