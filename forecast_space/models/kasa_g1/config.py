from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from basicts.configs import BasicTSModelConfig


@dataclass
class KasaG1ModelConfig(BasicTSModelConfig):
    """KASA G1_final_adaptive ChainForecasting parameters (aligned with KASA-ST config)."""

    num_nodes: int = field(default=307)
    input_len: int = field(default=12)
    output_len: int = field(default=12)
    input_dim: int = field(default=3)
    main_input_dim: int = field(default=3)
    patch_len: int = field(default=3)
    stride: int = field(default=4)
    td_size: int = field(default=288)
    dw_size: int = field(default=7)
    d_td: int = field(default=32)
    d_dw: int = field(default=32)
    d_d: int = field(default=32)
    d_spa: int = field(default=32)
    if_time_in_day: bool = field(default=True)
    if_day_in_week: bool = field(default=True)
    if_spatial: bool = field(default=True)
    num_layer: int = field(default=2)
    spatial_scheme: str = field(default="C")
    adj_mx_path: Optional[str] = field(default=None)
    use_gcn: bool = field(default=True)
    gcn_hidden_dim: int = field(default=64)
    use_dynamic_spatial: bool = field(default=True)
    dyn_hidden_dim: int = field(default=64)
    dyn_topk: int = field(default=20)
    dyn_tau: float = field(default=0.5)
    dyn_static_weight: float = field(default=0.2)
    use_adaptive_adj: bool = field(default=True)
    adp_hidden_dim: int = field(default=32)
    adp_topk: int = field(default=20)
    adp_tau: float = field(default=0.5)
    use_hybrid_graph: bool = field(default=True)
    hybrid_alpha: float = field(default=0.2)
    use_patch_branch: bool = field(default=True)
    use_downsample_branch: bool = field(default=True)
    use_linear_residual_branch: bool = field(default=True)
    patch_embedding_mode: str = field(default="serial_concat")
    patch_data_input_mode: str = field(default="all")
    post_spatial_mode: str = field(default="adaptive_only")
    spatial_placement: str = field(default="final")
    use_pre_temporal_spatial_enhancement: bool = field(default=False)
    keep_output_prior_residual: bool = field(default=False)
    use_input_prior_enhancement: bool = field(default=False)
    use_graph_spectral_calibration: bool = field(default=False)
    use_extra_prior_input: bool = field(default=False)
    use_prev_condition: bool = field(default=True)
    chain_lengths: List[int] = field(default_factory=lambda: [3, 6, 12])
    chain_loss_weights: List[float] = field(default_factory=lambda: [0.2, 0.3, 1.0])
    spatial_graph_loss_weights: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    spatial_stage_loss_weights: List[float] = field(default_factory=lambda: [0.0, 0.0, 1.0])
    unified_aux_loss_mode: str = field(default="none")
    variant_name: str = field(default="G1_final_adaptive")
    flow_mean: float = field(default=0.0)
    flow_std: float = field(default=1.0)
    null_val: float = field(default=0.0)

    def to_model_args(self) -> dict:
        return {
            "node_size": self.num_nodes,
            "input_len": self.input_len,
            "output_len": self.output_len,
            "input_dim": self.input_dim,
            "main_input_dim": self.main_input_dim,
            "patch_len": self.patch_len,
            "stride": self.stride,
            "td_size": self.td_size,
            "dw_size": self.dw_size,
            "d_td": self.d_td,
            "d_dw": self.d_dw,
            "d_d": self.d_d,
            "d_spa": self.d_spa,
            "if_time_in_day": self.if_time_in_day,
            "if_day_in_week": self.if_day_in_week,
            "if_spatial": self.if_spatial,
            "num_layer": self.num_layer,
            "spatial_scheme": self.spatial_scheme,
            "adj_mx_path": self.adj_mx_path,
            "use_gcn": self.use_gcn,
            "gcn_hidden_dim": self.gcn_hidden_dim,
            "use_dynamic_spatial": self.use_dynamic_spatial,
            "dyn_hidden_dim": self.dyn_hidden_dim,
            "dyn_topk": self.dyn_topk,
            "dyn_tau": self.dyn_tau,
            "dyn_static_weight": self.dyn_static_weight,
            "use_adaptive_adj": self.use_adaptive_adj,
            "adp_hidden_dim": self.adp_hidden_dim,
            "adp_topk": self.adp_topk,
            "adp_tau": self.adp_tau,
            "use_hybrid_graph": self.use_hybrid_graph,
            "hybrid_alpha": self.hybrid_alpha,
            "use_patch_branch": self.use_patch_branch,
            "use_downsample_branch": self.use_downsample_branch,
            "use_linear_residual_branch": self.use_linear_residual_branch,
            "patch_embedding_mode": self.patch_embedding_mode,
            "patch_data_input_mode": self.patch_data_input_mode,
            "post_spatial_mode": self.post_spatial_mode,
            "spatial_placement": self.spatial_placement,
            "use_pre_temporal_spatial_enhancement": self.use_pre_temporal_spatial_enhancement,
            "keep_output_prior_residual": self.keep_output_prior_residual,
            "use_input_prior_enhancement": self.use_input_prior_enhancement,
            "use_graph_spectral_calibration": self.use_graph_spectral_calibration,
            "use_extra_prior_input": self.use_extra_prior_input,
            "use_prev_condition": self.use_prev_condition,
            "chain_lengths": list(self.chain_lengths),
            "chain_loss_weights": list(self.chain_loss_weights),
            "spatial_graph_loss_weights": list(self.spatial_graph_loss_weights),
            "spatial_stage_loss_weights": list(self.spatial_stage_loss_weights),
            "unified_aux_loss_mode": self.unified_aux_loss_mode,
            "variant_name": self.variant_name,
        }
