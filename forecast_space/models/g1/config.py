from dataclasses import dataclass, field
from typing import List, Optional

from basicts.configs import BasicTSModelConfig


@dataclass
class G1FinalAdaptiveConfig(BasicTSModelConfig):
    """Configuration for G1_final_adaptive (forecast-state chain + final adaptive spatial)."""

    num_nodes: int = field(default=307)
    input_len: int = field(default=12)
    output_len: int = field(default=12)
    input_dim: int = field(default=3)
    patch_len: int = field(default=3)
    stride: int = field(default=4)
    td_size: int = field(default=288)
    dw_size: int = field(default=7)
    d_td: int = field(default=32)
    d_dw: int = field(default=32)
    d_d: int = field(default=32)
    d_spa: int = field(default=32)
    num_layer: int = field(default=2)
    if_time_in_day: bool = field(default=True)
    if_day_in_week: bool = field(default=True)
    if_spatial: bool = field(default=True)
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
    use_prev_condition: bool = field(default=True)
    chain_lengths: List[int] = field(default_factory=lambda: [3, 6, 12])
    spatial_placement: str = field(default="final")
    post_spatial_mode: str = field(default="adaptive_only")
