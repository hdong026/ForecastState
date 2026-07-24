"""Optional KASA-backed residual heads for resolution stages."""
from __future__ import annotations

import torch
from torch import nn

from baselines.ForecastSpace.arch.temporal_step import KASATemporalStep


class KASABackboneBundle(nn.Module):
    """Shared codebooks + factory for stage-wise KASATemporalStep."""

    def __init__(
        self,
        num_nodes: int,
        input_len: int,
        patch_len: int = 3,
        stride: int = 4,
        td_size: int = 288,
        dw_size: int = 7,
        d_td: int = 32,
        d_dw: int = 32,
        d_d: int = 32,
        d_spa: int = 32,
        num_layer: int = 2,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_len = input_len
        self.patch_len = patch_len
        self.stride = stride
        self.td_size = td_size
        self.dw_size = dw_size
        self.d_td = d_td
        self.d_dw = d_dw
        self.d_d = d_d
        self.d_spa = d_spa
        self.num_layer = num_layer

        self.td_codebook = nn.Parameter(torch.empty(td_size, d_td))
        self.dw_codebook = nn.Parameter(torch.empty(dw_size, d_dw))
        self.spa_codebook = nn.Parameter(torch.empty(num_nodes, d_spa))
        nn.init.xavier_uniform_(self.td_codebook)
        nn.init.xavier_uniform_(self.dw_codebook)
        nn.init.xavier_uniform_(self.spa_codebook)

    def make_step(self, output_len: int, spa_codebook: nn.Parameter | None = None) -> KASATemporalStep:
        spa = spa_codebook if spa_codebook is not None else self.spa_codebook
        return KASATemporalStep(
            output_len=output_len,
            input_len=self.input_len,
            patch_len=self.patch_len,
            stride=self.stride,
            td_size=self.td_size,
            dw_size=self.dw_size,
            td_codebook=self.td_codebook,
            dw_codebook=self.dw_codebook,
            spa_codebook=spa,
            if_time_in_day=True,
            if_day_in_week=True,
            if_spatial=True,
            d_d=self.d_d,
            d_td=self.d_td,
            d_dw=self.d_dw,
            d_spa=self.d_spa,
            num_layer=self.num_layer,
            use_prev_condition=True,
        )
