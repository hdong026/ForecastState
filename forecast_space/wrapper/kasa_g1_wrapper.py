"""BasicTS 1.1.0 adapter for KASA ChainForecasting (G1_final_adaptive)."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from forecast_space.models.kasa_g1 import ChainForecasting
from forecast_space.models.kasa_g1.config import KasaG1ModelConfig
from forecast_space.wrapper.chain_loss import compute_legacy_chain_loss


class KasaG1Wrapper(nn.Module):
    """Wrap KASA ChainForecasting; only import paths differ from upstream."""

    def __init__(self, config: KasaG1ModelConfig):
        super().__init__()
        self.config = config
        self.core = ChainForecasting(**config.to_model_args())
        self.chain_lengths = list(config.chain_lengths)
        self.chain_loss_weights = list(config.chain_loss_weights)
        self.spatial_stage_loss_weights = list(config.spatial_stage_loss_weights)
        self.spatial_graph_loss_weights = list(config.spatial_graph_loss_weights)

    def forward(
        self,
        inputs: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        future_inputs: Optional[torch.Tensor] = None,
        train: bool = True,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        **_,
    ) -> dict:
        if future_inputs is None:
            raise ValueError("future_inputs (3-channel future slice) is required for KASA G1.")

        out = self.core(
            history_data=inputs,
            future_data=future_inputs,
            batch_seen=int(step or 0),
            epoch=int(epoch or 0),
            train=train,
            return_all=True,
        )
        prediction = out["pred"]
        if prediction.shape[-1] > 1:
            prediction = prediction[..., :1]

        result = {"prediction": prediction}

        if train and targets is not None:
            chain_loss = compute_legacy_chain_loss(
                out=out,
                real_value=targets,
                chain_lengths=self.chain_lengths,
                chain_loss_weights=self.chain_loss_weights,
                spatial_stage_loss_weights=self.spatial_stage_loss_weights,
                spatial_graph_loss_weights=self.spatial_graph_loss_weights,
                flow_mean=float(self.config.flow_mean),
                flow_std=float(self.config.flow_std),
                null_val=float(self.config.null_val),
            )
            result["loss"] = chain_loss
            result["kasa_g1_chain_loss"] = True
        return result
