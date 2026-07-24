"""Runner for temporal-only progressive forecasting.

Supports final-only training and optional scale-matched auxiliary loss.
Does NOT implement gradient projection.
"""
from __future__ import annotations

from typing import Dict

import torch

from basicts.runners import SimpleTimeSeriesForecastingRunner

from .losses import progressive_temporal_loss


class ForecastStateProgressiveRunner(SimpleTimeSeriesForecastingRunner):
    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        param = cfg["MODEL"]["PARAM"]
        self.temporal_resolutions = list(param.get("temporal_resolutions", [3, 6, 12]))
        self.aux_loss_weight = float(param.get("aux_loss_weight", 0.0))
        self._last_out = None
        self._last_target_norm = None

    def forward(self, data: Dict, epoch: int = None, iter_num: int = None, train: bool = True, **kwargs):
        data = self.preprocessing(data)
        future_data, history_data = data["target"], data["inputs"]
        history_data = self.to_running_device(history_data)
        future_data = self.to_running_device(future_data)
        batch_size, length, num_nodes, _ = future_data.shape

        history_data = self.select_input_features(history_data)
        future_data_4_dec = self.select_input_features(future_data)
        if not train:
            future_data_4_dec[..., 0] = torch.empty_like(future_data_4_dec[..., 0])

        out = self.model(
            history_data=history_data,
            future_data=future_data_4_dec,
            batch_seen=iter_num,
            epoch=epoch,
            train=train,
            return_all=True,
        )
        self._last_out = out
        self._last_target_norm = self.select_target_features(future_data)

        model_return = {
            "prediction": out["prediction"],
            "inputs": self.select_target_features(history_data),
            "target": self._last_target_norm,
        }
        assert list(model_return["prediction"].shape)[:3] == [batch_size, length, num_nodes]
        return self.postprocessing(model_return)

    def _rescale_pair(self, pred: torch.Tensor, target: torch.Tensor):
        if self.scaler is not None and getattr(self.scaler, "rescale", False):
            pred = self.scaler.inverse_transform(pred.clone())
            target = self.scaler.inverse_transform(target.clone())
        return pred, target

    def train_iters(self, epoch: int, iter_index: int, data: Dict) -> torch.Tensor:
        iter_num = (epoch - 1) * self.iter_per_epoch + iter_index
        forward_return = self.forward(data=data, epoch=epoch, iter_num=iter_num, train=True)

        # Loss on rescaled tensors to match BasicTS metric convention
        pred = forward_return["prediction"]
        target = forward_return["target"]

        # Rebuild stage predictions in the same scale as pred/target.
        # Stage tensors are stored in normalized model space; rescale if needed.
        stage_preds = list(self._last_out["stage_predictions"])
        if self.scaler is not None and getattr(self.scaler, "rescale", False):
            stage_preds_rescaled = [
                self.scaler.inverse_transform(sp.clone()) for sp in stage_preds
            ]
        else:
            stage_preds_rescaled = stage_preds

        loss = progressive_temporal_loss(
            prediction=pred,
            target=target,
            stage_predictions=stage_preds_rescaled,
            temporal_resolutions=self.temporal_resolutions,
            masked_mae_fn=self.loss,
            null_val=self.null_val,
            aux_weight=self.aux_loss_weight,
        )

        self.update_epoch_meter("train/loss", loss.item())
        for metric_name, metric_func in self.metrics.items():
            metric_item = self.metric_forward(metric_func, forward_return)
            self.update_epoch_meter(f"train/{metric_name}", metric_item.item())
        return loss
