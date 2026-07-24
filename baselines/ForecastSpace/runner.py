from typing import Dict

import torch

from basicts.runners import SimpleTimeSeriesForecastingRunner

from .arch.chain_loss import compute_legacy_chain_loss


class ForecastSpaceRunner(SimpleTimeSeriesForecastingRunner):
    """Runner for ForecastSpace G1 with chain-weighted training loss."""

    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        param = cfg["MODEL"]["PARAM"]
        self.chain_lengths = list(param.get("chain_lengths", [3, 6, 12]))
        self.chain_loss_weights = list(param.get("chain_loss_weights", [0.2, 0.3, 1.0]))
        if len(self.chain_lengths) != len(self.chain_loss_weights):
            raise ValueError("chain_lengths and chain_loss_weights must have the same length.")
        self.spatial_stage_loss_weights = list(param.get("spatial_stage_loss_weights", [0.0, 0.0, 1.0]))
        self.spatial_graph_loss_weights = list(param.get("spatial_graph_loss_weights", [0.0, 0.0, 0.0]))
        self._last_chain_out = None
        self._last_real_value_norm = None

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
        self._last_chain_out = out
        self._last_real_value_norm = self.select_target_features(future_data)

        model_return = {
            "prediction": out["pred"],
            "inputs": self.select_target_features(history_data),
            "target": self._last_real_value_norm,
        }
        assert list(model_return["prediction"].shape)[:3] == [batch_size, length, num_nodes]
        return self.postprocessing(model_return)

    def train_iters(self, epoch: int, iter_index: int, data: Dict) -> torch.Tensor:
        iter_num = (epoch - 1) * self.iter_per_epoch + iter_index
        forward_return = self.forward(data=data, epoch=epoch, iter_num=iter_num, train=True)

        loss = compute_legacy_chain_loss(
            out=self._last_chain_out,
            real_value=self._last_real_value_norm,
            chain_lengths=self.chain_lengths,
            chain_loss_weights=self.chain_loss_weights,
            scaler=self.scaler,
            metric_forward=self.metric_forward,
            loss_fn=self.loss,
            spatial_stage_loss_weights=self.spatial_stage_loss_weights,
            spatial_graph_loss_weights=self.spatial_graph_loss_weights,
        )

        self.update_epoch_meter("train/loss", loss.item())
        for metric_name, metric_func in self.metrics.items():
            metric_item = self.metric_forward(metric_func, forward_return)
            self.update_epoch_meter(f"train/{metric_name}", metric_item.item())
        return loss
