"""Runner with final-primary gradient projection / weighted-sum / final-only modes."""
from __future__ import annotations

from typing import Dict, Optional

import torch

from basicts.runners import SimpleTimeSeriesForecastingRunner

from .arch.diagnostics import summarize_alphas
from .arch.gradient_projection import (
    aggregate_and_cap_aux,
    compute_param_grads,
    flatten_grads,
    project_auxiliary_grad,
    unflatten_and_set_grads,
)


class ForecastStateGraphResolutionRunner(SimpleTimeSeriesForecastingRunner):
    """
    optimization_mode:
      - final_only
      - weighted_sum
      - gradient_projection  (paper)
    """

    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        param = cfg["MODEL"]["PARAM"]
        self.optimization_mode = str(
            param.get("optimization_mode", cfg.get("OPTIMIZATION_MODE", "gradient_projection"))
        ).lower()
        self.aux_loss_weights = list(param.get("aux_loss_weights", []))
        self.grad_rho = float(param.get("grad_rho", 1.0))
        self.grad_eps = float(param.get("grad_eps", 1e-12))
        self.log_grad_stats = bool(param.get("log_grad_stats", True))
        self._last_out = None
        self._last_target_norm = None
        self._last_grad_stats = None
        # AMP is disabled for gradient projection correctness in v1
        if self.optimization_mode == "gradient_projection":
            self.logger.info(
                "FSGR: optimization_mode=gradient_projection (AMP not used; custom backward)."
            )

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
            "prediction": out["pred"],
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

    def _raw_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_r, tgt_r = self._rescale_pair(pred, target)
        return self.metric_forward(self.loss, {"prediction": pred_r, "target": tgt_r})

    def _compute_losses(self):
        out = self._last_out
        target = self._last_target_norm
        pairs = self.model.scale_matched_pairs(out, target)
        # Final loss L0 on full-resolution prediction
        l0 = self._raw_loss(out["pred"], target[..., : out["pred"].shape[-1]])
        # Aux losses for stages 0..S-2 (paper); if S==1, no aux
        aux_losses = []
        s_total = len(pairs)
        for s in range(max(0, s_total - 1)):
            pred_s, tgt_s, _ = pairs[s]
            aux_losses.append(self._raw_loss(pred_s, tgt_s))
        return l0, aux_losses, pairs

    def train_iters(self, epoch: int, iter_index: int, data: Dict) -> Optional[torch.Tensor]:
        iter_num = (epoch - 1) * self.iter_per_epoch + iter_index
        forward_return = self.forward(data=data, epoch=epoch, iter_num=iter_num, train=True)
        l0, aux_losses, _pairs = self._compute_losses()

        mode = self.optimization_mode
        if mode == "final_only":
            loss = l0
            self.update_epoch_meter("train/loss", loss.item())
            for metric_name, metric_func in self.metrics.items():
                metric_item = self.metric_forward(metric_func, forward_return)
                self.update_epoch_meter(f"train/{metric_name}", metric_item.item())
            return loss

        if mode == "weighted_sum":
            weights = list(self.aux_loss_weights)
            if not weights:
                weights = [0.2] * len(aux_losses)
            while len(weights) < len(aux_losses):
                weights.append(weights[-1])
            loss = l0
            for w, la in zip(weights, aux_losses):
                loss = loss + float(w) * la
            self.update_epoch_meter("train/loss", loss.item())
            for metric_name, metric_func in self.metrics.items():
                metric_item = self.metric_forward(metric_func, forward_return)
                self.update_epoch_meter(f"train/{metric_name}", metric_item.item())
            return loss

        if mode != "gradient_projection":
            raise ValueError(f"Unknown optimization_mode: {mode}")

        # ---- final-primary gradient projection (custom step; return None) ----
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optim.zero_grad(set_to_none=True)

        g0_list = compute_param_grads(l0, params, retain_graph=True)
        g0 = flatten_grads(params, g0_list)

        projected = []
        stage_stats = []
        for s, la in enumerate(aux_losses):
            gs_list = compute_param_grads(la, params, retain_graph=True)
            gs = flatten_grads(params, gs_list)
            g_proj, st = project_auxiliary_grad(gs, g0, eps=self.grad_eps)
            projected.append(g_proj)
            st["stage"] = s
            stage_stats.append(st)

        g_aux, cap_stats = aggregate_and_cap_aux(
            projected, g0, rho=self.grad_rho, eps=self.grad_eps
        )
        g = g0 + g_aux
        unflatten_and_set_grads(params, g)

        if self.clip_grad_param is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), **self.clip_grad_param)
        self.optim.step()

        self._last_grad_stats = {
            "stages": stage_stats,
            "cap": cap_stats,
            "final_grad_norm": float(torch.linalg.norm(g).item()),
            "g0_norm": float(torch.linalg.norm(g0).item()),
            "aux_grad_norm": float(torch.linalg.norm(g_aux).item()),
            "l0": float(l0.item()),
            "aux": [float(a.item()) for a in aux_losses],
            "alphas": summarize_alphas(
                [self.model._alpha(i) for i in range(self.model.num_stages)]
            ),
        }
        if self.log_grad_stats and iter_index == 0:
            self.logger.info(f"FSGR grad_stats epoch={epoch}: {self._last_grad_stats}")

        # meter uses L0 as primary reported loss
        self.update_epoch_meter("train/loss", l0.item())
        for metric_name, metric_func in self.metrics.items():
            metric_item = self.metric_forward(metric_func, forward_return)
            self.update_epoch_meter(f"train/{metric_name}", metric_item.item())
        return None  # skip BaseEpochRunner.backward
