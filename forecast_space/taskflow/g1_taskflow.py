"""Task flow matching KASA pkl protocol (no input re-normalization)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

import torch

from basicts.runners.taskflow import BasicTSForecastingTaskFlow
from basicts.utils.mask import null_val_mask

if TYPE_CHECKING:
    from basicts.runners.basicts_runner import BasicTSRunner


class G1PklTaskFlow(BasicTSForecastingTaskFlow):
    """Skip transform on already-normalized pkl inputs; inverse only for metrics."""

    def preprocess(self, runner: "BasicTSRunner", data: Dict[str, Any]) -> Dict[str, Any]:
        inputs_mask = null_val_mask(data["inputs"], runner.cfg.null_val)
        targets_mask = null_val_mask(data["targets"], runner.cfg.null_val)
        data["inputs"] = torch.where(
            inputs_mask,
            data["inputs"],
            torch.tensor(runner.cfg.null_to_num, device=data["inputs"].device),
        )
        data["targets"] = torch.where(
            targets_mask,
            data["targets"],
            torch.tensor(runner.cfg.null_to_num, device=data["targets"].device),
        )
        data["targets_mask"] = targets_mask
        return data

    def postprocess(self, runner: "BasicTSRunner", forward_return: Dict[str, Any]) -> Dict[str, Any]:
        if runner.cfg.rescale and runner.scaler is not None:
            forward_return["prediction"] = runner.scaler.inverse_transform(forward_return["prediction"])
            forward_return["targets"] = runner.scaler.inverse_transform(
                forward_return["targets"], forward_return["targets_mask"]
            )
        return forward_return
