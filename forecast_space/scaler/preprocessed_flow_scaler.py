"""Scaler for pre-normalized KASA pkl flow data."""

from __future__ import annotations

from typing import Union

import numpy as np
import torch

from basicts.scaler.base_scaler import BasicTSScaler


class PreprocessedFlowScaler(BasicTSScaler):
    """Identity transform on normalized data; inverse restores raw flow scale."""

    def __init__(self, norm_each_channel: bool = False, rescale: bool = True, stats: dict | None = None):
        super().__init__(norm_each_channel, rescale, stats or {})
        self._ensure_tensor_stats()

    def fit(self, data: Union[np.ndarray, torch.Tensor]) -> None:
        if self.stats.get("mean") is not None and self.stats.get("std") is not None:
            self._ensure_tensor_stats()
            return
        if isinstance(data, np.ndarray):
            mean = float(np.mean(data))
            std = float(np.std(data))
        else:
            mean = float(torch.mean(data).item())
            std = float(torch.std(data).item())
        if std < 1e-6:
            std = 1.0
        self.stats["mean"] = torch.tensor(mean)
        self.stats["std"] = torch.tensor(std)

    def _ensure_tensor_stats(self) -> None:
        mean = self.stats["mean"]
        std = self.stats["std"]
        if not isinstance(mean, torch.Tensor):
            self.stats["mean"] = torch.tensor(float(mean))
        if not isinstance(std, torch.Tensor):
            self.stats["std"] = torch.tensor(float(std))

    def transform(self, input_data: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        del mask
        return input_data

    def inverse_transform(self, input_data: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        self._ensure_tensor_stats()
        mean = self.stats["mean"].to(input_data.device)
        std = self.stats["std"].to(input_data.device)
        denormed = input_data * std + mean
        if mask is not None:
            denormed = torch.where(mask, denormed, input_data)
        return denormed
