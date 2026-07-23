"""PEMS04 dataset loader using KASA-ST pkl protocol (sample-based 6:2:2 split)."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Union

import numpy as np
import torch
from torch.utils.data import Dataset

from basicts.utils.constants import BasicTSMode


class Pems04PklDataset(Dataset):
    """Load PEMS04 from legacy pkl files to preserve KASA-ST split protocol."""

    def __init__(
        self,
        dataset_name: str,
        input_len: int,
        output_len: int,
        mode: Union[BasicTSMode, str],
        data_dir: str | None = None,
        **_,
    ) -> None:
        super().__init__()
        self.dataset_name = dataset_name
        self.input_len = input_len
        self.output_len = output_len
        self.mode = str(mode).lower()
        if self.mode not in {"train", "val", "valid", "test"}:
            raise ValueError(f"Unsupported mode: {mode}")
        if self.mode == "valid":
            self.mode = "val"

        root = Path(data_dir or f"datasets/{dataset_name}")
        data_path = root / f"data_in{input_len}_out{output_len}.pkl"
        index_path = root / f"index_in{input_len}_out{output_len}.pkl"
        if not data_path.is_file() or not index_path.is_file():
            raise FileNotFoundError(
                f"Missing pkl dataset under {root}. "
                f"Expected {data_path.name} and {index_path.name}."
            )

        with open(data_path, "rb") as f:
            processed = pickle.load(f)["processed_data"]
        with open(index_path, "rb") as f:
            index_obj = pickle.load(f)

        mode_key = "valid" if self.mode == "val" else self.mode
        self._data = torch.from_numpy(processed).float()
        self._index = index_obj[mode_key]

    @property
    def data(self) -> np.ndarray:
        """Train-flow values for scaler fitting (channel 0, normalized)."""
        if self.mode != "train":
            raise RuntimeError("data property is only defined for train split.")
        values = []
        for t0, t1, t2 in self._index:
            values.append(self._data[t0:t2, :, 0].numpy())
        return np.concatenate(values, axis=0)

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, index: int) -> dict:
        t0, t1, t2 = self._index[index]
        history = self._data[t0:t1]
        future = self._data[t1:t2]
        return {
            "inputs": history.clone(),
            "targets": future[..., :1].clone(),
        }
