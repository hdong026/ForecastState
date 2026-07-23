#!/usr/bin/env python3
"""Convert KASA-ST PEMS04 pkl protocol to BasicTS 1.1.0 npy layout (optional)."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_pkl(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def convert(input_len: int, output_len: int, src_dir: Path, dst_dir: Path) -> None:
    data_obj = load_pkl(src_dir / f"data_in{input_len}_out{output_len}.pkl")
    index_obj = load_pkl(src_dir / f"index_in{input_len}_out{output_len}.pkl")
    scaler_obj = load_pkl(src_dir / f"scaler_in{input_len}_out{output_len}.pkl")
    processed = data_obj["processed_data"]

    dst_dir.mkdir(parents=True, exist_ok=True)
    split_map = {"train": "train", "valid": "val", "test": "test"}
    stats = {
        "protocol": "kasa_pkl_sample_split",
        "input_len": input_len,
        "output_len": output_len,
        "scaler": scaler_obj,
        "split_samples": {},
    }

    for src_key, dst_key in split_map.items():
        indices = index_obj[src_key]
        windows = []
        for t0, t1, t2 in indices:
            sample = np.concatenate([processed[t0:t1], processed[t1:t2]], axis=0)
            windows.append(sample)
        arr = np.stack(windows, axis=0).astype(np.float32)
        np.save(dst_dir / f"{dst_key}_data.npy", arr)
        stats["split_samples"][dst_key] = len(indices)

    with open(dst_dir / "protocol_audit.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, default=str)
        f.write("\n")

    print(f"Wrote converted dataset to {dst_dir}")
    print(json.dumps(stats["split_samples"], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-len", type=int, default=12)
    parser.add_argument("--output-len", type=int, default=12)
    parser.add_argument("--src", type=Path, default=ROOT / "datasets" / "PEMS04")
    parser.add_argument("--dst", type=Path, default=ROOT / "datasets" / "PEMS04_latest")
    args = parser.parse_args()
    convert(args.input_len, args.output_len, args.src, args.dst)


if __name__ == "__main__":
    main()
