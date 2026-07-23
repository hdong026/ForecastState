#!/usr/bin/env python3
"""Inspect batch shapes, scaler, mask, and initial chain loss for KASA G1 reproduce."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basicts.utils.constants import BasicTSMode
from basicts.utils.mask import null_val_mask

from forecast_space.wrapper import KasaG1Wrapper, kasa_g1_chain_loss


def load_config(config_path: Path):
    spec = importlib.util.spec_from_file_location("g1_cfg", config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_config() if hasattr(module, "build_config") else module.CFG


def main() -> None:
    config_path = ROOT / "configs" / "g1" / "G1_final_adaptive_PEMS04_12to12_kasa_reproduce.py"
    cfg = load_config(config_path)

    dataset = cfg.dataset_type(
        dataset_name=cfg.dataset_name,
        input_len=cfg.dataset_params["input_len"],
        output_len=cfg.dataset_params["output_len"],
        mode=BasicTSMode.TRAIN,
        data_dir=cfg.dataset_params["data_dir"],
    )
    batch = next(iter(DataLoader(dataset, batch_size=4, shuffle=False)))

    inputs = batch["inputs"]
    targets = batch["targets"]
    future_inputs = batch["future_inputs"]
    print(f"history_data (inputs) shape: {tuple(inputs.shape)}")
    print(f"future_data (future_inputs) shape: {tuple(future_inputs.shape)}")
    print(f"targets shape: {tuple(targets.shape)}")
    print(f"input channels: {inputs.shape[-1]}")
    print(f"target channels: {targets.shape[-1]}")

    scaler = cfg.scaler(norm_each_channel=cfg.norm_each_channel, rescale=cfg.rescale, stats=cfg.stats)
    print(f"scaler type: {scaler.__class__.__name__}")
    print(f"scaler mean: {float(scaler.stats['mean']):.6f}")
    print(f"scaler std: {float(scaler.stats['std']):.6f}")

    null_val = cfg.null_val
    targets_mask = null_val_mask(targets, null_val)
    print(f"null_val (mask value): {null_val}")
    print(f"masked target ratio: {1.0 - targets_mask.float().mean().item():.6f}")

    model = KasaG1Wrapper(cfg.model_config)
    model.eval()
    with torch.no_grad():
        out_eval = model(
            inputs=inputs,
            targets=targets,
            future_inputs=future_inputs,
            train=False,
        )
    print(f"pred shape (eval): {tuple(out_eval['prediction'].shape)}")

    model.train()
    out_train = model(
        inputs=inputs,
        targets=targets,
        future_inputs=future_inputs,
        train=True,
        step=0,
        epoch=1,
    )
    initial_loss = out_train["loss"].item()
    print(f"initial chain loss (train, rescaled MAE weighted): {initial_loss:.6f}")

    fallback = kasa_g1_chain_loss(out_eval["prediction"], targets, targets_mask)
    print(f"fallback MAE (normalized, no postprocess): {fallback.item():.6f}")


if __name__ == "__main__":
    main()
