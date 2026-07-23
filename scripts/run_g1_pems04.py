#!/usr/bin/env python3
"""Train G1_final_adaptive on PEMS04 12->12 using BasicTS 1.1.0."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basicts.launcher import BasicTSLauncher

try:
    from easydict import EasyDict
    import torch

    torch.serialization.add_safe_globals([EasyDict])
except Exception:
    pass


def load_config(config_path: Path, num_epochs: int | None = None):
    spec = importlib.util.spec_from_file_location("g1_cfg", config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hasattr(module, "build_config"):
        cfg = module.build_config(num_epochs=num_epochs or 100)
    else:
        cfg = module.CFG
        if num_epochs is not None:
            cfg.num_epochs = num_epochs
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        default=str(ROOT / "configs" / "g1" / "G1_final_adaptive_PEMS04_12to12.py"),
    )
    parser.add_argument("--gpus", default="0", help="GPU ids, e.g. 0 or 0,1")
    parser.add_argument("--epochs", type=int, default=None, help="Override num_epochs")
    args = parser.parse_args()

    cfg = load_config(Path(args.config), num_epochs=args.epochs)
    cfg.gpus = args.gpus
    cfg.gpu_num = len(args.gpus.split(","))
    BasicTSLauncher.launch_training(cfg)


if __name__ == "__main__":
    main()
