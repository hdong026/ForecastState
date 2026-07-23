#!/usr/bin/env python3
"""Smoke test for G1FinalAdaptive forward/backward."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from forecast_space.models.g1 import G1FinalAdaptive, G1FinalAdaptiveConfig


def main() -> None:
    cfg = G1FinalAdaptiveConfig(
        num_nodes=307,
        input_len=12,
        output_len=12,
        patch_len=3,
        chain_lengths=[3, 6, 12],
        use_prev_condition=True,
        spatial_placement="final",
        post_spatial_mode="adaptive_only",
        adj_mx_path=str(ROOT / "datasets" / "PEMS04" / "adj_mx.pkl"),
    )
    model = G1FinalAdaptive(cfg)
    x = torch.randn(4, 12, 307, 3, requires_grad=True)
    y = model(x)
    assert y.shape == (4, 12, 307, 1), f"unexpected shape: {tuple(y.shape)}"
    loss = y.abs().mean()
    loss.backward()
    print("smoke test passed")
    print(f"output shape: {tuple(y.shape)}")


if __name__ == "__main__":
    main()
