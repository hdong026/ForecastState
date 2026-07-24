#!/usr/bin/env python3
"""One-batch overfit sanity check for ForecastStateGraphResolution."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines.ForecastStateGraphResolution.arch.model import ForecastStateGraphResolution


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default="experiments/overfit_one_batch.json")
    args = parser.parse_args()

    device = torch.device(args.device)
    n = 307
    adj_path = ROOT / "datasets/PEMS04/adj_mx.pkl"
    cache_dir = ROOT / "datasets/PEMS04/forecast_state_graph_cache"

    model = ForecastStateGraphResolution(
        num_nodes=n,
        input_len=12,
        pred_len=12,
        input_dim=3,
        output_dim=1,
        hidden_dim=64,
        num_encoder_layers=2,
        dropout=0.0,
        resolution_schedule=[
            {"h": 3, "capacity": 4},
            {"h": 6, "capacity": 2},
            {"h": 12, "capacity": 1},
        ],
        initial_forecast="linear_head",
        use_graph=True,
        adp_topk=8,
        dataset_name="PEMS04",
        adj_mx_path=str(adj_path),
        graph_cache_dir=str(cache_dir),
        clustering_seed=1,
        print_stage_shapes=True,
    ).to(device)

    # Fixed synthetic batch with learnable structure
    torch.manual_seed(0)
    x = torch.randn(8, 12, n, 3, device=device)
    # target = linear transform of last history step (easy)
    y = x[:, -1:, :, :1].expand(-1, 12, -1, -1).contiguous() + 0.1 * torch.randn(8, 12, n, 1, device=device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = []
    for step in range(args.steps):
        opt.zero_grad(set_to_none=True)
        out = model(history_data=x, return_all=True)
        # final + scale-matched aux (weighted for overfit simplicity)
        l0 = (out["pred"] - y).abs().mean()
        loss = l0
        pairs = model.scale_matched_pairs(out, y)
        stage_losses = []
        for s, (ps, ts, _) in enumerate(pairs[:-1]):
            ls = (ps - ts).abs().mean()
            stage_losses.append(float(ls.item()))
            loss = loss + 0.2 * ls
        loss.backward()
        # check grads
        graph_grad = float(model.graph_ops[-1].emb_src.grad.norm().item()) if model.graph_ops[-1].emb_src.grad is not None else 0.0
        alpha_grad = float(model.raw_alphas[-1].grad.norm().item()) if model.raw_alphas[-1].grad is not None else 0.0
        opt.step()
        row = {
            "step": step,
            "loss": float(loss.item()),
            "l0": float(l0.item()),
            "stage_losses": stage_losses,
            "alphas": [float(model._alpha(i).item()) for i in range(model.num_stages)],
            "graph_grad_norm": graph_grad,
            "alpha_grad_norm": alpha_grad,
        }
        history.append(row)
        if step % 50 == 0 or step == args.steps - 1:
            print(row)

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "start_loss": history[0]["loss"],
        "end_loss": history[-1]["loss"],
        "improved": history[-1]["loss"] < history[0]["loss"] * 0.5,
        "history": history[:: max(1, len(history) // 20)] + [history[-1]],
    }
    out_path.write_text(json.dumps(summary, indent=2))
    print("Wrote", out_path)
    print("improved_strongly", summary["improved"], "start", summary["start_loss"], "end", summary["end_loss"])
    if not summary["improved"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
