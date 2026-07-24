"""Forward / gradient smoke tests for ForecastStateGraphResolution."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines.ForecastStateGraphResolution.arch.model import ForecastStateGraphResolution


def _tiny_adj(path: Path, n: int = 12):
    a = np.eye(n, dtype=np.float32)
    for i in range(n - 1):
        a[i, i + 1] = 0.8
        a[i + 1, i] = 0.8
    import pickle

    with open(path, "wb") as f:
        pickle.dump(a, f)


def _build_model(tmpdir: Path, schedule=None, use_graph=True):
    n = 12
    adj_path = tmpdir / "adj.pkl"
    _tiny_adj(adj_path, n)
    schedule = schedule or [
        {"h": 3, "capacity": 4},
        {"h": 6, "capacity": 2},
        {"h": 12, "capacity": 1},
    ]
    model = ForecastStateGraphResolution(
        num_nodes=n,
        input_len=12,
        pred_len=12,
        input_dim=3,
        output_dim=1,
        hidden_dim=16,
        num_encoder_layers=1,
        dropout=0.0,
        resolution_schedule=schedule,
        initial_forecast="linear_head",
        use_graph=use_graph,
        adp_topk=2,
        dataset_name="TINY",
        adj_mx_path=str(adj_path),
        graph_cache_dir=str(tmpdir / "cache"),
        clustering_seed=1,
        print_stage_shapes=False,
        use_kasa_stages=False,
    )
    return model


def test_forward_shapes_finite():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        model = _build_model(tmp)
        x = torch.randn(2, 12, 12, 3)
        out = model(history_data=x, return_all=True)
        assert out["pred"].shape == (2, 12, 12, 1)
        assert torch.isfinite(out["pred"]).all()
        assert len(out["stage_full_preds"]) == 3


def test_graph_and_residual_grads():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        model = _build_model(tmp, use_graph=True)
        # Zero-init residual heads intentionally block first-step graph grads;
        # perturb heads so graph path is active for this check.
        for pred in model.residual_predictors:
            nn_init = torch.nn.init
            nn_init.xavier_uniform_(pred.head.weight)
            nn_init.zeros_(pred.head.bias)
        x = torch.randn(2, 12, 12, 3)
        y = torch.randn(2, 12, 12, 1)
        out = model(history_data=x, return_all=True)
        loss = (out["pred"] - y).abs().mean()
        loss.backward()
        # residual head
        assert model.residual_predictors[0].head.weight.grad is not None
        assert model.residual_predictors[0].head.weight.grad.abs().sum() > 0
        # graph adaptive embeddings
        assert model.graph_ops[0].emb_src.grad is not None
        gnorm = model.graph_ops[0].emb_src.grad.norm().item()
        assert gnorm > 0
        # temporal encoder
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.history_encoder.parameters())


def test_temporal_only_schedule():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        schedule = [
            {"h": 3, "capacity": 1},
            {"h": 6, "capacity": 1},
            {"h": 12, "capacity": 1},
        ]
        model = _build_model(tmp, schedule=schedule, use_graph=False)
        x = torch.randn(2, 12, 12, 3)
        pred = model(history_data=x)
        assert pred.shape == (2, 12, 12, 1)


if __name__ == "__main__":
    test_forward_shapes_finite()
    test_graph_and_residual_grads()
    test_temporal_only_schedule()
    print("test_forecast_state_forward: OK")
