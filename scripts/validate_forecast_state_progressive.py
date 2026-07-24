"""E0/E1/E2 validation: shapes, gradients, contribution, BasicTS dry-run."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines.ForecastState.model import ForecastStateProgressive
from baselines.ForecastState.temporal_ops import temporal_lift, temporal_project


def _build_model(resolutions, use_prev_condition=True):
    return ForecastStateProgressive(
        num_nodes=307,
        input_len=12,
        output_len=12,
        output_dim=1,
        temporal_resolutions=resolutions,
        use_prev_condition=use_prev_condition,
        learnable_stage_scale=False,
        aux_loss_weight=0.0,
        patch_len=3,
        stride=4,
        td_size=288,
        dw_size=7,
        d_td=32,
        d_dw=32,
        d_d=32,
        d_spa=32,
        num_layer=2,
        if_time_in_day=True,
        if_day_in_week=True,
        if_spatial=True,
    )


def test_temporal_ops():
    y = torch.ones(2, 12, 5, 1) * 4.0
    z3 = temporal_project(y, 3)
    z6 = temporal_project(y, 6)
    assert z3.shape == (2, 3, 5, 1)
    assert z6.shape == (2, 6, 5, 1)
    assert torch.allclose(z3, torch.ones_like(z3) * 4.0)
    y12 = temporal_lift(z3, 12)
    assert y12.shape == (2, 12, 5, 1)
    assert torch.allclose(y12, torch.ones_like(y12) * 4.0)
    assert torch.allclose(temporal_project(y, 12), y)
    assert torch.allclose(temporal_lift(y, 12), y)
    print("temporal_ops: OK")


def test_forward_shapes():
    # KASA TemporalStep requires time/day channels; use C_in=3.
    history_data = torch.randn(2, 12, 307, 3)
    model = _build_model([3, 6, 12], use_prev_condition=True)
    prediction = model(history_data=history_data)

    assert prediction.shape == (2, 12, 307, 1)
    assert len(model.latest_stage_residuals) == 3
    assert len(model.latest_stage_predictions) == 3
    assert len(model.latest_stage_states) == 3

    assert model.latest_stage_residuals[0].shape == (2, 3, 307, 1)
    assert model.latest_stage_residuals[1].shape == (2, 6, 307, 1)
    assert model.latest_stage_residuals[2].shape == (2, 12, 307, 1)

    assert model.latest_stage_predictions[0].shape == (2, 12, 307, 1)
    assert model.latest_stage_predictions[1].shape == (2, 12, 307, 1)
    assert model.latest_stage_predictions[2].shape == (2, 12, 307, 1)

    assert model.latest_stage_states[0].shape == (2, 3, 307, 1)
    assert model.latest_stage_states[1].shape == (2, 6, 307, 1)
    assert model.latest_stage_states[2].shape == (2, 12, 307, 1)
    print("forward_shapes: OK", tuple(prediction.shape))


def test_backward_gradients():
    history_data = torch.randn(2, 12, 307, 3)
    model = _build_model([3, 6, 12], use_prev_condition=True)
    prediction = model(history_data=history_data)
    target = torch.randn_like(prediction)
    loss = torch.mean(torch.abs(prediction - target))
    loss.backward()

    for stage_idx, adapter in enumerate(model.core.temporal_steps):
        grads = [
            p.grad
            for p in adapter.kasa_step.parameters()
            if p.requires_grad
        ]
        assert grads, f"stage {stage_idx} has no trainable parameters"
        assert any(g is not None for g in grads), f"stage {stage_idx} has all-None grads"
        for g in grads:
            if g is not None:
                assert torch.isfinite(g).all(), f"stage {stage_idx} has non-finite grad"
        nonzero = sum(float(g.abs().sum()) for g in grads if g is not None)
        assert nonzero > 0.0, f"stage {stage_idx} has zero gradients"
        print(f"stage {stage_idx} (h={model.temporal_resolutions[stage_idx]}): grad_l1={nonzero:.4f}")
    print("backward_gradients: OK")


def test_contribution():
    history_data = torch.randn(2, 12, 307, 3)
    model = _build_model([3, 6, 12], use_prev_condition=True)
    model.eval()
    with torch.no_grad():
        base = model(history_data=history_data).clone()
        # Zero T3 residual contribution by temporarily replacing stage scale
        scales = model.core.stage_scales.clone()
        model.core.stage_scales[0] = 0.0
        no_t3 = model(history_data=history_data).clone()
        model.core.stage_scales.copy_(scales)
        model.core.stage_scales[1] = 0.0
        no_t6 = model(history_data=history_data).clone()
        model.core.stage_scales.copy_(scales)

    assert not torch.allclose(base, no_t3), "T3 should affect final prediction"
    assert not torch.allclose(base, no_t6), "T6 should affect final prediction"
    print("contribution: OK")


def test_e0_single_stage():
    history_data = torch.randn(2, 12, 307, 3)
    model = _build_model([12], use_prev_condition=False)
    pred = model(history_data=history_data)
    assert pred.shape == (2, 12, 307, 1)
    assert len(model.latest_stage_residuals) == 1
    assert model.latest_stage_residuals[0].shape == (2, 12, 307, 1)
    print("e0_single_stage: OK")


def test_config_imports():
    from baselines.ForecastState.PEMS04_E0 import CFG as CFG0
    from baselines.ForecastState.PEMS04_E1 import CFG as CFG1
    from baselines.ForecastState.PEMS04_E2 import CFG as CFG2
    assert CFG0.MODEL.PARAM["temporal_resolutions"] == [12]
    assert CFG0.MODEL.PARAM["use_prev_condition"] is False
    assert CFG0.MODEL.PARAM["aux_loss_weight"] == 0.0
    assert CFG1.MODEL.PARAM["temporal_resolutions"] == [3, 6, 12]
    assert CFG1.MODEL.PARAM["use_prev_condition"] is True
    assert CFG1.MODEL.PARAM["aux_loss_weight"] == 0.0
    assert CFG2.MODEL.PARAM["temporal_resolutions"] == [3, 6, 12]
    assert CFG2.MODEL.PARAM["aux_loss_weight"] == 0.1
    print("config_import E0/E1/E2: OK")


def test_hyperd_config_still_imports():
    from baselines.HyperD import PEMS04 as hyperd_cfg
    assert hasattr(hyperd_cfg, "CFG")
    print("hyperd_config_import: OK")


def test_basicts_dry_run():
    from basicts.data import TimeSeriesForecastingDataset
    from basicts.scaler import ZScoreScaler
    from torch.utils.data import DataLoader

    dataset = TimeSeriesForecastingDataset(
        dataset_name="PEMS04",
        train_val_test_ratio=[0.6, 0.2, 0.2],
        mode="train",
        input_len=12,
        output_len=12,
    )
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    batch = next(iter(loader))
    history = batch["inputs"]
    target = batch["target"]
    if not torch.is_tensor(history):
        history = torch.as_tensor(history)
    if not torch.is_tensor(target):
        target = torch.as_tensor(target)

    scaler = ZScoreScaler(
        dataset_name="PEMS04",
        train_ratio=0.6,
        norm_each_channel=False,
        rescale=True,
    )
    history = scaler.transform(history.float())
    target = scaler.transform(target.float())

    model = _build_model([3, 6, 12], use_prev_condition=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    pred = model(history_data=history[..., :3], train=True)
    tgt = target[..., :1]
    assert pred.shape == tgt.shape
    loss = torch.mean(torch.abs(pred - tgt))
    opt.zero_grad()
    loss.backward()
    opt.step()

    model.eval()
    with torch.no_grad():
        val_pred = model(history_data=history[..., :3], train=False)
        val_pred_rescaled = scaler.inverse_transform(val_pred.clone())
    assert torch.isfinite(val_pred_rescaled).all()
    print(
        "basicts_dry_run: OK",
        f"train_loss={float(loss):.4f}",
        f"pred_shape={tuple(pred.shape)}",
        f"rescaled_finite={bool(torch.isfinite(val_pred_rescaled).all())}",
    )


if __name__ == "__main__":
    test_temporal_ops()
    test_forward_shapes()
    test_backward_gradients()
    test_contribution()
    test_e0_single_stage()
    test_config_imports()
    test_hyperd_config_still_imports()
    test_basicts_dry_run()
    print("ALL VALIDATION PASSED")
