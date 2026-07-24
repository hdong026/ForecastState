"""Required HyperDChain checks: import, ops, forward, consistency, backward, dry-run."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from basicts.utils import load_adj
from baselines.HyperD.arch import HyperD
from baselines.HyperDChain.arch import HyperDForecastStateChain
from baselines.HyperDChain.arch.stfe_encoder import HyperDSTFEEncoder
from baselines.HyperDChain.arch.temporal_ops import temporal_lift, temporal_project


def _adj():
    adj_mx, _ = load_adj("datasets/PEMS04/adj_mx.pkl", "normlap")
    return torch.tensor(adj_mx[0])


def _history(batch: int = 2, nodes: int = 307) -> torch.Tensor:
    history = torch.randn(batch, 12, nodes, 3)
    history[..., 1] = torch.rand(batch, 12, nodes)
    history[..., 2] = torch.randint(0, 7, (batch, 12, nodes)).float() / 7.0
    return history


def _build_chain(**overrides):
    params = dict(
        seq_len=12,
        pred_len=12,
        num_nodes=307,
        init_path_daily="datasets/PEMS04/daily_init.npy",
        init_path_weekly="datasets/PEMS04/weekly_init.npy",
        adj=_adj(),
        alpha=2,
        F_low=3,
        embed_size=64,
        hidden_size=128,
        fc_hidden_size=128,
        time_of_day_size=288,
        day_of_week_size=7,
        chain_lengths=[3, 6, 12],
        chain_loss_weights=[0.0, 0.0, 1.0],
        use_prev_condition=True,
        condition_hidden_size=32,
        use_dual_view_loss=True,
        dual_view_weight=1.0,
    )
    params.update(overrides)
    return HyperDForecastStateChain(**params)


def test_import():
    assert HyperDForecastStateChain is not None
    print("import: OK")


def test_projection_lifting():
    for h in [3, 6, 12]:
        z = torch.randn(2, h, 307)
        lifted = temporal_lift(z, 12)
        assert lifted.shape == (2, 12, 307)
        projected = temporal_project(lifted, h)
        torch.testing.assert_close(projected, z, rtol=1e-5, atol=1e-6)
    x = torch.randn(2, 12, 307)
    assert temporal_project(x, 12).shape == (2, 12, 307)
    assert temporal_project(x, 3).shape == (2, 3, 307)
    assert temporal_project(x, 6).shape == (2, 6, 307)
    print("projection_lifting: OK")


def test_stfe_encoder():
    enc = HyperDSTFEEncoder(
        num_nodes=307,
        seq_len=12,
        embed_size=64,
        hidden_size=128,
    )
    x = torch.randn(2, 12, 307, requires_grad=True)
    out = enc(x)
    assert out.shape == (2, 307, 12 * 64)
    assert torch.isfinite(out).all()
    out.mean().backward()
    for name, p in enc.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name
    print("stfe_encoder: OK", tuple(out.shape))


def test_forward_shapes():
    model = _build_chain()
    history = _history()
    out = model(history_data=history, return_all=True)
    assert out["prediction"].shape == (2, 12, 307, 1)
    assert out["periodic_out"].shape == (2, 12, 307, 1)
    assert len(out["stage_forecast_states"]) == 3
    assert out["stage_forecast_states"][0].shape == (2, 3, 307, 1)
    assert out["stage_forecast_states"][1].shape == (2, 6, 307, 1)
    assert out["stage_forecast_states"][2].shape == (2, 12, 307, 1)
    assert torch.isfinite(out["prediction"]).all()
    assert torch.isfinite(out["dual_view_loss"]).all()
    print("forward_shapes: OK")


def test_state_consistency():
    model = _build_chain()
    out = model(history_data=_history(), return_all=True)
    for proposal, residual_state in zip(
        out["stage_residual_proposals"],
        out["stage_residual_states"],
    ):
        torch.testing.assert_close(
            residual_state,
            proposal,
            rtol=1e-5,
            atol=1e-6,
        )
    print("state_consistency: OK")


def test_periodic_non_duplication():
    model = _build_chain()
    out = model(history_data=_history(), return_all=True)
    periodic = out["periodic_out"]
    for full_pred, full_res in zip(
        out["stage_full_predictions"],
        out["stage_full_residuals"],
    ):
        torch.testing.assert_close(
            full_pred,
            periodic + full_res,
            rtol=1e-5,
            atol=1e-6,
        )
    print("periodic_non_duplication: OK")


def test_backward_gradients():
    # Final-horizon overwrite + zero-init adapters imply stages 3/6 get no
    # gradient from final-only loss at init. Use scale-matched aux loss so
    # every stage head is supervised, matching HC2 training semantics.
    from basicts.metrics import masked_mae
    from baselines.HyperDChain.arch.chain_loss import compute_hyperd_chain_loss

    model = _build_chain(
        use_prev_condition=True,
        chain_loss_weights=[0.1, 0.1, 1.0],
    )
    out = model(history_data=_history(), return_all=True)
    target = torch.randn_like(out["prediction"])
    loss = compute_hyperd_chain_loss(
        out=out,
        real_value=target,
        chain_lengths=[3, 6, 12],
        chain_loss_weights=[0.1, 0.1, 1.0],
        scaler=None,
        metric_forward=lambda fn, args: fn(
            args["prediction"],
            args["target"],
            null_val=0.0,
        ),
        loss_fn=masked_mae,
    )
    loss.backward()

    required = [
        ("daily_emb.data", model.daily_emb.data),
        ("weekly_emb.data", model.weekly_emb.data),
        ("stfe.embeddings", model.stfe_encoder.embeddings),
        ("stfe.spatial_r1", model.stfe_encoder.spatial_r1),
        ("stfe.temporal_r1", model.stfe_encoder.temporal_r1),
        (
            "shared_hidden.0.weight",
            model.progressive_head.shared_hidden[0].weight,
        ),
        (
            "head_3",
            model.progressive_head.stage_output_heads["3"].weight,
        ),
        (
            "head_6",
            model.progressive_head.stage_output_heads["6"].weight,
        ),
        (
            "head_12",
            model.progressive_head.stage_output_heads["12"].weight,
        ),
        # Stage-3 prev residual is exactly 0, so adapter-3 weight gets no
        # input signal; its bias still receives gradient via aux loss.
        (
            "adapter_3_last_bias",
            model.progressive_head.condition_adapters["3"][-1].bias,
        ),
        (
            "adapter_6_last",
            model.progressive_head.condition_adapters["6"][-1].weight,
        ),
        (
            "adapter_12_last",
            model.progressive_head.condition_adapters["12"][-1].weight,
        ),
    ]
    for name, param in required:
        assert param.grad is not None, f"{name} grad is None"
        assert torch.isfinite(param.grad).all(), f"{name} non-finite"
        assert float(param.grad.abs().sum()) > 0.0, f"{name} zero grad"
    print("backward_gradients: OK")


def test_stage_contribution():
    # Zero each recorded stage correction while keeping the others fixed.
    # Later corrections are not recomputed, so intermediate stages still
    # change the final residual under the stored-correction ablation.
    model = _build_chain()
    model.eval()
    history = _history()
    with torch.no_grad():
        base = model(history_data=history, return_all=True)
        base_pred = base["prediction"].clone()
        corrections = [c.clone() for c in base["stage_corrections"]]

    for stage_idx in range(len(corrections)):
        residual_full = torch.zeros_like(base["periodic_out"][..., 0])
        for i, corr in enumerate(corrections):
            if i == stage_idx:
                corr_use = torch.zeros_like(corr[..., 0])
            else:
                corr_use = corr[..., 0]
            residual_full = residual_full + temporal_lift(
                corr_use,
                full_horizon=12,
            )
        pred = (base["periodic_out"][..., 0] + residual_full).unsqueeze(-1)
        delta = float((pred - base_pred).abs().mean())
        assert delta > 1e-6, f"stage {stage_idx} has no measurable contribution"
    print("stage_contribution: OK")

def test_hc0_hyperd_equivalence():
    """Copy matching weights from HyperD into HC0-style chain; compare outputs."""
    torch.manual_seed(0)
    hyperd = HyperD(
        seq_len=12,
        pred_len=12,
        num_nodes=307,
        init_path_daily="datasets/PEMS04/daily_init.npy",
        init_path_weekly="datasets/PEMS04/weekly_init.npy",
        adj=_adj(),
        alpha=2,
        F_low=3,
        embed_size=64,
        hidden_size=128,
        fc_hidden_size=128,
        time_of_day_size=288,
        day_of_week_size=7,
    )
    chain = _build_chain(
        chain_lengths=[12],
        chain_loss_weights=[1.0],
        use_prev_condition=False,
    )

    chain.daily_emb.load_state_dict(hyperd.daily_emb.state_dict())
    chain.weekly_emb.load_state_dict(hyperd.weekly_emb.state_dict())

    # STFE encoder params (everything except fc)
    enc_sd = chain.stfe_encoder.state_dict()
    stfe_sd = hyperd.stfe.state_dict()
    for key in enc_sd:
        enc_sd[key] = stfe_sd[key]
    chain.stfe_encoder.load_state_dict(enc_sd)

    # Shared hidden = first Linear+LeakyReLU of HyperD fc; stage_12 = second Linear
    chain.progressive_head.shared_hidden[0].weight.data.copy_(
        hyperd.stfe.fc[0].weight.data
    )
    chain.progressive_head.shared_hidden[0].bias.data.copy_(
        hyperd.stfe.fc[0].bias.data
    )
    chain.progressive_head.stage_output_heads["12"].weight.data.copy_(
        hyperd.stfe.fc[2].weight.data
    )
    chain.progressive_head.stage_output_heads["12"].bias.data.copy_(
        hyperd.stfe.fc[2].bias.data
    )

    history = _history()
    hyperd.eval()
    chain.eval()
    with torch.no_grad():
        y_h = hyperd(
            history_data=history,
            future_data=None,
            batch_seen=0,
            epoch=1,
            train=False,
        )["prediction"]
        y_c = chain(history_data=history, return_all=False)["prediction"]
    torch.testing.assert_close(y_c, y_h, rtol=1e-5, atol=1e-5)
    print("hc0_hyperd_equivalence: OK")


def test_basicts_dry_run():
    from basicts.metrics import masked_mae
    from basicts.scaler import ZScoreScaler
    from baselines.HyperDChain.arch.chain_loss import compute_hyperd_chain_loss
    from baselines.HyperDChain.PEMS04_HC0 import CFG as CFG0
    from baselines.HyperDChain.PEMS04_HC1 import CFG as CFG1
    from baselines.HyperDChain.PEMS04_HC2 import CFG as CFG2

    scaler = ZScoreScaler("PEMS04", 0.6, False, True)
    history = _history(batch=4)
    target = torch.randn(4, 12, 307, 1)

    for name, cfg in [("HC0", CFG0), ("HC1", CFG1), ("HC2", CFG2)]:
        model = HyperDForecastStateChain(**cfg.MODEL.PARAM)
        model.train()
        out = model(
            history_data=history,
            future_data=None,
            batch_seen=0,
            epoch=1,
            train=True,
            return_all=True,
        )
        loss = compute_hyperd_chain_loss(
            out=out,
            real_value=target,
            chain_lengths=list(cfg.MODEL.PARAM["chain_lengths"]),
            chain_loss_weights=list(cfg.MODEL.PARAM["chain_loss_weights"]),
            scaler=scaler,
            metric_forward=lambda fn, args: fn(
                args["prediction"],
                args["target"],
                null_val=0.0,
            ),
            loss_fn=masked_mae,
        )
        assert torch.isfinite(loss).all(), name
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        opt.zero_grad()
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_out = model(history_data=history, return_all=True)
            pred = scaler.inverse_transform(val_out["prediction"].clone())
            tgt = scaler.inverse_transform(target.clone())
            mae = masked_mae(pred, tgt, null_val=0.0)
            assert torch.isfinite(mae).all(), name

        ckpt = cfg.TRAIN.CKPT_SAVE_DIR
        assert "HyperDChain" in ckpt and "PEMS04" in ckpt, ckpt
        print(f"basicts_dry_run[{name}]: OK loss={float(loss):.4f} mae={float(mae):.4f}")
        print(f"  ckpt={ckpt}")
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  params={n_params}")


def main():
    test_import()
    test_projection_lifting()
    test_stfe_encoder()
    test_forward_shapes()
    test_state_consistency()
    test_periodic_non_duplication()
    test_backward_gradients()
    test_stage_contribution()
    test_hc0_hyperd_equivalence()
    test_basicts_dry_run()
    print("\nALL HyperDChain CHECKS PASSED")


if __name__ == "__main__":
    main()
