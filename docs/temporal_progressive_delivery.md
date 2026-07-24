# Temporal-Only Progressive Forecasting — Delivery Report

**Branch:** `temporal-only-progressive-forecasting`  
**Task doc:** `CURSOR_TEMPORAL_PROGRESSIVE_IMPLEMENTATION.md`

---

## 1. Repository inspection summary

| Item | Finding |
|---|---|
| Repo layout | BasicTS-style: `baselines/`, `basicts/`, `train.py`, `datasets/` |
| Migrated KASA model | `baselines/ForecastSpace/arch/temporal_step.py` (`KASATemporalStep`) |
| Upstream KASA-ST | `/home/dhz/KASA-ST/.../kasa_temporal_step.py` (reference; this repo uses the migrated copy) |
| TemporalStep deps | `PatchEncoder`, `DownsampEncoder`, `MultiLayerPerceptron`, shared td/dw/spa codebooks |
| BasicTS `forward` | `history_data, future_data, batch_seen, epoch, train` → `Tensor` or `Dict` with `prediction` |
| Runner | `SimpleTimeSeriesForecastingRunner` extracts `prediction`; Tensor auto-wrapped |
| Launch | `python train.py --cfg <cfg.py> -g 0` |
| Graph in this task | **Not used.** New package is temporal-only. |

Path adapted to repo convention: `baselines/ForecastState/` (instead of `models/ForecastState/`).

---

## 2. Exact KASA TemporalStep source

`baselines/ForecastSpace/arch/temporal_step.py` → class `KASATemporalStep`

Re-exported by `baselines/ForecastState/kasa_temporal_step.py` (no redesign).

Real call interface:

```python
KASATemporalStep(... output_len=h ...)
forward(history_data, prev_forecast=None, spatial_codebook=None) -> [B, h, N, 1]
```

---

## 3. Copied / adapted dependencies

- **Reused as-is:** `KASATemporalStep` + Patch/Downsamp encoders via import
- **Thin adapter only:** `kasa_adapter.py` (rename `prev_condition`→`prev_forecast`, validate shapes)
- **New ops:** `temporal_project` / `temporal_lift` per document Parts 3–4
- **No** graph ops, clustering, adaptive adjacency, gradient projection

---

## 4–5. Modified / new files

**New (only):**

```text
baselines/ForecastState/
  __init__.py
  temporal_ops.py
  kasa_temporal_step.py
  kasa_adapter.py
  progressive_temporal.py
  losses.py
  model.py
  runner.py
  config_utils.py
  PEMS04_E0.py
  PEMS04_E1.py
  PEMS04_E2.py
scripts/validate_forecast_state_progressive.py
docs/temporal_progressive_delivery.md
```

**Unmodified:** `baselines/HyperD/**`, `baselines/ForecastSpace/**` (except reused via import).

---

## 6. Data-flow

```text
Ŷ^(0) = 0
for h in [3,6,12] (or [12] for E0):
  Z̄ = D_h(Ŷ)                  # temporal_project
  R  = KASA_adapter(X, Z̄, h)  # residual in [B,h,N,1]
  Ŷ = Ŷ + 1 · U_h(R)          # temporal_lift + accumulate
return Ŷ                      # [B,12,N,1]
```

Terminology:
- `stage_residuals[s]` = R_s  
- `stage_predictions[s]` = Ŷ^(s)  
- `stage_states[s]` = D_h(Ŷ^(s))

---

## 7–9. Validation output

```text
temporal_ops: OK
forward_shapes: OK (2, 12, 307, 1)
stage 0 (h=3): grad_l1>0
stage 1 (h=6): grad_l1>0
stage 2 (h=12): grad_l1>0
backward_gradients: OK
contribution: OK
e0_single_stage: OK
config_import E0/E1/E2: OK
hyperd_config_import: OK
basicts_dry_run: OK
ALL VALIDATION PASSED
```

Note: shape tests use `C_in=3` because KASA TemporalStep requires time/day channels; output remains `C_out=1` as required.

---

## 10. E0 / E1 / E2 commands

```bash
conda activate basicts
cd /home/dhz/ForecastSpace

python train.py --cfg baselines/ForecastState/PEMS04_E0.py -g 0
python train.py --cfg baselines/ForecastState/PEMS04_E1.py -g 0
python train.py --cfg baselines/ForecastState/PEMS04_E2.py -g 0

python scripts/validate_forecast_state_progressive.py
```

| Exp | resolutions | prev_condition | aux_weight |
|---|---|---|---|
| E0 | `[12]` | False | 0.0 |
| E1 | `[3,6,12]` | True | 0.0 |
| E2 | `[3,6,12]` | True | 0.1 (scale-matched) |

---

## 11. Unresolved incompatibilities

None blocking. Document’s illustrative `history_data` with `C_in=1` is incompatible with KASA’s time/day embeddings; production configs correctly use `FORWARD_FEATURES=[0,1,2]`.

---

## 12. Confirmation: no graph-resolution code

Confirmed: no graph convolution, clustering/pooling, graph resolution, adaptive/dynamic adjacency, spectral clustering, spatial attention, frequency branch, gradient projection, or dataset/metric/normalization changes.
