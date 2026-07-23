# ForecastSpace

Clean migration of **G1_final_adaptive** from KASA-ST onto **BasicTS 1.1.0**.

## Layout

- `forecast_space/models/g1/` — standalone G1 model (no vendored basicts)
- `configs/g1/` — BasicTS 1.1.0 training config
- `datasets/PEMS04` — symlink to KASA-ST pkl data (sample-based 6:2:2 split)
- `scripts/` — smoke test, training entry, optional data conversion

## Environment

Use the `basicts` conda env (BasicTS **1.1.0** from site-packages):

```bash
conda activate basicts
cd /home/dhz/ForecastSpace
```

## Smoke test

```bash
python scripts/smoke_test_g1.py
```

## 1-epoch integration check

```bash
python scripts/run_g1_pems04.py --gpus 0 --epochs 1
```

## Full training (PEMS04 12→12)

```bash
python scripts/run_g1_pems04.py --gpus 0
```

Equivalent BasicTS launcher API:

```bash
python -c "
from basicts.launcher import BasicTSLauncher
from configs.g1.G1_final_adaptive_PEMS04_12to12 import build_config
cfg = build_config()
cfg.gpus = '0'; cfg.gpu_num = 1
BasicTSLauncher.launch_training(cfg)
"
```

## Data protocol

Default training uses **KASA pkl protocol** via `Pems04PklDataset` to match old results (~18.12 MAE).

Optional conversion to BasicTS `{train,val,test}_data.npy` windows:

```bash
python scripts/convert_pems04_to_latest_basicts.py
```

Output: `datasets/PEMS04_latest/` (does not overwrite symlinked source).

## G1 settings

- input/output: 12 / 12
- chain_lengths: [3, 6, 12]
- use_prev_condition: True
- spatial_placement: final
- post_spatial_mode: adaptive_only

## Reference (KASA-ST)

G1_final_adaptive PEMS04 12→12 test MAE ≈ **18.12** (best-val checkpoint).
