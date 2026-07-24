# HyperD PEMS04 Baseline Report

**Status:** verified against current `baselines/HyperD/PEMS04.py` (byte-identical to checkpointed config).  
**Audit commit:** `0bfe50bb54ef8842217a4511a6f0077374fe7b00`  
**Run artifact:** `checkpoints/HyperD/PEMS04_100_12_12/2a6347ff1388253de6ad9a9c506700d9/`

## Protocol

| Item | Value |
|---|---|
| Dataset | PEMS04 |
| Seed | 1 |
| Input / Output | 12 / 12 |
| Split | 0.6 / 0.2 / 0.2 |
| Scaler | ZScoreScaler (`norm_each_channel=False`, `rescale=True`) |
| Features | forward `[0,1,2]`, target `[0]` |
| Null val | 0.0 |
| Epochs | 100 |
| Batch size | 64 |
| Optimizer | Adam, lr=0.005 |
| Scheduler | OneCycleLR (see caveat) |
| Params | 14,456,246 |

## Results (best-val checkpoint)

| Metric | Value |
|---|---|
| Best val epoch | 89 |
| Best val MAE | **18.1578** |
| Test MAE | **18.2532** |
| Test RMSE | 30.0432 |
| Test MAPE | 0.1256 |
| Mean epoch time | ~7.15 s |

Horizon breakdown (final best-val test):

- H3 MAE 17.3736 / RMSE 28.6234 / MAPE 0.1190  
- H6 MAE 18.2777 / RMSE 30.0847 / MAPE 0.1256  
- H12 MAE 19.5505 / RMSE 32.0534 / MAPE 0.1357  

## Scheduler caveat (fairness-critical)

Configured `steps_per_epoch = ceil(16992 * 0.6) = 10196` (time points), not `len(train_loader) ≈ 159`.  
BasicTS steps the scheduler **once per epoch**, so OneCycle barely advances and LR stays ≈ `0.005/25 = 2e-4` (matches training logs).

New FSGR configs default to `SCHEDULER_MODE="match_hyperd"` for fair comparison with this baseline. A corrected mode is available but requires re-running HyperD under the same corrected schedule before claiming unfair-advantage-free gains from the fix alone.

## Reproduce command

```bash
conda activate basicts
cd /home/dhz/ForecastSpace
python train.py --cfg baselines/HyperD/PEMS04.py -g 0
```

Machine-readable copy: `experiments/hyperd_pems04_seed1.json`.
