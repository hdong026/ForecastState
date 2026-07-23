# G1_final_adaptive PEMS04 12→12 (ForecastSpace / BasicTS 1.1.0)

| 字段 | 值 |
|------|-----|
| dataset | PEMS04 |
| model | G1_final_adaptive |
| input_len | 12 |
| output_len | 12 |
| chain_lengths | [3, 6, 12] |
| use_prev_condition | True |
| spatial_placement | final |
| post_spatial_mode | adaptive_only |
| best val MAE | **19.2035** |
| test MAE@best-val | **19.0327** |
| test RMSE@best-val | 30.9440 |
| test MAPE@best-val | 0.1305 |
| best epoch | 99 |
| test MAE@h3 (best-val ep) | 18.1992 |
| test MAE@h6 (best-val ep) | 19.1223 |
| test MAE@h12 (best-val ep) | 20.6805 |
| epoch time (avg) | ~27.0 s |
| GPU memory | (未单独记录; 参数量 2,162,184) |
| config path | configs/g1/G1_final_adaptive_PEMS04_12to12.py |
| log path | results/g1_final_adaptive_pems04_12to12/train_full.log |
| checkpoint dir | checkpoints/G1_final_adaptive_PEMS04_12to12/48dc9e5f854a23d3f75799f01c4d5490 |

## 与 KASA-ST 参考对比

| 指标 | KASA-ST G1_final_adaptive | ForecastSpace |
|------|-------------------------|---------------|
| test MAE@best-val | ~18.12 (ep87) | **19.03** (ep99) |
| 差距 | — | **+0.91** |

## 差距可能原因（待后续排查）

1. **训练目标不同**：KASA `ChainForecastingRunner` 对 chain 中间步有 `chain_loss_weights=[0.2,0.3,1.0]` 加权监督；ForecastSpace 第一阶段按需求仅对最终输出做 MAE（无 chain auxiliary loss）。
2. **best epoch 不同**：KASA ep87 vs 本次 ep99。
3. **框架差异**：BasicTS 1.1.0 runner/taskflow vs 旧版 EasyTorch CFG；数据协议已对齐 KASA pkl sample-split。

## 训练命令

```bash
conda activate basicts
cd /home/dhz/ForecastSpace
python scripts/run_g1_pems04.py --gpus 0
```
