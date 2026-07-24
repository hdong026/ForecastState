# Forecast-State Graph Resolution — Implementation Audit

**Audit date:** 2026-07-24  
**Repo commit at audit:** `0bfe50bb54ef8842217a4511a6f0077374fe7b00`  
**Authority:** `ForecastStateGraphResolution.pdf`  
**Branch:** `full-forecast-state-graph-resolution`

---

## Opening answers (required)

| Question | Answer |
|---|---|
| 当前模型是否只是 `X → Y3 → Y6 → Y12 → final adaptive spatial`？ | **是。** `ForecastStateChain` 文档与实现明确为 G1_final_adaptive：时间链 `[3,6,12]` 后，仅在最终全分辨率预测上做 `ABCDSpatialModule.refine_prediction`。 |
| 当前是否真的存在多级 graph resolution？ | **否。** 无 `(h_s, K_s)` schedule、无 capacity-constrained clustering、无 `Π_s / Λ_s` 统一投影。空间仅在最后一层以自适应邻接做 residual refine，且作用在 full node 上。 |
| 当前传播的是 forecast state 还是 hidden feature？ | **部分是 forecast state。** 阶段间传播的是短 horizon 预测 `Y3→Y6→Y12`（经 `interpolate_forecast` 作为条件），不是任意 latent。但 **不是** 论文中的 resolution-space forecast state（缺 graph 轴投影与 residual lift 更新）。 |
| 当前 loss 是否仍然是 weighted-sum auxiliary loss？ | **是。** `compute_legacy_chain_loss`：`L = Σ w_s L_s + w_final L_final`，权重默认 `[0.2, 0.3, 1.0]`。 |
| 当前 runner 是否真的执行论文中的 gradient projection？ | **否。** `ForecastSpaceRunner.train_iters` 只返回加权 loss，由通用 `backward(loss)` 反传；无 `g0/gs` 分离、无冲突投影、无 norm cap。 |
| 当前实现与论文方法最核心的差距是什么？ | **缺少“多分辨率预测空间中的 residual refinement + scale-matched supervision + final-primary gradient projection”整条主线。** 现实现是“时间粗到细直接预测链 + 末端自适应图平滑 + 加权辅助损失”，不等价于论文 Algorithm 1。 |

---

## Component checklist

| 论文组件 | 论文公式/章节 | 当前代码位置 | 是否完整实现 | 存在的问题 | 计划实现位置 |
|---|---|---|---|---|---|
| 1. resolution schedule Ω={(h_s,K_s)} | Eq.(2), Methodology | `ForecastSpace/PEMS04.py` 仅有 `chain_lengths=[3,6,12]` | 否 | 只有 temporal lengths，无 capacity；硬编码语义在 chain 中 | `baselines/ForecastStateGraphResolution/` + `PEMS04.py` |
| 2. temporal projection D_{h_s} | Eq.(4)(5) | `ForecastStateChain.pool_target` | 部分 | 仅用于 loss target pooling；forward 中阶段直接预测短序列，而非对 full-H forecast 做 D_h | `arch/resolution_space.py` |
| 3. temporal lifting U_{h_s} | Eq.(6) | `interpolate_forecast` in `temporal_step.py` | 部分 | 只用于把 prev forecast 插到下一 horizon / input_len，不是对 residual 的 Λ 提升 | `arch/resolution_space.py` |
| 4. graph assignment C_s | Eq.(7)(15)(16) | 无 | 否 | 完全缺失 capacity-constrained clustering | `arch/graph_resolution.py` |
| 5. graph projection P_s=D_s^{-1}C_s^⊤ | Eq.(7)(8) | 无 | 否 | — | `arch/graph_resolution.py` / `resolution_space.py` |
| 6. graph lifting L_{K_s} | Eq.(8) | 无 | 否 | — | `arch/resolution_space.py` |
| 7. unified projection Π_s | Eq.(9)(43) | 无 | 否 | — | `arch/resolution_space.py` |
| 8. unified lifting Λ_s | Eq.(9)(43) | 无 | 否 | — | `arch/resolution_space.py` |
| 9. distance-aware affinity W | Eq.(10)–(12) | 无（`final_adaptive_graph` 用原始 adj / adaptive embeddings） | 否 | 无 d̃_ij、无 W=A_sym⊙exp(-d²/σ²) | `arch/graph_resolution.py` |
| 10. capacity-constrained clustering | Eq.(13)–(17) | 无 | 否 | — | `arch/graph_resolution.py` + cache |
| 11. structural graph A_s^{str} | Eq.(18) | 无 stage-wise；仅可选 static adj | 否 | 最终 refine 默认 `adaptive_only`，甚至可不依赖结构图 | `arch/graph_operator.py` |
| 12. adaptive graph A_s^{adp} | Eq.(19)–(22) | `ABCDSpatialModule._build_adaptive_adj` | 部分 | 仅最终全节点一层；非 stage-specific region embeddings | `arch/graph_operator.py` |
| 13. mixed graph operator A_s | Eq.(22) `λ A_str+(1-λ)A_adp` | hybrid 存在但未按 stage | 否 | G1 配置 `post_spatial_mode=adaptive_only`，结构图未进入主路径 | `arch/graph_operator.py` |
| 14. projected previous forecast Z̄_s | Eq.(24) | 有 prev 条件，但是插值到 target_len，非 Π_s(Ŷ^{(s-1)}) | 否 | 下一阶段重预测整个 Y_k，不是 residual on projected state | `arch/model.py` |
| 15. resolution-space residual R_s | Eq.(25) | 无 | 否 | 每阶段直接输出绝对预测 | `arch/forecast_refinement.py` |
| 16. lifted residual update | Eq.(26)(27) | 无 | 否 | 无 Ŷ^{(s)}=Ŷ^{(s-1)}+α_s Λ_s(R_s) | `arch/model.py` |
| 17. stage coefficient α_s | Eq.(27) | 无（仅有 hybrid_alpha 等空间 refine 系数） | 否 | — | `arch/model.py` |
| 18. scale-matched supervision | Eq.(29)(30) | `pool_target` 对 temporal 部分匹配 | 部分 | 仅时间轴匹配；无 graph 轴；且与绝对 stage pred 比较，非 Π_s(Ŷ^{(s)}) | `runner.py` + diagnostics |
| 19. final-primary gradient projection | Eq.(33)–(38), Alg.1 | 无 | 否 | weighted-sum only | `arch/gradient_projection.py` + `runner.py` |
| 20. auxiliary-gradient norm cap | Eq.(37) | 无 | 否 | — | `arch/gradient_projection.py` |
| 21. graph structure cache | Implementation Notes | 无 | 否 | — | `datasets/.../forecast_state_graph_cache/` |
| 22. intermediate-state logging | 工程要求 | `return_all` 返回 chain_preds | 部分 | 无 stage resolution metadata / α / λ / graph diagnostics | `arch/diagnostics.py` |

---

## High-risk training issues observed

1. **OneCycleLR `steps_per_epoch` 使用时间点数而非 `len(train_loader)`**  
   - HyperD / ForecastSpace 配置：`math.ceil(num_time_steps * train_ratio)` = 10196  
   - 实际 batch 数约 `ceil((10196-12-12+1)/64) ≈ 159`  
   - 框架 `base_epoch_runner` **每个 epoch 只 `scheduler.step()` 一次**  
   - 结果：训练过程中 LR 基本停留在 OneCycle 初始值 ≈ `max_lr/25 = 2e-4`（与 HyperD 日志一致）  
   - **公平比较策略：** 新模型默认提供 `scheduler_mode="match_hyperd"` 复现同一调度；同时提供 `correct_batches` 并在报告中说明。

2. ForecastSpace 测试 MAE ≈ **19.05**，HyperD best-val 对应 test MAE ≈ **18.25**（seed=1），差距与“方法不完整”一致，而非单纯调参问题。

---

## Fairness baseline (existing HyperD run)

- Checkpoint dir: `checkpoints/HyperD/PEMS04_100_12_12/2a6347ff1388253de6ad9a9c506700d9/`  
- Config file identical to current `baselines/HyperD/PEMS04.py` (`diff` empty)  
- Seed 1, 100 epochs, batch 64, Adam lr=0.005 (effective ~2e-4 under broken OneCycle), ZScoreScaler, features [0,1,2]→[0], 12→12, split 0.6/0.2/0.2  
- Params: 14,456,246  
- Best val MAE: **18.1578** (epoch 89)  
- Corresponding test: MAE **18.2532**, RMSE **30.0432**, MAPE **0.1256**  
- Mean epoch time ≈ **7.15 s**

详见 `docs/baseline_report.md` / `experiments/hyperd_pems04_seed1.json`。

---

## Implementation plan (next stages)

1. 新建 `baselines/ForecastStateGraphResolution/`，不覆盖 ForecastSpace / HyperD。  
2. 实现 resolution space、capacity-constrained graph resolution、stage residual refinement、gradient projection runner。  
3. 单元测试 → one-batch overfit → tiny train → controlled E2–E6 实验。  
4. 以 HyperD seed1 为 E0，判定是否在同等协议下超过。
