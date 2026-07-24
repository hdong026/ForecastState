# Cursor Implementation Guide
# HyperD-Based Forecast-State Chain with Maximum Code Reuse

## 0. Instruction to Cursor

Read this document completely before editing any file.

Implement a new temporal-only model that combines:

1. the existing HyperD periodic and frequency-domain forecasting backbone; and
2. the Forecast-State Chain mechanism with temporal resolutions `[3, 6, 12]`.

The implementation must reuse as much code as possible from the existing repository:

```text
baselines/HyperD/
```

Do not rewrite HyperD from memory.

Do not modify the original HyperD baseline unless a genuine repository-level compatibility bug makes it unavoidable. The preferred solution is to create a new independent package:

```text
baselines/HyperDChain/
```

The original commands below must continue to work unchanged:

```bash
python train.py --cfg='baselines/HyperD/PEMS04.py'
python train.py --cfg='baselines/ForecastSpace/PEMS04.py'
```

The new model should initially implement only temporal progressive forecasting:

\[
T_3 \rightarrow T_6 \rightarrow T_{12}.
\]

Do not add graph resolution, graph clustering, graph pooling, adaptive adjacency, KASA temporal modules, PTSE, DSHP, priors beyond HyperD's original periodic prior, or any unrelated architecture.

---

# 1. Repository Context That Must Be Verified Locally

The current repository is expected to contain:

```text
baselines/
├── HyperD/
│   ├── arch/
│   │   ├── STFE.py
│   │   ├── attention.py
│   │   ├── hyperd_arch.py
│   │   ├── loss.py
│   │   └── __init__.py
│   ├── PEMS03.py
│   ├── PEMS04.py
│   ├── PEMS07.py
│   └── PEMS08.py
│
└── ForecastSpace/
    ├── arch/
    │   ├── forecast_state_chain.py
    │   ├── chain_loss.py
    │   └── ...
    ├── runner.py
    ├── PEMS03.py
    ├── PEMS04.py
    ├── PEMS07.py
    └── PEMS08.py
```

Before coding, inspect the local versions of these files:

```text
baselines/HyperD/arch/hyperd_arch.py
baselines/HyperD/arch/STFE.py
baselines/HyperD/arch/attention.py
baselines/HyperD/arch/loss.py
baselines/HyperD/PEMS04.py

baselines/ForecastSpace/arch/forecast_state_chain.py
baselines/ForecastSpace/arch/chain_loss.py
baselines/ForecastSpace/runner.py
baselines/ForecastSpace/PEMS04.py

basicts/runners/
train.py
```

Do not rely only on this document. If the local code differs, preserve the local BasicTS interfaces and report the differences.

Before editing, record:

```bash
git status --short
git rev-parse HEAD
git branch --show-current
```

Do not overwrite unrelated local changes.

---

# 2. Existing HyperD Computation

The current HyperD model uses the following structure.

For history:

\[
\mathbf X\in\mathbb R^{B\times P\times N}.
\]

It produces daily and weekly periodic patterns:

\[
\mathbf S_{\mathrm{in}}
=
\mathbf S^D_{\mathrm{in}}
+
\mathbf S^W_{\mathrm{in}},
\]

and defines the history residual:

\[
\mathbf R_{\mathrm{in}}
=
\mathbf X-\mathbf S_{\mathrm{in}}.
\]

The STFE module predicts the future residual:

\[
\widehat{\mathbf R}_{\mathrm{out}}
=
\operatorname{STFE}(\mathbf R_{\mathrm{in}}).
\]

The periodic module generates the future periodic baseline:

\[
\mathbf S_{\mathrm{out}}
=
\mathbf S^D_{\mathrm{out}}
+
\mathbf S^W_{\mathrm{out}}.
\]

The final prediction is:

\[
\widehat{\mathbf Y}
=
\mathbf S_{\mathrm{out}}
+
\widehat{\mathbf R}_{\mathrm{out}}.
\]

The original code also computes:

\[
\mathcal L_{\mathrm{DVA}}
=
\alpha
\left(
\mathcal L_{\mathrm{low}}
+
\mathcal L_{\mathrm{high}}
\right)
\]

through the existing `dual_view_loss`.

The new implementation must preserve these useful HyperD ideas:

- statistical daily/weekly periodic initialization;
- `Hybrid_Periodic_Pattern`;
- `SelfAttention`;
- spatial FFT complex MLP;
- temporal FFT complex MLP;
- sparsity thresholding;
- original residual decomposition;
- final dual-view alignment loss;
- BasicTS input features `[flow, time_of_day, day_of_week]`.

---

# 3. Critical Design Decision

## 3.1 Do Not Stack Three Complete HyperD Models

Do not implement:

```python
y3 = HyperD(pred_len=3)(...)
y6 = HyperD(pred_len=6)(...)
y12 = HyperD(pred_len=12)(...)
y = lift(y3) + lift(y6) + y12
```

This is incorrect because:

1. the daily and weekly periodic terms would be repeatedly added;
2. `pred_len=3` in the original HyperD means the first three future timestamps, not three coarse bins spanning the full 12-step horizon;
3. three full STFE backbones would unnecessarily triple parameters and compute;
4. it would not implement the intended projection–correction–lifting chain.

## 3.2 Correct Interpretation

HyperD should provide:

1. one shared full-horizon periodic baseline;
2. one shared STFE frequency representation of the history residual;
3. HyperD-style prediction heads at temporal resolutions 3, 6, and 12.

Forecast-State Chain should provide:

1. the accumulated residual forecast state;
2. projection to the current temporal resolution;
3. a stage-conditioned state proposal;
4. a correction relative to the current state;
5. lifting and cumulative refinement;
6. scale-matched intermediate supervision.

---

# 4. Recommended Mathematical Model

Let the full horizon be:

\[
H=12,
\]

and let:

\[
\mathcal H=[3,6,12].
\]

## 4.1 HyperD Periodic Baseline

Reuse the original daily and weekly modules to produce:

\[
\mathbf S_{\mathrm{in}}
\in\mathbb R^{B\times P\times N},
\]

and:

\[
\mathbf S_{\mathrm{out}}
\in\mathbb R^{B\times H\times N}.
\]

The future periodic pattern must always be generated at the full horizon \(H=12\):

```python
S_D_out = daily_emb(..., full_horizon)
S_W_out = weekly_emb(..., full_horizon)
S_out = S_D_out + S_W_out
```

Do not generate separate periodic outputs by calling the periodic modules with lengths 3 and 6.

The coarse periodic states are obtained by projection:

\[
D_3(\mathbf S_{\mathrm{out}}),
\qquad
D_6(\mathbf S_{\mathrm{out}}).
\]

## 4.2 Shared HyperD STFE Encoder

Compute:

\[
\mathbf R_{\mathrm{in}}
=
\mathbf X-\mathbf S_{\mathrm{in}}.
\]

Reuse the STFE token embedding, spatial complex MLP, temporal complex MLP, and residual connection to obtain:

\[
\mathbf H_R
=
E_{\mathrm{STFE}}(\mathbf R_{\mathrm{in}})
\in
\mathbb R^{B\times N\times(Pd)}.
\]

The code before the original STFE `fc` must remain mathematically unchanged.

## 4.3 Residual Forecast State

Initialize the full-horizon residual forecast as:

\[
\widehat{\mathbf R}^{(0)}
=
\mathbf 0
\in
\mathbb R^{B\times H\times N}.
\]

The corresponding initial forecast is:

\[
\widehat{\mathbf Y}^{(0)}
=
\mathbf S_{\mathrm{out}}.
\]

Thus, HyperD's periodic output is the initial explicit forecast, and the chain progressively predicts deviations from it.

## 4.4 Stagewise Residual-State Proposal

For stage \(s\) with temporal resolution \(h_s\), project the current accumulated residual forecast:

\[
\overline{\mathbf R}_s
=
D_{h_s}
\left(
\widehat{\mathbf R}^{(s-1)}
\right).
\]

Use a HyperD-style base head to produce an independent residual-state proposal:

\[
\mathbf B_s
=
G_s^{\mathrm{HyperD}}(\mathbf H_R)
\in
\mathbb R^{B\times h_s\times N}.
\]

Use a lightweight condition adapter:

\[
\mathbf C_s
=
A_s(\overline{\mathbf R}_s).
\]

The conditioned residual-state proposal is:

\[
\widetilde{\mathbf R}_s
=
\mathbf B_s+\mathbf C_s.
\]

The actual stage correction is:

\[
\Delta\mathbf R_s
=
\widetilde{\mathbf R}_s
-
\overline{\mathbf R}_s.
\]

Lift the correction:

\[
U_{h_s}(\Delta\mathbf R_s)
\in
\mathbb R^{B\times H\times N}.
\]

Update:

\[
\widehat{\mathbf R}^{(s)}
=
\widehat{\mathbf R}^{(s-1)}
+
U_{h_s}(\Delta\mathbf R_s).
\]

The full forecast after stage \(s\) is:

\[
\widehat{\mathbf Y}^{(s)}
=
\mathbf S_{\mathrm{out}}
+
\widehat{\mathbf R}^{(s)}.
\]

The final output is:

\[
\widehat{\mathbf Y}
=
\widehat{\mathbf Y}^{(S)}.
\]

## 4.5 Why Use Proposal Minus Previous State?

Do not directly accumulate three independent HyperD residual predictions:

\[
U_3(\mathbf B_3)+U_6(\mathbf B_6)+\mathbf B_{12}.
\]

That would repeatedly count the same residual signal.

Instead:

\[
\Delta\mathbf R_s
=
\widetilde{\mathbf R}_s-\overline{\mathbf R}_s
\]

means each stage updates the currently represented residual state toward a finer proposal.

For block averaging and block repetition:

\[
D_h(U_h(\mathbf Z))=\mathbf Z.
\]

Therefore, after the update, the current stage's projected residual state becomes the proposal:

\[
D_{h_s}
\left(
\widehat{\mathbf R}^{(s)}
\right)
=
\widetilde{\mathbf R}_s.
\]

This gives the chain a clean state-transition interpretation.

---

# 5. Target File Structure

Create:

```text
baselines/
└── HyperDChain/
    ├── __init__.py
    ├── PEMS04.py
    ├── runner.py
    └── arch/
        ├── __init__.py
        ├── hyperd_chain_arch.py
        ├── stfe_encoder.py
        ├── progressive_head.py
        ├── temporal_ops.py
        └── chain_loss.py
```

Initially implement and validate only PEMS04.

Do not immediately create PEMS03/07/08 configs until PEMS04:

- imports successfully;
- passes random forward/backward tests;
- runs one BasicTS batch;
- completes at least one epoch;
- produces finite validation metrics.

After PEMS04 is stable, copy its config to the other datasets with only dataset-specific values changed.

---

# 6. Maximum Code-Reuse Policy

## 6.1 Reuse Directly by Import

Prefer direct imports for unchanged components:

```python
from baselines.HyperD.arch.hyperd_arch import Hybrid_Periodic_Pattern
from baselines.HyperD.arch.loss import dual_view_loss
```

This directly reuses:

- original periodic pattern code;
- original attention implementation through `Hybrid_Periodic_Pattern`;
- original statistical initialization behavior;
- original DVA loss.

Do not copy `attention.py` or `loss.py` unless import mechanics make reuse impossible.

If direct imports fail only because package markers are missing, first fix package initialization cleanly rather than duplicating code.

## 6.2 Copy and Minimally Refactor STFE

The original `stfe` combines:

1. frequency encoder; and
2. final prediction head.

The new chain needs the shared frequency representation before the final `fc`.

Create `stfe_encoder.py` by copying the following parts from:

```text
baselines/HyperD/arch/STFE.py
```

without mathematical changes:

- `scale`;
- `feature_size`;
- `seq_length`;
- `sparsity_threshold`;
- `embeddings`;
- all spatial real/imaginary parameters;
- all temporal real/imaginary parameters;
- `tokenEmb`;
- `C_MLP_s`;
- `C_MLP_t`;
- `C_MLP`;
- the residual connection `x = x + bias`.

Only remove the original final `fc` from the encoder and return the flattened feature.

Do not change:

- FFT dimensions;
- `norm='ortho'`;
- complex multiplication formulas;
- activation functions inside complex MLP;
- `softshrink`;
- parameter initialization scale;
- residual connection order.

## 6.3 Reuse the Original HyperD FC Structure

The original STFE forecasting head is:

```python
nn.Sequential(
    nn.Linear(seq_len * embed_size, fc_hidden_size),
    nn.LeakyReLU(),
    nn.Linear(fc_hidden_size, pred_len),
)
```

The new model should preserve this structure.

Prefer:

```python
self.shared_hidden_head = nn.Sequential(
    nn.Linear(seq_len * embed_size, fc_hidden_size),
    nn.LeakyReLU(),
)
```

and separate output layers:

```python
self.stage_output_heads = nn.ModuleDict({
    "3": nn.Linear(fc_hidden_size, 3),
    "6": nn.Linear(fc_hidden_size, 6),
    "12": nn.Linear(fc_hidden_size, 12),
})
```

This preserves the original HyperD head form while sharing the expensive first projection.

Do not replace it with a Transformer, convolutional decoder, GRU, or deep MLP.

---

# 7. Temporal Projection and Lifting

Create `temporal_ops.py`.

## 7.1 Projection

For:

```python
x.shape == [B, H, N]
```

define:

```python
def temporal_project(x, target_len):
    ...
```

When divisible, use non-overlapping block average:

\[
(D_h(\mathbf X))_\tau
=
\frac{1}{a}
\sum_{t=(\tau-1)a+1}^{\tau a}
\mathbf X_t,
\qquad
a=H/h.
\]

For \(H=12\):

```text
12 -> 3: average each block of 4
12 -> 6: average each block of 2
12 -> 12: identity
```

Reference implementation:

```python
from __future__ import annotations

import torch
import torch.nn.functional as F


def temporal_project(
    x: torch.Tensor,
    target_len: int,
) -> torch.Tensor:
    """
    Args:
        x: [B, H, N]
        target_len: target temporal resolution

    Returns:
        [B, target_len, N]
    """
    if x.ndim != 3:
        raise ValueError(
            f"Expected [B, H, N], got {tuple(x.shape)}."
        )

    batch_size, horizon, num_nodes = x.shape

    if target_len <= 0 or target_len > horizon:
        raise ValueError(
            f"target_len must be in [1, {horizon}], got {target_len}."
        )

    if target_len == horizon:
        return x

    if horizon % target_len == 0:
        block_size = horizon // target_len
        return x.reshape(
            batch_size,
            target_len,
            block_size,
            num_nodes,
        ).mean(dim=2)

    pooled = x.transpose(1, 2)
    pooled = F.adaptive_avg_pool1d(
        pooled,
        output_size=target_len,
    )
    return pooled.transpose(1, 2).contiguous()
```

## 7.2 Lifting

For:

```python
x.shape == [B, h, N]
```

lift to `[B, H, N]`.

When divisible, use block repetition:

```python
def temporal_lift(
    x: torch.Tensor,
    full_horizon: int,
) -> torch.Tensor:
    """
    Args:
        x: [B, h, N]
        full_horizon: H

    Returns:
        [B, H, N]
    """
    if x.ndim != 3:
        raise ValueError(
            f"Expected [B, h, N], got {tuple(x.shape)}."
        )

    _, current_len, _ = x.shape

    if current_len <= 0 or current_len > full_horizon:
        raise ValueError(
            f"Cannot lift length {current_len} to {full_horizon}."
        )

    if current_len == full_horizon:
        return x

    if full_horizon % current_len == 0:
        repeat_factor = full_horizon // current_len
        return x.repeat_interleave(
            repeat_factor,
            dim=1,
        )

    interpolated = F.interpolate(
        x.transpose(1, 2),
        size=full_horizon,
        mode="linear",
        align_corners=False,
    )
    return interpolated.transpose(1, 2).contiguous()
```

## 7.3 Required Operator Test

For random:

```python
z = torch.randn(B, h, N)
```

verify numerically:

```python
projected = temporal_project(
    temporal_lift(z, 12),
    h,
)
torch.testing.assert_close(projected, z)
```

for:

```python
h in [3, 6, 12]
```

---

# 8. Shared STFE Encoder

Create:

```text
baselines/HyperDChain/arch/stfe_encoder.py
```

Reference structure:

```python
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HyperDSTFEEncoder(nn.Module):
    """
    HyperD STFE before the original final fc head.

    The frequency-domain computation must remain equivalent
    to baselines/HyperD/arch/STFE.py.
    """

    def __init__(
        self,
        num_nodes: int,
        seq_len: int,
        embed_size: int,
        hidden_size: int,
    ) -> None:
        super().__init__()

        self.scale = 0.02
        self.feature_size = num_nodes
        self.seq_length = seq_len
        self.embed_size = embed_size
        self.sparsity_threshold = 0.01

        self.embeddings = nn.Parameter(
            torch.randn(1, embed_size)
        )

        self.spatial_r1 = nn.Parameter(
            self.scale * torch.randn(embed_size, hidden_size)
        )
        self.spatial_i1 = nn.Parameter(
            self.scale * torch.randn(embed_size, hidden_size)
        )
        self.spatial_rb1 = nn.Parameter(
            self.scale * torch.randn(hidden_size)
        )
        self.spatial_ib1 = nn.Parameter(
            self.scale * torch.randn(hidden_size)
        )
        self.spatial_r2 = nn.Parameter(
            self.scale * torch.randn(hidden_size, embed_size)
        )
        self.spatial_i2 = nn.Parameter(
            self.scale * torch.randn(hidden_size, embed_size)
        )
        self.spatial_rb2 = nn.Parameter(
            self.scale * torch.randn(embed_size)
        )
        self.spatial_ib2 = nn.Parameter(
            self.scale * torch.randn(embed_size)
        )

        self.temporal_r1 = nn.Parameter(
            self.scale * torch.randn(embed_size, hidden_size)
        )
        self.temporal_i1 = nn.Parameter(
            self.scale * torch.randn(embed_size, hidden_size)
        )
        self.temporal_rb1 = nn.Parameter(
            self.scale * torch.randn(hidden_size)
        )
        self.temporal_ib1 = nn.Parameter(
            self.scale * torch.randn(hidden_size)
        )
        self.temporal_r2 = nn.Parameter(
            self.scale * torch.randn(hidden_size, embed_size)
        )
        self.temporal_i2 = nn.Parameter(
            self.scale * torch.randn(hidden_size, embed_size)
        )
        self.temporal_rb2 = nn.Parameter(
            self.scale * torch.randn(embed_size)
        )
        self.temporal_ib2 = nn.Parameter(
            self.scale * torch.randn(embed_size)
        )

    @property
    def output_dim(self) -> int:
        return self.seq_length * self.embed_size

    def tokenEmb(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(3)
        return x * self.embeddings

    def C_MLP_s(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.fft.rfft(
            x,
            dim=2,
            norm="ortho",
        )
        y = self.C_MLP(
            x,
            self.spatial_r1,
            self.spatial_i1,
            self.spatial_r2,
            self.spatial_i2,
            self.spatial_rb1,
            self.spatial_rb2,
            self.spatial_ib1,
            self.spatial_ib2,
        )
        return torch.fft.irfft(
            y,
            n=self.feature_size,
            dim=2,
            norm="ortho",
        )

    def C_MLP_t(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = torch.fft.rfft(
            x,
            dim=2,
            norm="ortho",
        )
        y = self.C_MLP(
            x,
            self.temporal_r1,
            self.temporal_i1,
            self.temporal_r2,
            self.temporal_i2,
            self.temporal_rb1,
            self.temporal_rb2,
            self.temporal_ib1,
            self.temporal_ib2,
        )
        x = torch.fft.irfft(
            y,
            n=self.seq_length,
            dim=2,
            norm="ortho",
        )
        return x.transpose(1, 2)

    def C_MLP(
        self,
        x,
        r1,
        i1,
        r2,
        i2,
        rb1,
        rb2,
        ib1,
        ib2,
    ):
        o1_real = F.relu(
            torch.einsum("bijd,df->bijf", x.real, r1)
            - torch.einsum("bijd,df->bijf", x.imag, i1)
            + rb1
        )
        o1_imag = F.relu(
            torch.einsum("bijd,df->bijf", x.imag, r1)
            + torch.einsum("bijd,df->bijf", x.real, i1)
            + ib1
        )
        o2_real = F.relu(
            torch.einsum("bijf,fd->bijd", o1_real, r2)
            - torch.einsum("bijf,fd->bijd", o1_imag, i2)
            + rb2
        )
        o2_imag = F.relu(
            torch.einsum("bijf,fd->bijd", o1_imag, r2)
            + torch.einsum("bijf,fd->bijd", o1_real, i2)
            + ib2
        )
        y = torch.stack(
            [o2_real, o2_imag],
            dim=-1,
        )
        y = F.softshrink(
            y,
            lambd=self.sparsity_threshold,
        )
        return torch.view_as_complex(y)

    def forward(
        self,
        residual_history: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            residual_history: [B, P, N]

        Returns:
            encoded: [B, N, P * embed_size]
        """
        if residual_history.ndim != 3:
            raise ValueError(
                "residual_history must have shape [B, P, N], "
                f"got {tuple(residual_history.shape)}."
            )

        batch_size, seq_len, num_nodes = residual_history.shape

        if seq_len != self.seq_length:
            raise ValueError(
                f"Expected seq_len={self.seq_length}, got {seq_len}."
            )

        if num_nodes != self.feature_size:
            raise ValueError(
                f"Expected num_nodes={self.feature_size}, got {num_nodes}."
            )

        x = self.tokenEmb(residual_history)
        bias = x

        x = self.C_MLP_s(x)
        x = self.C_MLP_t(x)
        x = x + bias

        return x.transpose(1, 2).reshape(
            batch_size,
            num_nodes,
            -1,
        )
```

The final implementation should be compared line-by-line against the local original STFE.

Do not introduce cosmetic mathematical changes while copying.

---

# 9. Progressive HyperD Head

Create:

```text
baselines/HyperDChain/arch/progressive_head.py
```

The base prediction pathway should retain the original HyperD FC form.

## 9.1 Shared Hidden Projection

```python
self.shared_hidden = nn.Sequential(
    nn.Linear(
        seq_len * embed_size,
        fc_hidden_size,
    ),
    nn.LeakyReLU(),
)
```

## 9.2 Stage Output Heads

```python
self.stage_output_heads = nn.ModuleDict({
    str(h): nn.Linear(fc_hidden_size, h)
    for h in chain_lengths
})
```

## 9.3 Lightweight Previous-State Adapter

The previous residual state has shape:

```python
[B, h, N]
```

Transpose it to:

```python
[B, N, h]
```

Use a small per-node adapter:

```python
nn.Sequential(
    nn.Linear(h, condition_hidden_size),
    nn.LeakyReLU(),
    nn.Linear(condition_hidden_size, h),
)
```

Zero-initialize the last linear layer:

```python
nn.init.zeros_(adapter[-1].weight)
nn.init.zeros_(adapter[-1].bias)
```

This ensures that training begins from the HyperD-style base proposal rather than a random condition perturbation.

Do not use a gate in the first version.

Do not add attention, graph convolution, Fourier layers, or deep decoders inside the condition adapter.

## 9.4 Reference Module

```python
from __future__ import annotations

import torch
import torch.nn as nn


class ProgressiveHyperDHead(nn.Module):
    """
    HyperD-style forecasting head plus a lightweight
    previous-state adapter.
    """

    def __init__(
        self,
        feature_dim: int,
        fc_hidden_size: int,
        chain_lengths: list[int],
        condition_hidden_size: int = 32,
        use_prev_condition: bool = True,
    ) -> None:
        super().__init__()

        self.chain_lengths = list(chain_lengths)
        self.use_prev_condition = use_prev_condition

        self.shared_hidden = nn.Sequential(
            nn.Linear(feature_dim, fc_hidden_size),
            nn.LeakyReLU(),
        )

        self.stage_output_heads = nn.ModuleDict()
        self.condition_adapters = nn.ModuleDict()

        for stage_len in self.chain_lengths:
            key = str(stage_len)

            self.stage_output_heads[key] = nn.Linear(
                fc_hidden_size,
                stage_len,
            )

            adapter = nn.Sequential(
                nn.Linear(
                    stage_len,
                    condition_hidden_size,
                ),
                nn.LeakyReLU(),
                nn.Linear(
                    condition_hidden_size,
                    stage_len,
                ),
            )

            nn.init.zeros_(adapter[-1].weight)
            nn.init.zeros_(adapter[-1].bias)

            self.condition_adapters[key] = adapter

    def forward(
        self,
        encoded_history: torch.Tensor,
        prev_residual_state: torch.Tensor,
        stage_len: int,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            encoded_history:
                [B, N, D]
            prev_residual_state:
                [B, stage_len, N]
            stage_len:
                current temporal resolution

        Returns:
            base_proposal:
                [B, stage_len, N]
            condition_delta:
                [B, stage_len, N]
            state_proposal:
                [B, stage_len, N]
            correction:
                [B, stage_len, N]
        """
        key = str(stage_len)

        if key not in self.stage_output_heads:
            raise KeyError(
                f"Unknown stage_len={stage_len}. "
                f"Available={self.chain_lengths}."
            )

        if prev_residual_state.ndim != 3:
            raise ValueError(
                "prev_residual_state must have shape [B, h, N], "
                f"got {tuple(prev_residual_state.shape)}."
            )

        hidden = self.shared_hidden(encoded_history)

        base_proposal = self.stage_output_heads[key](
            hidden
        ).transpose(1, 2)

        if self.use_prev_condition:
            condition_delta = self.condition_adapters[key](
                prev_residual_state.transpose(1, 2)
            ).transpose(1, 2)
        else:
            condition_delta = torch.zeros_like(base_proposal)

        state_proposal = base_proposal + condition_delta
        correction = state_proposal - prev_residual_state

        return {
            "base_proposal": base_proposal,
            "condition_delta": condition_delta,
            "state_proposal": state_proposal,
            "correction": correction,
        }
```

---

# 10. Main HyperD Forecast-State Chain

Create:

```text
baselines/HyperDChain/arch/hyperd_chain_arch.py
```

## 10.1 Initialization

Reuse the model arguments from the original HyperD:

```python
seq_len
pred_len
num_nodes
init_path_daily
init_path_weekly
adj
alpha
F_low
embed_size
hidden_size
fc_hidden_size
time_of_day_size
day_of_week_size
```

Add:

```python
chain_lengths=[3, 6, 12]
chain_loss_weights=[0.0, 0.0, 1.0]
use_prev_condition=True
condition_hidden_size=32
use_dual_view_loss=True
dual_view_weight=1.0
```

Require:

```python
chain_lengths[-1] == pred_len
```

Require non-decreasing positive chain lengths.

For the first PEMS04 version:

```python
pred_len = 12
chain_lengths = [3, 6, 12]
```

## 10.2 Periodic Modules

Instantiate the exact original class:

```python
self.daily_emb = Hybrid_Periodic_Pattern(...)
self.weekly_emb = Hybrid_Periodic_Pattern(...)
```

Do not rewrite it.

## 10.3 Main Forward

Reference structure:

```python
from __future__ import annotations

from argparse import Namespace

import torch
import torch.nn as nn

from baselines.HyperD.arch.hyperd_arch import (
    Hybrid_Periodic_Pattern,
)
from baselines.HyperD.arch.loss import dual_view_loss

from .progressive_head import ProgressiveHyperDHead
from .stfe_encoder import HyperDSTFEEncoder
from .temporal_ops import temporal_lift, temporal_project


class HyperDForecastStateChain(nn.Module):
    """
    HyperD backbone with temporal Forecast-State Chain.

    The periodic future is computed once. The chain progressively
    refines the residual forecast at resolutions 3, 6, and 12.
    """

    def __init__(self, **model_args) -> None:
        super().__init__()

        configs = Namespace(**model_args)

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.num_nodes = configs.num_nodes

        self.path_daily = configs.init_path_daily
        self.path_weekly = configs.init_path_weekly
        self.adj = configs.adj

        self.alpha = configs.alpha
        self.F_low = configs.F_low

        self.embed_size = configs.embed_size
        self.hidden_size = configs.hidden_size
        self.fc_hidden_size = configs.fc_hidden_size

        self.daily_len = configs.time_of_day_size
        self.weekly_len = (
            configs.day_of_week_size * self.daily_len
        )

        self.chain_lengths = list(
            getattr(configs, "chain_lengths", [3, 6, 12])
        )
        self.chain_loss_weights = list(
            getattr(
                configs,
                "chain_loss_weights",
                [0.0, 0.0, 1.0],
            )
        )
        self.use_prev_condition = bool(
            getattr(configs, "use_prev_condition", True)
        )
        self.condition_hidden_size = int(
            getattr(
                configs,
                "condition_hidden_size",
                32,
            )
        )
        self.use_dual_view_loss = bool(
            getattr(configs, "use_dual_view_loss", True)
        )
        self.dual_view_weight = float(
            getattr(configs, "dual_view_weight", 1.0)
        )

        self._validate_configuration()

        self.daily_emb = Hybrid_Periodic_Pattern(
            period_len=self.daily_len,
            num_nodes=self.num_nodes,
            adj=self.adj,
            init_npy_path=self.path_daily,
        )

        self.weekly_emb = Hybrid_Periodic_Pattern(
            period_len=self.weekly_len,
            num_nodes=self.num_nodes,
            adj=self.adj,
            init_npy_path=self.path_weekly,
        )

        self.stfe_encoder = HyperDSTFEEncoder(
            num_nodes=self.num_nodes,
            seq_len=self.seq_len,
            embed_size=self.embed_size,
            hidden_size=self.hidden_size,
        )

        self.progressive_head = ProgressiveHyperDHead(
            feature_dim=self.stfe_encoder.output_dim,
            fc_hidden_size=self.fc_hidden_size,
            chain_lengths=self.chain_lengths,
            condition_hidden_size=self.condition_hidden_size,
            use_prev_condition=self.use_prev_condition,
        )

    def _validate_configuration(self) -> None:
        if not self.chain_lengths:
            raise ValueError(
                "chain_lengths must contain at least one stage."
            )

        if any(stage_len <= 0 for stage_len in self.chain_lengths):
            raise ValueError(
                f"Invalid chain_lengths={self.chain_lengths}."
            )

        if self.chain_lengths != sorted(self.chain_lengths):
            raise ValueError(
                "chain_lengths must be non-decreasing."
            )

        if self.chain_lengths[-1] != self.pred_len:
            raise ValueError(
                f"Last chain length {self.chain_lengths[-1]} "
                f"must equal pred_len {self.pred_len}."
            )

        if len(self.chain_lengths) != len(
            self.chain_loss_weights
        ):
            raise ValueError(
                "chain_lengths and chain_loss_weights "
                "must have equal length."
            )

    def _periodic_patterns(
        self,
        history_data: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        index_daily = (
            history_data[..., 1] * self.daily_len
        )
        index_daily = index_daily[:, -1, 0]

        index_weekly = (
            history_data[..., 1] * self.daily_len
            + history_data[..., 2] * self.weekly_len
        )
        index_weekly = index_weekly[:, -1, 0]

        S_D_in = self.daily_emb(
            index_daily,
            self.seq_len,
        )
        S_W_in = self.weekly_emb(
            index_weekly,
            self.seq_len,
        )
        S_in = S_D_in + S_W_in

        S_D_out = self.daily_emb(
            (index_daily + self.seq_len) % self.daily_len,
            self.pred_len,
        )
        S_W_out = self.weekly_emb(
            (index_weekly + self.seq_len) % self.weekly_len,
            self.pred_len,
        )
        S_out = S_D_out + S_W_out

        return S_in, S_out

    def forward(
        self,
        history_data: torch.Tensor,
        future_data: torch.Tensor | None = None,
        batch_seen: int | None = None,
        epoch: int | None = None,
        train: bool = False,
        return_all: bool = False,
        **kwargs,
    ):
        del future_data, batch_seen, epoch, train, kwargs

        if history_data.ndim != 4:
            raise ValueError(
                "history_data must have shape [B, P, N, C], "
                f"got {tuple(history_data.shape)}."
            )

        if history_data.shape[-1] < 3:
            raise ValueError(
                "HyperD requires flow, time-of-day, and "
                "day-of-week input features."
            )

        x = history_data[..., 0]

        S_in, S_out = self._periodic_patterns(
            history_data
        )

        residual_in = x - S_in

        encoded_history = self.stfe_encoder(
            residual_in
        )

        residual_full = torch.zeros_like(S_out)

        stage_residual_states = []
        stage_residual_proposals = []
        stage_corrections = []
        stage_full_residuals = []
        stage_full_predictions = []
        stage_forecast_states = []
        stage_base_proposals = []
        stage_condition_deltas = []

        for stage_len in self.chain_lengths:
            prev_residual_state = temporal_project(
                residual_full,
                target_len=stage_len,
            )

            head_out = self.progressive_head(
                encoded_history=encoded_history,
                prev_residual_state=prev_residual_state,
                stage_len=stage_len,
            )

            correction = head_out["correction"]

            residual_full = (
                residual_full
                + temporal_lift(
                    correction,
                    full_horizon=self.pred_len,
                )
            )

            full_prediction = S_out + residual_full

            residual_state = temporal_project(
                residual_full,
                target_len=stage_len,
            )
            forecast_state = temporal_project(
                full_prediction,
                target_len=stage_len,
            )

            stage_base_proposals.append(
                head_out["base_proposal"]
            )
            stage_condition_deltas.append(
                head_out["condition_delta"]
            )
            stage_residual_proposals.append(
                head_out["state_proposal"]
            )
            stage_corrections.append(correction)
            stage_residual_states.append(residual_state)
            stage_full_residuals.append(residual_full)
            stage_full_predictions.append(full_prediction)
            stage_forecast_states.append(forecast_state)

        final_prediction = stage_full_predictions[-1]
        final_residual = stage_full_residuals[-1]

        if self.use_dual_view_loss:
            loss_low, loss_high = dual_view_loss(
                final_prediction,
                S_out,
                final_residual,
                self.F_low,
            )
            dva_loss = (
                loss_low + loss_high
            ) * self.alpha * self.dual_view_weight
        else:
            dva_loss = final_prediction.new_zeros(())

        output = {
            "prediction": final_prediction.unsqueeze(-1),
            "pred": final_prediction.unsqueeze(-1),
            "periodic_in": S_in.unsqueeze(-1),
            "periodic_out": S_out.unsqueeze(-1),
            "residual_in": residual_in.unsqueeze(-1),
            "residual_final": final_residual.unsqueeze(-1),
            "dual_view_loss": dva_loss,
            "stage_base_proposals": [
                tensor.unsqueeze(-1)
                for tensor in stage_base_proposals
            ],
            "stage_condition_deltas": [
                tensor.unsqueeze(-1)
                for tensor in stage_condition_deltas
            ],
            "stage_residual_proposals": [
                tensor.unsqueeze(-1)
                for tensor in stage_residual_proposals
            ],
            "stage_corrections": [
                tensor.unsqueeze(-1)
                for tensor in stage_corrections
            ],
            "stage_residual_states": [
                tensor.unsqueeze(-1)
                for tensor in stage_residual_states
            ],
            "stage_full_residuals": [
                tensor.unsqueeze(-1)
                for tensor in stage_full_residuals
            ],
            "stage_full_predictions": [
                tensor.unsqueeze(-1)
                for tensor in stage_full_predictions
            ],
            "stage_forecast_states": [
                tensor.unsqueeze(-1)
                for tensor in stage_forecast_states
            ],
            "chain_preds": [
                tensor.unsqueeze(-1)
                for tensor in stage_forecast_states
            ],
            "chain_lengths": list(self.chain_lengths),
        }

        if return_all:
            return output

        return {
            "prediction": output["prediction"],
            "dual_view_loss": output["dual_view_loss"],
        }
```

Cursor must adapt the output behavior to the actual local runner contract.

Do not silently remove the dictionary output if the original HyperD runner expects it.

---

# 11. Required State Consistency Checks

At each stage verify:

```python
torch.testing.assert_close(
    stage_residual_states[-1],
    stage_residual_proposals[-1],
    rtol=1e-5,
    atol=1e-6,
)
```

This is expected for the divisible schedules and block operators.

Also verify:

```python
stage_full_predictions[s] == periodic_out + stage_full_residuals[s]
```

within numerical tolerance.

The first stage condition should initially be zero:

```python
prev_residual_state_T3 == 0
```

because:

```python
residual_full = 0
```

The periodic baseline itself must not be repeatedly accumulated.

---

# 12. Loss Design

Create:

```text
baselines/HyperDChain/arch/chain_loss.py
```

Reuse the loss-rescaling pattern from:

```text
baselines/ForecastSpace/arch/chain_loss.py
```

Do not rewrite BasicTS rescaling logic inconsistently.

## 12.1 Final Forecasting Loss

\[
\mathcal L_{\mathrm{final}}
=
\ell
\left(
\widehat{\mathbf Y}^{(S)},
\mathbf Y
\right).
\]

Use the same `masked_mae` and scaler behavior as the original HyperD config.

## 12.2 Scale-Matched Auxiliary Forecast Loss

For stages \(s<S\):

\[
\mathbf Y_s^\star
=
D_{h_s}(\mathbf Y),
\]

and:

\[
\mathcal L_s
=
\ell
\left(
D_{h_s}(\widehat{\mathbf Y}^{(s)}),
D_{h_s}(\mathbf Y)
\right).
\]

The model already returns:

```python
stage_forecast_states[s]
```

with shape:

```python
[B, h_s, N, 1]
```

Do not compare coarse predictions directly with `[B, 12, N, 1]`.

## 12.3 DVA Loss

Use the original HyperD `dual_view_loss` only on:

- final full-resolution prediction;
- one full-horizon periodic baseline;
- final full-horizon residual prediction.

Do not apply DVA separately at lengths 3 and 6 in the first version.

The total initial loss is:

\[
\mathcal L
=
\sum_s w_s\mathcal L_s
+
\mathcal L_{\mathrm{DVA}}.
\]

Example:

```python
chain_loss_weights = [0.0, 0.0, 1.0]
```

for final-only training.

Later:

```python
chain_loss_weights = [0.1, 0.1, 1.0]
```

for scale-matched supervision.

## 12.4 Reference Loss Skeleton

```python
from __future__ import annotations

import torch

from .temporal_ops import temporal_project


def _rescale_pair(
    scaler,
    pred: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if scaler is not None and getattr(
        scaler,
        "rescale",
        False,
    ):
        pred = scaler.inverse_transform(pred.clone())
        target = scaler.inverse_transform(target.clone())

    return pred, target


def _raw_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    scaler,
    metric_forward,
    loss_fn,
) -> torch.Tensor:
    pred, target = _rescale_pair(
        scaler,
        pred,
        target,
    )

    return metric_forward(
        loss_fn,
        {
            "prediction": pred,
            "target": target,
        },
    )


def compute_hyperd_chain_loss(
    out: dict,
    real_value: torch.Tensor,
    chain_lengths: list[int],
    chain_loss_weights: list[float],
    scaler,
    metric_forward,
    loss_fn,
) -> torch.Tensor:
    if len(chain_lengths) != len(chain_loss_weights):
        raise ValueError(
            "chain_lengths and chain_loss_weights "
            "must have the same length."
        )

    stage_states = out["stage_forecast_states"]

    if len(stage_states) != len(chain_lengths):
        raise ValueError(
            "Number of returned stage states does not "
            "match chain_lengths."
        )

    total_loss = real_value.new_zeros(())

    for stage_len, weight, stage_prediction in zip(
        chain_lengths,
        chain_loss_weights,
        stage_states,
    ):
        weight = float(weight)

        if weight == 0.0:
            continue

        target_state = temporal_project(
            real_value[..., 0],
            target_len=stage_len,
        ).unsqueeze(-1)

        total_loss = total_loss + weight * _raw_loss(
            stage_prediction,
            target_state,
            scaler,
            metric_forward,
            loss_fn,
        )

    total_loss = total_loss + out["dual_view_loss"]

    return total_loss
```

Before using this exact code, verify the local scaler accepts variable temporal lengths. If it does not, use the tested `_rescale_pair` logic from the existing ForecastSpace chain loss and preserve channel layout.

---

# 13. Custom Runner

Create:

```text
baselines/HyperDChain/runner.py
```

Reuse the structure of:

```text
baselines/ForecastSpace/runner.py
```

The runner should:

1. call preprocessing;
2. select HyperD input features `[0, 1, 2]`;
3. call the model with `return_all=True`;
4. store the full output dictionary;
5. expose only final prediction to BasicTS metrics;
6. compute the chain loss plus DVA in `train_iters`;
7. preserve original postprocessing and inverse scaling.

Reference outline:

```python
from __future__ import annotations

from typing import Dict

import torch

from basicts.runners import (
    SimpleTimeSeriesForecastingRunner,
)

from .arch.chain_loss import (
    compute_hyperd_chain_loss,
)


class HyperDChainRunner(
    SimpleTimeSeriesForecastingRunner
):
    def __init__(self, cfg: Dict):
        super().__init__(cfg)

        param = cfg["MODEL"]["PARAM"]

        self.chain_lengths = list(
            param.get(
                "chain_lengths",
                [3, 6, 12],
            )
        )
        self.chain_loss_weights = list(
            param.get(
                "chain_loss_weights",
                [0.0, 0.0, 1.0],
            )
        )

        if len(self.chain_lengths) != len(
            self.chain_loss_weights
        ):
            raise ValueError(
                "chain_lengths and chain_loss_weights "
                "must have the same length."
            )

        self._last_chain_out = None
        self._last_real_value_norm = None

    def forward(
        self,
        data: Dict,
        epoch: int = None,
        iter_num: int = None,
        train: bool = True,
        **kwargs,
    ):
        data = self.preprocessing(data)

        future_data = data["target"]
        history_data = data["inputs"]

        history_data = self.to_running_device(
            history_data
        )
        future_data = self.to_running_device(
            future_data
        )

        batch_size, length, num_nodes, _ = (
            future_data.shape
        )

        history_data = self.select_input_features(
            history_data
        )
        future_data_4_dec = self.select_input_features(
            future_data
        )

        if not train:
            future_data_4_dec[..., 0] = torch.empty_like(
                future_data_4_dec[..., 0]
            )

        out = self.model(
            history_data=history_data,
            future_data=future_data_4_dec,
            batch_seen=iter_num,
            epoch=epoch,
            train=train,
            return_all=True,
        )

        self._last_chain_out = out
        self._last_real_value_norm = (
            self.select_target_features(future_data)
        )

        model_return = {
            "prediction": out["prediction"],
            "inputs": self.select_target_features(
                history_data
            ),
            "target": self._last_real_value_norm,
        }

        assert list(
            model_return["prediction"].shape
        )[:3] == [
            batch_size,
            length,
            num_nodes,
        ]

        return self.postprocessing(model_return)

    def train_iters(
        self,
        epoch: int,
        iter_index: int,
        data: Dict,
    ) -> torch.Tensor:
        iter_num = (
            (epoch - 1) * self.iter_per_epoch
            + iter_index
        )

        forward_return = self.forward(
            data=data,
            epoch=epoch,
            iter_num=iter_num,
            train=True,
        )

        loss = compute_hyperd_chain_loss(
            out=self._last_chain_out,
            real_value=self._last_real_value_norm,
            chain_lengths=self.chain_lengths,
            chain_loss_weights=self.chain_loss_weights,
            scaler=self.scaler,
            metric_forward=self.metric_forward,
            loss_fn=self.loss,
        )

        self.update_epoch_meter(
            "train/loss",
            loss.item(),
        )

        self.update_epoch_meter(
            "train/dual_view_loss",
            self._last_chain_out[
                "dual_view_loss"
            ].item(),
        )

        for metric_name, metric_func in self.metrics.items():
            metric_item = self.metric_forward(
                metric_func,
                forward_return,
            )
            self.update_epoch_meter(
                f"train/{metric_name}",
                metric_item.item(),
            )

        return loss
```

Adapt only where required by the local BasicTS runner version.

Do not change BasicTS globally to support one model.

---

# 14. PEMS04 Configuration

Create:

```text
baselines/HyperDChain/PEMS04.py
```

Start by copying:

```text
baselines/HyperD/PEMS04.py
```

Preserve:

- dataset settings;
- scaler;
- adjacency loading;
- input/output lengths;
- model dimensions;
- optimizer;
- OneCycleLR;
- batch sizes;
- metrics;
- target features;
- input features;
- seed;
- gradient clipping;
- evaluation horizons.

Change only:

```python
from .arch import HyperDForecastStateChain
from .runner import HyperDChainRunner
```

and:

```python
MODEL_ARCH = HyperDForecastStateChain
CFG.RUNNER = HyperDChainRunner
```

Add:

```python
MODEL_PARAM.update({
    "chain_lengths": [3, 6, 12],
    "chain_loss_weights": [0.0, 0.0, 1.0],
    "use_prev_condition": True,
    "condition_hidden_size": 32,
    "use_dual_view_loss": True,
    "dual_view_weight": 1.0,
})
```

Keep:

```python
CFG.MODEL.FORWARD_FEATURES = [0, 1, 2]
CFG.MODEL.TARGET_FEATURES = [0]
CFG.ENV.SEED = 1
```

Use a distinct checkpoint directory containing:

```text
HyperDChain
PEMS04
chain_3_6_12
seed_1
```

Do not overwrite original HyperD checkpoints.

---

# 15. Package Exports

Create:

```text
baselines/HyperDChain/arch/__init__.py
```

with:

```python
from .hyperd_chain_arch import (
    HyperDForecastStateChain,
)

__all__ = [
    "HyperDForecastStateChain",
]
```

Create:

```text
baselines/HyperDChain/__init__.py
```

as needed for local import behavior.

---

# 16. Implementation Phases

Do not implement everything in one uncontrolled edit.

## Phase A: Baseline Integrity

Run the original HyperD PEMS04 config before any new model training:

```bash
python train.py --cfg='baselines/HyperD/PEMS04.py'
```

A full retraining is not necessary if expensive, but at minimum verify:

- config import;
- model construction;
- one random or real batch forward;
- output keys;
- output shape;
- finite DVA loss.

Record the original model parameter count.

## Phase B: Operator and Encoder Unit Tests

Test:

- temporal projection shapes;
- temporal lifting shapes;
- `D_h(U_h(z)) = z`;
- STFE encoder output shape;
- finite encoder output;
- backward through all STFE frequency parameters.

## Phase C: Single-Stage HyperDChain Equivalence Sanity

Use:

```python
chain_lengths = [12]
chain_loss_weights = [1.0]
use_prev_condition = False
```

This model is not required to numerically equal a separately initialized HyperD instance, but it must have the same conceptual computation:

```text
periodic baseline + STFE-based 12-step residual head
```

For a stronger equivalence test, copy matching weights from an original HyperD instance:

- periodic modules;
- STFE encoder parameters;
- first FC layer;
- final 12-step FC layer.

Then compare outputs with `use_prev_condition=False`.

The outputs should match to numerical tolerance if the refactoring is correct.

Implement a temporary test utility if needed, but do not leave fragile monkey patches in production code.

## Phase D: Multi-Stage Final-Only Chain

Use:

```python
chain_lengths = [3, 6, 12]
chain_loss_weights = [0.0, 0.0, 1.0]
use_prev_condition = True
```

Verify all stages affect the final output and receive gradients.

## Phase E: Scale-Matched Supervision

Use:

```python
chain_loss_weights = [0.1, 0.1, 1.0]
```

Do not implement final-primary gradient projection yet.

## Phase F: Dataset Expansion

Only after PEMS04 works, create PEMS03/07/08 configs by copying the corresponding original HyperD configs.

---

# 17. Required Experiment Configurations

Create at least three PEMS04 configs or clearly switchable config variants.

## HC0: Single-Stage Refactored HyperD

```python
chain_lengths = [12]
chain_loss_weights = [1.0]
use_prev_condition = False
use_dual_view_loss = True
```

Purpose:

- validate that the refactored HyperD encoder/head remains competitive;
- detect implementation damage before testing the chain.

## HC1: HyperD Forecast-State Chain, Final Loss Only

```python
chain_lengths = [3, 6, 12]
chain_loss_weights = [0.0, 0.0, 1.0]
use_prev_condition = True
use_dual_view_loss = True
```

Purpose:

- test the architectural value of progressive state refinement.

## HC2: HyperD Forecast-State Chain with Intermediate Supervision

```python
chain_lengths = [3, 6, 12]
chain_loss_weights = [0.1, 0.1, 1.0]
use_prev_condition = True
use_dual_view_loss = True
```

Purpose:

- test scale-matched forecast-state supervision.

## Recommended Additional Ablation

```python
chain_lengths = [3, 6, 12]
chain_loss_weights = [0.0, 0.0, 1.0]
use_prev_condition = False
```

This isolates whether performance comes from:

- multiple temporal heads; or
- actual previous-state conditioning.

---

# 18. Tensor Shape Contract

For PEMS04:

```text
B: batch size
P: 12
H: 12
N: 307
C_in: 3
C_out: 1
```

Expected shapes:

```python
history_data:
    [B, 12, 307, 3]

S_in:
    [B, 12, 307]

S_out:
    [B, 12, 307]

residual_in:
    [B, 12, 307]

encoded_history:
    [B, 307, 12 * embed_size]

residual_full:
    [B, 12, 307]
```

Stage 3:

```python
prev_residual_state:
    [B, 3, 307]

base_proposal:
    [B, 3, 307]

state_proposal:
    [B, 3, 307]

correction:
    [B, 3, 307]

forecast_state:
    [B, 3, 307, 1]
```

Stage 6:

```python
[B, 6, 307]
```

Stage 12:

```python
[B, 12, 307]
```

Final:

```python
prediction:
    [B, 12, 307, 1]
```

Fail loudly on mismatches.

Do not crop, pad, broadcast, squeeze, or transpose silently just to make the model run.

---

# 19. Required Automated Tests

Create a focused test script such as:

```text
tests/test_hyperd_chain.py
```

or:

```text
scripts/test_hyperd_chain.py
```

depending on repository conventions.

## 19.1 Import Test

Verify:

```python
from baselines.HyperDChain.arch import (
    HyperDForecastStateChain,
)
```

## 19.2 Projection/Lifting Test

Test lengths:

```python
[3, 6, 12]
```

Verify shape and left-inverse property.

## 19.3 Forward Test

Use a small random model where possible. If the periodic module requires `.npy` files, use the actual PEMS04 initialization files or create temporary correctly shaped arrays for the unit test.

Input:

```python
history_data = torch.randn(
    2,
    12,
    307,
    3,
)
```

Ensure time features are in a valid range if required:

```python
history_data[..., 1] = torch.rand(
    2,
    12,
    307,
)
history_data[..., 2] = torch.randint(
    0,
    7,
    (2, 12, 307),
).float() / 7.0
```

Verify all returned shapes.

## 19.4 State Consistency Test

Verify:

```python
project(stage_full_residual, h)
==
stage_residual_proposal
```

for every stage.

## 19.5 Periodic Non-Duplication Test

Verify that:

```python
stage_full_prediction
==
periodic_out + stage_full_residual
```

There must be only one `periodic_out` term.

## 19.6 Backward Test

Compute:

```python
loss = (
    prediction.abs().mean()
    + output["dual_view_loss"]
)
loss.backward()
```

Verify finite, non-`None` gradients for:

- daily embedding parameters;
- weekly embedding parameters;
- STFE embeddings;
- spatial real/imaginary parameters;
- temporal real/imaginary parameters;
- shared hidden head;
- stage 3 output head;
- stage 6 output head;
- stage 12 output head;
- condition adapters when enabled.

## 19.7 Stage Contribution Test

Run a normal forward.

Then independently disable or zero each stage correction and confirm the final prediction changes.

Do not conclude that a stage is active only because it appears in a list.

## 19.8 BasicTS Dry Run

Run:

- one training batch;
- loss computation;
- backward;
- optimizer step;
- one validation batch;
- metric computation;
- checkpoint path construction.

All quantities must remain finite.

---

# 20. Diagnostic Logging

During initial debugging, log once per run:

```text
chain_lengths
chain_loss_weights
use_prev_condition
use_dual_view_loss
periodic_out shape
encoded_history shape
each stage proposal shape
each stage correction norm
each stage residual-state norm
final residual norm
final prediction shape
dual_view_loss
```

Do not print every batch indefinitely.

Prefer a debug flag:

```python
debug_shapes=False
```

or one-time logging.

Useful diagnostics:

```python
correction.abs().mean()
condition_delta.abs().mean()
base_proposal.abs().mean()
residual_state.abs().mean()
```

This helps detect:

- condition adapter remaining exactly zero;
- one stage dominating;
- exploding residual accumulation;
- periodic output accidentally repeated.

---

# 21. Forbidden Changes

Do not add:

- graph resolution;
- graph clustering;
- graph pooling;
- adaptive graph;
- dynamic graph;
- spectral clustering;
- KASA `TemporalStep`;
- ForecastSpace spatial modules;
- GCN beyond HyperD's existing periodic adjacency operation;
- new FFT branches;
- wavelets;
- seasonal-trend decomposition;
- prior mapper;
- KAN;
- PTSE;
- DSHP;
- MTSR-P;
- Transformer;
- RNN;
- SSM;
- learnable temporal lifting;
- stage gates;
- gradient projection;
- uncertainty heads;
- extra datasets;
- changed data splits;
- changed normalization;
- changed metrics;
- changed optimizer for the first comparison.

Do not modify original HyperD performance settings in HC0.

Do not remove HyperD's daily/weekly features.

Do not add `try/except` blocks that suppress shape or import errors.

Do not silently fall back to an unrelated module.

---

# 22. What May Be Changed Later, But Not Now

After HC0–HC2 are stable, later experiments may consider:

- separate vs shared hidden FC;
- condition adapter hidden size;
- condition adapter on forecast state instead of residual state;
- linear interpolation vs block repetition;
- chain schedules `[3, 12]`, `[6, 12]`, `[2, 4, 6, 12]`;
- DVA on/off;
- final-primary gradient projection;
- graph-resolution stages.

Do not implement these in the first pass.

---

# 23. Acceptance Criteria

The implementation is accepted only if all conditions are satisfied.

## Code Reuse

- Original HyperD directory remains runnable.
- `Hybrid_Periodic_Pattern` is directly reused.
- Original `dual_view_loss` is directly reused.
- STFE frequency computations are copied without mathematical changes.
- Original HyperD FC architecture is retained in the new head.
- No unrelated backbone is added.

## Mathematical Correctness

- Future periodic baseline is generated once at length 12.
- Residual history is `x - S_in`.
- The chain state is an accumulated full-horizon residual forecast.
- Current residual state is projected to each stage.
- Each stage creates a conditioned residual-state proposal.
- Correction is proposal minus projected previous state.
- Correction is lifted and accumulated.
- Full forecast equals periodic baseline plus accumulated residual.
- Intermediate forecast supervision is scale matched.
- DVA is applied only at the final full resolution.

## Engineering Correctness

- HC0 imports and runs.
- HC1 imports and runs.
- HC2 imports and runs.
- All stage shapes are correct.
- Every stage contributes to final prediction.
- All stage heads receive finite gradients.
- Condition adapters receive gradients when enabled.
- Original HyperD config still runs.
- Original ForecastSpace config still imports.
- No unrelated files are modified.
- Full training commands are provided.

---

# 24. Required Final Report from Cursor

After implementation, report:

1. current branch and commit before editing;
2. local repository structure found;
3. exact original HyperD files reused by import;
4. exact STFE code copied and what was minimally changed;
5. every new file;
6. every modified existing file;
7. whether original HyperD was modified;
8. parameter count of original HyperD;
9. parameter count of HC0;
10. parameter count of HC1;
11. projection/lifting test results;
12. forward shape test results;
13. state consistency test results;
14. backward gradient test results;
15. BasicTS dry-run result;
16. commands for HC0, HC1, HC2;
17. any unresolved issue;
18. confirmation that no graph-resolution component was added.

Do not claim completion if full training has not been run. Clearly distinguish:

- static checks;
- random tensor tests;
- one-batch dry run;
- one-epoch run;
- full experiment.

---

# 25. Example Commands

Use the actual filenames created locally.

Suggested naming:

```bash
python train.py --cfg='baselines/HyperDChain/PEMS04_HC0.py'
python train.py --cfg='baselines/HyperDChain/PEMS04_HC1.py'
python train.py --cfg='baselines/HyperDChain/PEMS04_HC2.py'
```

If only one config is created initially, expose experiment flags clearly and provide three exact commands or three small wrapper configs.

---

# 26. Final Conceptual Summary

The implementation should realize:

\[
\boxed{
\begin{aligned}
\mathbf S_{\mathrm{in}},
\mathbf S_{\mathrm{out}}
&=
\operatorname{HyperPeriodic}(\mathbf X),\\
\mathbf H_R
&=
E_{\mathrm{STFE}}
\left(
\mathbf X-\mathbf S_{\mathrm{in}}
\right),\\
\widehat{\mathbf R}^{(0)}
&=
\mathbf 0,\\
\overline{\mathbf R}_s
&=
D_{h_s}
\left(
\widehat{\mathbf R}^{(s-1)}
\right),\\
\widetilde{\mathbf R}_s
&=
G_s^{\mathrm{HyperD}}(\mathbf H_R)
+
A_s(\overline{\mathbf R}_s),\\
\Delta\mathbf R_s
&=
\widetilde{\mathbf R}_s
-
\overline{\mathbf R}_s,\\
\widehat{\mathbf R}^{(s)}
&=
\widehat{\mathbf R}^{(s-1)}
+
U_{h_s}(\Delta\mathbf R_s),\\
\widehat{\mathbf Y}^{(s)}
&=
\mathbf S_{\mathrm{out}}
+
\widehat{\mathbf R}^{(s)}.
\end{aligned}
}
\]

HyperD determines how strong temporal residual proposals are produced.

Forecast-State Chain determines how explicit predictions evolve across temporal resolutions.

That division of responsibility must remain visible in both the code and the experiment design.
