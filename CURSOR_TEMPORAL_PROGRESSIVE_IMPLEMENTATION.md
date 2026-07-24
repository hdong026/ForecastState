# Cursor Implementation Task: Temporal-Only Progressive Forecasting

## Objective

Implement the first-stage version of **Forecast-State Chain** as a **temporal-only progressive forecasting model**.

The target schedule is:

\[
T_3 \rightarrow T_6 \rightarrow T_{12}
\]

For a forecasting horizon \(H=12\), each stage predicts a residual correction in an explicit temporal forecast space. Graph resolution must remain unchanged throughout this implementation.

Do **not** implement graph clustering, graph pooling, adaptive graph learning, spatial downsampling, or any other graph-resolution component in this task.

---

## Core Principle

The implementation must reuse the existing, verified **KASA-ST TemporalStep**.

Do not redesign the temporal block.

Do not replace it with Transformer, GRU/LSTM, generic MLP, a new temporal convolution, frequency branch, decomposition module, state-space model, or any other new temporal architecture.

The goal is to validate the progressive forecast-state formulation, not to introduce a new temporal backbone.

---

## Mathematical Formulation

Let the forecasting horizon be \(H=12\), and let the temporal resolution schedule be:

\[
\mathcal H=[3,6,12].
\]

Initialize:

\[
\widehat{\mathbf Y}^{(0)}=\mathbf 0.
\]

At stage \(s\), with temporal resolution \(h_s\):

\[
\overline{\mathbf Z}_s=D_{h_s}\left(\widehat{\mathbf Y}^{(s-1)}\right),
\]

\[
\mathbf R_s=F_s\left(\mathbf X,\overline{\mathbf Z}_s\right)
\in\mathbb R^{B\times h_s\times N\times C_y},
\]

\[
\Delta\widehat{\mathbf Y}^{(s)}=U_{h_s}(\mathbf R_s),
\]

\[
\widehat{\mathbf Y}^{(s)}=
\widehat{\mathbf Y}^{(s-1)}+
\alpha_s\Delta\widehat{\mathbf Y}^{(s)}.
\]

For the first implementation, fix \(\alpha_s=1\). The final prediction is:

\[
\widehat{\mathbf Y}=\widehat{\mathbf Y}^{(S)}.
\]

---

## Important Terminology

Do not call \(\mathbf R_s\) the complete forecast state. It is a **resolution-specific forecast residual** or **forecast correction**.

The forecast state at stage \(s\) is:

\[
\mathbf S_s=D_{h_s}\left(\widehat{\mathbf Y}^{(s)}\right).
\]

Therefore:

- `stage_residuals[s]` is \(\mathbf R_s\);
- `stage_predictions[s]` is \(\widehat{\mathbf Y}^{(s)}\);
- `stage_states[s]` is \(\mathbf S_s\).

---

## Required Tensor Shapes

```python
history_data.shape == [B, P, N, C_in]
prediction.shape == [B, 12, N, C_out]
```

For schedule `[3, 6, 12]`:

```python
stage_residuals[0].shape == [B, 3,  N, C_out]
stage_residuals[1].shape == [B, 6,  N, C_out]
stage_residuals[2].shape == [B, 12, N, C_out]

stage_predictions[s].shape == [B, 12, N, C_out]

stage_states[0].shape == [B, 3,  N, C_out]
stage_states[1].shape == [B, 6,  N, C_out]
stage_states[2].shape == [B, 12, N, C_out]
```

---

# Part 1: Inspect the Repository Before Editing

Before changing code:

1. Inspect the complete repository structure.
2. Locate the model already migrated from KASA-ST.
3. Locate the exact KASA-ST `TemporalStep` used by the verified KASA-ST model.
4. Identify all direct dependencies of `TemporalStep`.
5. Identify the BasicTS model `forward` interface.
6. Identify how the runner extracts predictions and computes loss.
7. Identify whether model outputs must be tensors or may be dictionaries.
8. Identify the configuration layout and experiment launch command.
9. Do not infer interfaces from filenames alone; read the implementation.

Report these findings before making broad architectural changes.

---

# Part 2: Suggested File Structure

```text
models/
└── ForecastState/
    ├── __init__.py
    ├── progressive_temporal.py
    ├── temporal_ops.py
    ├── kasa_temporal_step.py
    └── losses.py
```

Adapt exact paths to repository conventions. Do not break HyperD or move unrelated files.

---

# Part 3: Temporal Projection

When \(H\) is divisible by `target_length`, use non-overlapping block averaging:

- `12 -> 3`: average every 4 consecutive steps;
- `12 -> 6`: average every 2 consecutive steps;
- `12 -> 12`: identity.

Use adaptive average pooling only when lengths are not divisible.

```python
from __future__ import annotations

import torch
import torch.nn.functional as F


def temporal_project(
    y: torch.Tensor,
    target_length: int,
) -> torch.Tensor:
    """Project [B, H, N, C] to [B, target_length, N, C]."""
    if y.ndim != 4:
        raise ValueError(
            f"Expected [B, H, N, C], got {tuple(y.shape)}."
        )

    batch_size, horizon, num_nodes, channels = y.shape

    if not 1 <= target_length <= horizon:
        raise ValueError(
            f"target_length must be in [1, {horizon}], got {target_length}."
        )

    if target_length == horizon:
        return y

    if horizon % target_length == 0:
        block_size = horizon // target_length
        return y.reshape(
            batch_size,
            target_length,
            block_size,
            num_nodes,
            channels,
        ).mean(dim=2)

    y_pool = y.permute(0, 2, 3, 1).reshape(
        batch_size * num_nodes,
        channels,
        horizon,
    )
    y_pool = F.adaptive_avg_pool1d(y_pool, target_length)

    return y_pool.reshape(
        batch_size,
        num_nodes,
        channels,
        target_length,
    ).permute(0, 3, 1, 2).contiguous()
```

---

# Part 4: Temporal Lifting

When divisible, use `repeat_interleave`:

- `3 -> 12`: repeat each step 4 times;
- `6 -> 12`: repeat each step 2 times;
- `12 -> 12`: identity.

Use linear interpolation only when lengths are not divisible.

```python
def temporal_lift(
    z: torch.Tensor,
    full_horizon: int,
) -> torch.Tensor:
    """Lift [B, h, N, C] to [B, H, N, C]."""
    if z.ndim != 4:
        raise ValueError(
            f"Expected [B, h, N, C], got {tuple(z.shape)}."
        )

    batch_size, current_length, num_nodes, channels = z.shape

    if current_length > full_horizon:
        raise ValueError(
            f"Cannot lift length {current_length} to {full_horizon}."
        )

    if current_length == full_horizon:
        return z

    if full_horizon % current_length == 0:
        repeat_factor = full_horizon // current_length
        return z.repeat_interleave(repeat_factor, dim=1)

    z_interp = z.permute(0, 2, 3, 1).reshape(
        batch_size * num_nodes,
        channels,
        current_length,
    )
    z_interp = F.interpolate(
        z_interp,
        size=full_horizon,
        mode="linear",
        align_corners=False,
    )

    return z_interp.reshape(
        batch_size,
        num_nodes,
        channels,
        full_horizon,
    ).permute(0, 3, 1, 2).contiguous()
```

---

# Part 5: KASA TemporalStep Adapter

Reuse the exact KASA-ST temporal implementation. If its interface differs, add only a thin adapter.

The adapter may rename arguments, reshape tensors, select output channels, support variable `target_length`, inject `prev_condition` according to the existing KASA design, and restore BasicTS layout.

It must not replace or redesign KASA temporal computation.

Expected interface:

```python
residual = temporal_step(
    history_data=history_data,
    prev_condition=prev_condition,
    target_length=resolution,
    batch_seen=batch_seen,
    epoch=epoch,
    train=train,
)
```

Expected output:

```python
residual.shape == [B, resolution, N, C_out]
```

Adapter skeleton:

```python
from __future__ import annotations

import torch
import torch.nn as nn


class KASATemporalStepAdapter(nn.Module):
    """Thin adapter around the verified KASA-ST TemporalStep."""

    def __init__(self, kasa_step: nn.Module, output_channels: int) -> None:
        super().__init__()
        self.kasa_step = kasa_step
        self.output_channels = output_channels

    def forward(
        self,
        history_data: torch.Tensor,
        prev_condition: torch.Tensor,
        target_length: int,
        batch_seen: int | None = None,
        epoch: int | None = None,
        train: bool = True,
    ) -> torch.Tensor:
        # Replace this call with the actual inspected KASA-ST interface.
        residual = self.kasa_step(
            history_data=history_data,
            prev_condition=prev_condition,
            target_length=target_length,
            batch_seen=batch_seen,
            epoch=epoch,
            train=train,
        )

        expected_shape = (
            history_data.shape[0],
            target_length,
            history_data.shape[2],
            self.output_channels,
        )

        if residual.shape != expected_shape:
            raise RuntimeError(
                f"KASA TemporalStep returned {tuple(residual.shape)}, "
                f"expected {expected_shape}."
            )

        return residual
```

Do not leave a placeholder interface in the final implementation. Replace it after inspecting the real KASA-ST code.

---

# Part 6: Progressive Temporal Model

```python
from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from .temporal_ops import temporal_lift, temporal_project


class ProgressiveTemporalForecasting(nn.Module):
    """Temporal-only Forecast-State Chain."""

    def __init__(
        self,
        temporal_steps: Sequence[nn.Module],
        temporal_resolutions: Sequence[int],
        full_horizon: int,
        output_channels: int = 1,
        use_prev_condition: bool = True,
        learnable_stage_scale: bool = False,
    ) -> None:
        super().__init__()

        temporal_resolutions = list(temporal_resolutions)

        if not temporal_resolutions:
            raise ValueError("temporal_resolutions cannot be empty.")

        if len(temporal_steps) != len(temporal_resolutions):
            raise ValueError(
                "The number of temporal steps must equal the number "
                "of temporal resolutions."
            )

        if temporal_resolutions != sorted(temporal_resolutions):
            raise ValueError("Temporal resolutions must be non-decreasing.")

        if temporal_resolutions[-1] != full_horizon:
            raise ValueError(
                "The final temporal resolution must equal full_horizon."
            )

        self.temporal_steps = nn.ModuleList(temporal_steps)
        self.temporal_resolutions = temporal_resolutions
        self.full_horizon = full_horizon
        self.output_channels = output_channels
        self.use_prev_condition = use_prev_condition

        if learnable_stage_scale:
            self.stage_scales = nn.Parameter(
                torch.ones(len(temporal_resolutions))
            )
        else:
            self.register_buffer(
                "stage_scales",
                torch.ones(len(temporal_resolutions)),
                persistent=False,
            )

        self.latest_stage_residuals: list[torch.Tensor] = []
        self.latest_stage_predictions: list[torch.Tensor] = []
        self.latest_stage_states: list[torch.Tensor] = []

    def forward(
        self,
        history_data: torch.Tensor,
        future_data: torch.Tensor | None = None,
        batch_seen: int | None = None,
        epoch: int | None = None,
        train: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        if history_data.ndim != 4:
            raise ValueError(
                "history_data must have shape [B, P, N, C_in], "
                f"got {tuple(history_data.shape)}."
            )

        batch_size, _, num_nodes, _ = history_data.shape

        full_forecast = history_data.new_zeros(
            batch_size,
            self.full_horizon,
            num_nodes,
            self.output_channels,
        )

        stage_residuals = []
        stage_predictions = []
        stage_states = []

        for stage_index, (resolution, temporal_step) in enumerate(
            zip(self.temporal_resolutions, self.temporal_steps)
        ):
            prev_condition = temporal_project(
                full_forecast,
                target_length=resolution,
            )

            if not self.use_prev_condition:
                prev_condition = torch.zeros_like(prev_condition)

            residual = temporal_step(
                history_data=history_data,
                prev_condition=prev_condition,
                target_length=resolution,
                batch_seen=batch_seen,
                epoch=epoch,
                train=train,
            )

            expected_shape = (
                batch_size,
                resolution,
                num_nodes,
                self.output_channels,
            )
            if residual.shape != expected_shape:
                raise RuntimeError(
                    f"Stage {stage_index} returned {tuple(residual.shape)}, "
                    f"expected {expected_shape}."
                )

            lifted_residual = temporal_lift(
                residual,
                full_horizon=self.full_horizon,
            )

            full_forecast = (
                full_forecast
                + self.stage_scales[stage_index] * lifted_residual
            )

            forecast_state = temporal_project(
                full_forecast,
                target_length=resolution,
            )

            stage_residuals.append(residual)
            stage_predictions.append(full_forecast)
            stage_states.append(forecast_state)

        self.latest_stage_residuals = stage_residuals
        self.latest_stage_predictions = stage_predictions
        self.latest_stage_states = stage_states

        return full_forecast
```

---

# Part 7: Intermediate Supervision

The first runnable version must support final-only training. Do not implement final-primary gradient projection yet.

Optional scale-matched auxiliary supervision:

\[
\mathcal L_s=
\ell\left(
D_{h_s}(\widehat{\mathbf Y}^{(s)}),
D_{h_s}(\mathbf Y)
\right).
\]

Initial diagnostic total loss:

\[
\mathcal L=
\mathcal L_{\mathrm{final}}+
\lambda_{\mathrm{aux}}
\frac{1}{S-1}
\sum_{s=1}^{S-1}\mathcal L_s.
\]

Use `AUX_LOSS_WEIGHT = 0.1` only for E2.

```python
from __future__ import annotations

import torch

from .temporal_ops import temporal_project


def progressive_temporal_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    stage_predictions: list[torch.Tensor],
    temporal_resolutions: list[int],
    masked_mae_fn,
    null_val: float = 0.0,
    aux_weight: float = 0.0,
) -> torch.Tensor:
    final_loss = masked_mae_fn(
        prediction,
        target,
        null_val=null_val,
    )

    if aux_weight <= 0.0 or len(stage_predictions) <= 1:
        return final_loss

    auxiliary_losses = []

    for stage_prediction, resolution in zip(
        stage_predictions[:-1],
        temporal_resolutions[:-1],
    ):
        prediction_state = temporal_project(
            stage_prediction,
            target_length=resolution,
        )
        target_state = temporal_project(
            target,
            target_length=resolution,
        )
        auxiliary_losses.append(
            masked_mae_fn(
                prediction_state,
                target_state,
                null_val=null_val,
            )
        )

    return final_loss + aux_weight * torch.stack(auxiliary_losses).mean()
```

Never compare a `[B, 3, N, C]` or `[B, 6, N, C]` state directly with a `[B, 12, N, C]` target.

---

# Part 8: Configuration

Add options equivalent to:

```python
TEMPORAL_RESOLUTIONS = [3, 6, 12]
USE_PREV_CONDITION = True
STAGE_SCALE = 1.0
LEARNABLE_STAGE_SCALE = False
AUX_LOSS_WEIGHT = 0.0
```

Do not hard-code dataset-specific node counts inside the model.

---

# Part 9: Required Experiments

## E0: Single-Stage Temporal Baseline

```python
TEMPORAL_RESOLUTIONS = [12]
USE_PREV_CONDITION = False
AUX_LOSS_WEIGHT = 0.0
```

Purpose: verify the migrated KASA TemporalStep and establish a single-stage temporal baseline.

## E1: Progressive Temporal Forecasting

```python
TEMPORAL_RESOLUTIONS = [3, 6, 12]
USE_PREV_CONDITION = True
AUX_LOSS_WEIGHT = 0.0
```

Purpose: isolate the architectural effect of the progressive residual chain.

## E2: Progressive Forecasting with Scale-Matched Supervision

```python
TEMPORAL_RESOLUTIONS = [3, 6, 12]
USE_PREV_CONDITION = True
AUX_LOSS_WEIGHT = 0.1
```

Purpose: evaluate direct supervision of explicit forecast states. Do not add gradient projection in E2.

---

# Part 10: Required Validation

## Static Checks

1. Run `py_compile` on every new or modified Python file.
2. Verify all imports.
3. Confirm no circular imports.
4. Confirm existing HyperD configs still import.
5. Confirm no unrelated files were modified.

## Forward Shape Test

```python
history_data = torch.randn(2, 12, 307, 1)
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
```

## Backward Test

```python
target = torch.randn_like(prediction)
loss = torch.mean(torch.abs(prediction - target))
loss.backward()
```

Verify:

1. every KASA TemporalStep has parameters with non-`None` gradients;
2. all gradients are finite;
3. T3, T6, and T12 parameters all receive gradients;
4. no stage is disconnected from the final prediction.

## Contribution Test

Verify that T3 and T6 affect the final output by temporarily zeroing each stage residual and confirming the final prediction changes.

## BasicTS Dry Run

Run one training batch, loss, backward, optimizer step, and validation batch. Confirm output extraction and inverse scaling remain correct.

---

# Part 11: Explicitly Forbidden Changes

Do not add:

- graph convolution;
- graph clustering or pooling;
- graph resolution;
- adaptive or dynamic adjacency;
- spectral clustering;
- spatial attention;
- frequency branch;
- prior mapper or KAN prior;
- DSHP, PTSE, or MTSR-P;
- decomposition;
- new encoder or decoder unrelated to the thin KASA adapter;
- learnable lifting;
- gradient projection or manual gradient surgery;
- dataset split, metric, or normalization changes;
- hidden exception handling;
- silent tensor cropping or broadcasting;
- `try/except` blocks that suppress shape errors.

Fail loudly with informative errors.

---

# Part 12: Deliverables

After implementation, provide:

1. repository inspection summary;
2. exact KASA-ST TemporalStep source location;
3. copied or adapted dependencies;
4. every modified file;
5. every newly created file;
6. concise data-flow explanation;
7. forward shape-test output;
8. backward gradient-test output;
9. BasicTS dry-run output;
10. complete E0, E1, and E2 commands;
11. unresolved incompatibilities;
12. confirmation that no graph-resolution code was added.

Do not expand the method beyond this task after the implementation passes.

---

# Final Acceptance Criteria

The task is complete only when:

- the exact KASA-ST TemporalStep is reused;
- `[12]` and `[3, 6, 12]` are both supported;
- `use_prev_condition=True` works;
- the accumulated forecast is projected into each current stage;
- every stage predicts a residual in its own temporal resolution;
- every residual is lifted and added to the full forecast;
- the final output is `[B, 12, N, C_out]`;
- T3, T6, and T12 all receive finite gradients;
- E0, E1, and E2 have independent runnable configs;
- E2 uses scale-matched targets;
- no graph-resolution component or unrelated architecture is introduced.
