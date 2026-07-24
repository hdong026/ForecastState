"""Re-export the verified KASA TemporalStep without redesign.

Source of truth in this repository:
  baselines/ForecastSpace/arch/temporal_step.py

That file is the migrated KASA-ST TemporalStep used by ForecastSpace.
This module only re-exports it so ForecastState depends on the exact
implementation rather than a rewritten temporal backbone.
"""
from __future__ import annotations

from baselines.ForecastSpace.arch.temporal_step import (
    DownsampEncoder,
    KASATemporalStep,
    MultiLayerPerceptron,
    PatchEncoder,
    interpolate_forecast,
)

__all__ = [
    "KASATemporalStep",
    "PatchEncoder",
    "DownsampEncoder",
    "MultiLayerPerceptron",
    "interpolate_forecast",
]
