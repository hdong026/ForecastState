"""E2: progressive forecasting with scale-matched auxiliary supervision."""
from .config_utils import build_pems04_cfg

CFG = build_pems04_cfg(
    experiment_name="E2_progressive_aux",
    temporal_resolutions=[3, 6, 12],
    use_prev_condition=True,
    aux_loss_weight=0.1,
)
