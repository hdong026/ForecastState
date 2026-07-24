"""E1: progressive temporal forecasting [3,6,12], prev condition, final-only loss."""
from .config_utils import build_pems04_cfg

CFG = build_pems04_cfg(
    experiment_name="E1_progressive",
    temporal_resolutions=[3, 6, 12],
    use_prev_condition=True,
    aux_loss_weight=0.0,
)
