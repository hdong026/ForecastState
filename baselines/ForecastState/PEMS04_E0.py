"""E0: single-stage temporal baseline [12], no prev condition, final-only loss."""
from .config_utils import build_pems04_cfg

CFG = build_pems04_cfg(
    experiment_name="E0_single_stage",
    temporal_resolutions=[12],
    use_prev_condition=False,
    aux_loss_weight=0.0,
)
