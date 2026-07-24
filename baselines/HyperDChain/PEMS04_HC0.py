"""HC0: single-stage refactored HyperD [12], no prev condition."""
from .config_utils import build_pems04_cfg

CFG = build_pems04_cfg(
    experiment_name="HC0",
    chain_lengths=[12],
    chain_loss_weights=[1.0],
    use_prev_condition=False,
)
