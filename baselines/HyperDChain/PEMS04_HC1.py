"""HC1: HyperD Forecast-State Chain [3,6,12], final loss only."""
from .config_utils import build_pems04_cfg

CFG = build_pems04_cfg(
    experiment_name="HC1",
    chain_lengths=[3, 6, 12],
    chain_loss_weights=[0.0, 0.0, 1.0],
    use_prev_condition=True,
)
