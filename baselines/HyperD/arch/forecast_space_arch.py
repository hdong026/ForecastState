"""ForecastSpace entry model for HyperD / BasicTS."""

from torch import nn

from .forecast_state_chain import ForecastStateChain


class ForecastSpace(nn.Module):
    """G1_final_adaptive ForecastSpace variant integrated under baselines/HyperD/."""

    def __init__(self, **model_args):
        super().__init__()
        self.core = ForecastStateChain(**model_args)
        self.chain_lengths = list(model_args.get("chain_lengths", [3, 6, 12]))
        self.chain_loss_weights = list(model_args.get("chain_loss_weights", [0.2, 0.3, 1.0]))

    def forward(
        self,
        history_data,
        future_data=None,
        batch_seen: int = 0,
        epoch: int = 0,
        train: bool = False,
        return_all: bool = False,
        **kwargs,
    ):
        return self.core(
            history_data=history_data,
            future_data=future_data,
            batch_seen=batch_seen,
            epoch=epoch,
            train=train,
            return_all=return_all,
            **kwargs,
        )

    @staticmethod
    def pool_target(future_target, target_len: int):
        return ForecastStateChain.pool_target(future_target, target_len)
