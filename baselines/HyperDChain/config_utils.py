"""Shared PEMS04 config builder for HyperDChain HC0/HC1/HC2."""
from __future__ import annotations

import math
import os
import sys

import torch
from easydict import EasyDict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from basicts.data import TimeSeriesForecastingDataset
from basicts.metrics import masked_mae, masked_mape, masked_rmse
from basicts.scaler import ZScoreScaler
from basicts.utils import get_regular_settings, load_adj, load_dataset_desc

from .arch import HyperDForecastStateChain
from .runner import HyperDChainRunner


def build_pems04_cfg(
    experiment_name: str,
    chain_lengths: list[int],
    chain_loss_weights: list[float],
    use_prev_condition: bool,
    use_dual_view_loss: bool = True,
    dual_view_weight: float = 1.0,
    condition_hidden_size: int = 32,
) -> EasyDict:
    data_name = "PEMS04"
    regular_settings = get_regular_settings(data_name)
    input_len = regular_settings["INPUT_LEN"]
    output_len = regular_settings["OUTPUT_LEN"]
    train_val_test_ratio = regular_settings["TRAIN_VAL_TEST_RATIO"]
    norm_each_channel = regular_settings["NORM_EACH_CHANNEL"]
    rescale = regular_settings["RESCALE"]
    null_val = regular_settings["NULL_VAL"]
    num_epochs = 100

    adj_mx, _ = load_adj(f"datasets/{data_name}/adj_mx.pkl", "normlap")

    model_param = {
        "seq_len": input_len,
        "pred_len": output_len,
        "num_nodes": 307,
        "init_path_daily": "datasets/PEMS04/daily_init.npy",
        "init_path_weekly": "datasets/PEMS04/weekly_init.npy",
        "adj": torch.tensor(adj_mx[0]),
        "alpha": 2,
        "F_low": 3,
        "embed_size": 64,
        "hidden_size": 128,
        "fc_hidden_size": 128,
        "time_of_day_size": 288,
        "day_of_week_size": 7,
        "chain_lengths": list(chain_lengths),
        "chain_loss_weights": list(chain_loss_weights),
        "use_prev_condition": bool(use_prev_condition),
        "condition_hidden_size": int(condition_hidden_size),
        "use_dual_view_loss": bool(use_dual_view_loss),
        "dual_view_weight": float(dual_view_weight),
    }

    cfg = EasyDict()
    cfg.DESCRIPTION = f"HyperDChain PEMS04 {experiment_name}"
    cfg.GPU_NUM = 1
    cfg.RUNNER = HyperDChainRunner

    cfg.ENV = EasyDict()
    cfg.ENV.SEED = 1

    cfg.DATASET = EasyDict()
    cfg.DATASET.NAME = data_name
    cfg.DATASET.TYPE = TimeSeriesForecastingDataset
    cfg.DATASET.PARAM = EasyDict({
        "dataset_name": data_name,
        "train_val_test_ratio": train_val_test_ratio,
        "input_len": input_len,
        "output_len": output_len,
    })

    cfg.SCALER = EasyDict()
    cfg.SCALER.TYPE = ZScoreScaler
    cfg.SCALER.PARAM = EasyDict({
        "dataset_name": data_name,
        "train_ratio": train_val_test_ratio[0],
        "norm_each_channel": norm_each_channel,
        "rescale": rescale,
    })

    cfg.MODEL = EasyDict()
    cfg.MODEL.NAME = HyperDForecastStateChain.__name__
    cfg.MODEL.ARCH = HyperDForecastStateChain
    cfg.MODEL.PARAM = model_param
    cfg.MODEL.FORWARD_FEATURES = [0, 1, 2]
    cfg.MODEL.TARGET_FEATURES = [0]

    cfg.METRICS = EasyDict()
    cfg.METRICS.FUNCS = EasyDict({
        "MAE": masked_mae,
        "RMSE": masked_rmse,
        "MAPE": masked_mape,
    })
    cfg.METRICS.TARGET = "MAE"
    cfg.METRICS.NULL_VAL = null_val

    chain_tag = "_".join(str(h) for h in chain_lengths)
    cfg.TRAIN = EasyDict()
    cfg.TRAIN.NUM_EPOCHS = num_epochs
    cfg.TRAIN.CKPT_SAVE_DIR = os.path.join(
        "checkpoints",
        "HyperDChain",
        data_name,
        f"{experiment_name}_chain_{chain_tag}_seed_1",
        f"{num_epochs}_{input_len}_{output_len}",
    )
    cfg.TRAIN.LOSS = masked_mae
    cfg.TRAIN.OPTIM = EasyDict()
    cfg.TRAIN.OPTIM.TYPE = "Adam"
    cfg.TRAIN.OPTIM.PARAM = {"lr": 0.005}
    cfg.TRAIN.LR_SCHEDULER = EasyDict()
    desc = load_dataset_desc(data_name)
    train_steps = math.ceil(desc["num_time_steps"] * train_val_test_ratio[0])
    cfg.TRAIN.LR_SCHEDULER.TYPE = "OneCycleLR"
    cfg.TRAIN.LR_SCHEDULER.PARAM = {
        "pct_start": 0.3,
        "epochs": num_epochs,
        "steps_per_epoch": train_steps,
        "max_lr": cfg.TRAIN.OPTIM.PARAM["lr"],
    }
    cfg.TRAIN.CLIP_GRAD_PARAM = {"max_norm": 5.0}
    cfg.TRAIN.DATA = EasyDict()
    cfg.TRAIN.DATA.BATCH_SIZE = 64
    cfg.TRAIN.DATA.SHUFFLE = True

    cfg.VAL = EasyDict()
    cfg.VAL.INTERVAL = 1
    cfg.VAL.DATA = EasyDict()
    cfg.VAL.DATA.BATCH_SIZE = 64

    cfg.TEST = EasyDict()
    cfg.TEST.INTERVAL = 1
    cfg.TEST.DATA = EasyDict()
    cfg.TEST.DATA.BATCH_SIZE = 64

    cfg.EVAL = EasyDict()
    cfg.EVAL.HORIZONS = [3, 6, 12]
    cfg.EVAL.USE_GPU = True

    return cfg
