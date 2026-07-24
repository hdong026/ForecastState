"""Shared PEMS04 config builder for ForecastState progressive experiments."""
from __future__ import annotations

import math
import os
import sys

from easydict import EasyDict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from basicts.data import TimeSeriesForecastingDataset
from basicts.metrics import masked_mae, masked_mape, masked_rmse
from basicts.scaler import ZScoreScaler
from basicts.utils import get_regular_settings, load_dataset_desc

from baselines.ForecastState.model import ForecastStateProgressive
from baselines.ForecastState.runner import ForecastStateProgressiveRunner


def build_pems04_cfg(
    *,
    experiment_name: str,
    temporal_resolutions: list[int],
    use_prev_condition: bool,
    aux_loss_weight: float,
    num_epochs: int = 100,
    seed: int = 1,
) -> EasyDict:
    data_name = "PEMS04"
    regular_settings = get_regular_settings(data_name)
    input_len = regular_settings["INPUT_LEN"]
    output_len = regular_settings["OUTPUT_LEN"]
    train_val_test_ratio = regular_settings["TRAIN_VAL_TEST_RATIO"]
    norm_each_channel = regular_settings["NORM_EACH_CHANNEL"]
    rescale = regular_settings["RESCALE"]
    null_val = regular_settings["NULL_VAL"]
    batch_size = 64

    model_param = {
        "num_nodes": 307,
        "input_len": input_len,
        "output_len": output_len,
        "output_dim": 1,
        "temporal_resolutions": list(temporal_resolutions),
        "use_prev_condition": bool(use_prev_condition),
        "learnable_stage_scale": False,
        "aux_loss_weight": float(aux_loss_weight),
        "patch_len": 3,
        "stride": 4,
        "td_size": 288,
        "dw_size": 7,
        "d_td": 32,
        "d_dw": 32,
        "d_d": 32,
        "d_spa": 32,
        "num_layer": 2,
        "if_time_in_day": True,
        "if_day_in_week": True,
        "if_spatial": True,
        "use_patch_branch": True,
        "use_downsample_branch": True,
        "use_linear_residual_branch": True,
        "patch_data_input_mode": "all",
        "patch_embedding_mode": "serial_concat",
    }

    cfg = EasyDict()
    cfg.DESCRIPTION = (
        f"ForecastState temporal-only progressive {experiment_name} "
        f"resolutions={temporal_resolutions} aux={aux_loss_weight}"
    )
    cfg.GPU_NUM = 1
    cfg.RUNNER = ForecastStateProgressiveRunner

    cfg.ENV = EasyDict()
    cfg.ENV.SEED = seed

    cfg.DATASET = EasyDict()
    cfg.DATASET.NAME = data_name
    cfg.DATASET.TYPE = TimeSeriesForecastingDataset
    cfg.DATASET.PARAM = EasyDict(
        {
            "dataset_name": data_name,
            "train_val_test_ratio": train_val_test_ratio,
            "input_len": input_len,
            "output_len": output_len,
        }
    )

    cfg.SCALER = EasyDict()
    cfg.SCALER.TYPE = ZScoreScaler
    cfg.SCALER.PARAM = EasyDict(
        {
            "dataset_name": data_name,
            "train_ratio": train_val_test_ratio[0],
            "norm_each_channel": norm_each_channel,
            "rescale": rescale,
        }
    )

    cfg.MODEL = EasyDict()
    cfg.MODEL.NAME = ForecastStateProgressive.__name__
    cfg.MODEL.ARCH = ForecastStateProgressive
    cfg.MODEL.PARAM = model_param
    cfg.MODEL.FORWARD_FEATURES = [0, 1, 2]
    cfg.MODEL.TARGET_FEATURES = [0]

    cfg.METRICS = EasyDict()
    cfg.METRICS.FUNCS = EasyDict(
        {
            "MAE": masked_mae,
            "RMSE": masked_rmse,
            "MAPE": masked_mape,
        }
    )
    cfg.METRICS.TARGET = "MAE"
    cfg.METRICS.NULL_VAL = null_val

    cfg.TRAIN = EasyDict()
    cfg.TRAIN.NUM_EPOCHS = num_epochs
    cfg.TRAIN.CKPT_SAVE_DIR = os.path.join(
        "checkpoints",
        "ForecastState",
        experiment_name,
        "_".join([data_name, str(num_epochs), str(input_len), str(output_len)]),
    )
    cfg.TRAIN.LOSS = masked_mae
    cfg.TRAIN.OPTIM = EasyDict()
    cfg.TRAIN.OPTIM.TYPE = "Adam"
    cfg.TRAIN.OPTIM.PARAM = {"lr": 0.005}

    # Match HyperD / ForecastSpace PEMS04 scheduler configuration for fairness.
    desc = load_dataset_desc(data_name)
    train_steps = math.ceil(desc["num_time_steps"] * train_val_test_ratio[0])
    cfg.TRAIN.LR_SCHEDULER = EasyDict()
    cfg.TRAIN.LR_SCHEDULER.TYPE = "OneCycleLR"
    cfg.TRAIN.LR_SCHEDULER.PARAM = {
        "pct_start": 0.3,
        "epochs": num_epochs,
        "steps_per_epoch": train_steps,
        "max_lr": cfg.TRAIN.OPTIM.PARAM["lr"],
    }
    cfg.TRAIN.CLIP_GRAD_PARAM = {"max_norm": 5.0}
    cfg.TRAIN.DATA = EasyDict()
    cfg.TRAIN.DATA.BATCH_SIZE = batch_size
    cfg.TRAIN.DATA.SHUFFLE = True

    cfg.VAL = EasyDict()
    cfg.VAL.INTERVAL = 1
    cfg.VAL.DATA = EasyDict()
    cfg.VAL.DATA.BATCH_SIZE = batch_size

    cfg.TEST = EasyDict()
    cfg.TEST.INTERVAL = 1
    cfg.TEST.DATA = EasyDict()
    cfg.TEST.DATA.BATCH_SIZE = batch_size

    cfg.EVAL = EasyDict()
    cfg.EVAL.HORIZONS = [3, 6, 12]
    cfg.EVAL.USE_GPU = True
    return cfg
