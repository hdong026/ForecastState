import os
import sys
import math
from easydict import EasyDict

sys.path.append(os.path.abspath(__file__ + "/../../.."))

from basicts.metrics import masked_mae, masked_mape, masked_rmse
from basicts.data import TimeSeriesForecastingDataset
from basicts.scaler import ZScoreScaler
from basicts.utils import get_regular_settings, load_dataset_desc

from .arch import ForecastStateGraphResolution
from .runner import ForecastStateGraphResolutionRunner

############################## Hot Parameters ##############################
DATA_NAME = "PEMS04"
regular_settings = get_regular_settings(DATA_NAME)
INPUT_LEN = regular_settings["INPUT_LEN"]
OUTPUT_LEN = regular_settings["OUTPUT_LEN"]
TRAIN_VAL_TEST_RATIO = regular_settings["TRAIN_VAL_TEST_RATIO"]
NORM_EACH_CHANNEL = regular_settings["NORM_EACH_CHANNEL"]
RESCALE = regular_settings["RESCALE"]
NULL_VAL = regular_settings["NULL_VAL"]

# Configurable schedules (ablation)
JOINT_REFINEMENT = [
    {"h": 3, "capacity": 4},
    {"h": 6, "capacity": 2},
    {"h": 12, "capacity": 1},
]
TEMPORAL_ONLY = [
    {"h": 3, "capacity": 1},
    {"h": 6, "capacity": 1},
    {"h": 12, "capacity": 1},
]
GRAPH_ONLY = [
    {"h": 12, "capacity": 4},
    {"h": 12, "capacity": 2},
    {"h": 12, "capacity": 1},
]
INTERLEAVED_REFINEMENT = [
    {"h": 3, "capacity": 4},
    {"h": 6, "capacity": 4},
    {"h": 6, "capacity": 2},
    {"h": 12, "capacity": 2},
    {"h": 12, "capacity": 1},
]

# Active schedule / optimization (change for E2–E6)
RESOLUTION_SCHEDULE = JOINT_REFINEMENT
OPTIMIZATION_MODE = "final_only"  # final_only | weighted_sum | gradient_projection
INITIAL_FORECAST = "kasa"  # zero | last_value | linear_head
USE_GRAPH = True
USE_KASA_STAGES = True

# Scheduler fairness:
# - match_hyperd: same buggy steps_per_epoch as HyperD/ForecastSpace configs
# - correct_batches: steps_per_epoch = ceil(n_train_samples / batch_size)
SCHEDULER_MODE = "match_hyperd"

MODEL_ARCH = ForecastStateGraphResolution
MODEL_PARAM = {
    "num_nodes": 307,
    "input_len": INPUT_LEN,
    "pred_len": OUTPUT_LEN,
    "input_dim": 3,
    "output_dim": 1,
    "hidden_dim": 64,
    "num_encoder_layers": 2,
    "dropout": 0.1,
    "resolution_schedule": RESOLUTION_SCHEDULE,
    "initial_forecast": INITIAL_FORECAST,
    "use_kasa_stages": USE_KASA_STAGES,
    "patch_len": 3,
    "stride": 4,
    "td_size": 288,
    "dw_size": 7,
    "d_td": 32,
    "d_dw": 32,
    "d_d": 32,
    "d_spa": 32,
    "num_layer": 2,
    "optimization_mode": OPTIMIZATION_MODE,
    "aux_loss_weights": [0.2, 0.3],
    "grad_rho": 1.0,
    "use_graph": USE_GRAPH,
    "adp_topk": 8,
    "adp_tau": 0.5,
    "adp_embed_dim": 32,
    "lambda_init": 0.9,
    "learnable_lambda": True,
    "alpha_mode": "softplus",
    "temporal_lift_mode": "linear",
    "dataset_name": DATA_NAME,
    "adj_mx_path": os.path.join("datasets", DATA_NAME, "adj_mx.pkl"),
    "distance_mode": "auto",
    "sigma_d": 0.5,
    "lambda_d": 0.1,
    "clustering_seed": 1,
    "graph_cache_dir": os.path.join("datasets", DATA_NAME, "forecast_state_graph_cache"),
    "print_stage_shapes": True,
    "log_grad_stats": True,
}
NUM_EPOCHS = 2
BATCH_SIZE = 64

############################## General Configuration ##############################
CFG = EasyDict()
CFG.DESCRIPTION = "FSGR tiny smoke on PEMS04 (2 epochs, final_only)"
CFG.GPU_NUM = 1
CFG.RUNNER = ForecastStateGraphResolutionRunner

############################## Environment Configuration ##############################
CFG.ENV = EasyDict()
CFG.ENV.SEED = 1

############################## Dataset Configuration ##############################
CFG.DATASET = EasyDict()
CFG.DATASET.NAME = DATA_NAME
CFG.DATASET.TYPE = TimeSeriesForecastingDataset
CFG.DATASET.PARAM = EasyDict(
    {
        "dataset_name": DATA_NAME,
        "train_val_test_ratio": TRAIN_VAL_TEST_RATIO,
        "input_len": INPUT_LEN,
        "output_len": OUTPUT_LEN,
    }
)

############################## Scaler Configuration ##############################
CFG.SCALER = EasyDict()
CFG.SCALER.TYPE = ZScoreScaler
CFG.SCALER.PARAM = EasyDict(
    {
        "dataset_name": DATA_NAME,
        "train_ratio": TRAIN_VAL_TEST_RATIO[0],
        "norm_each_channel": NORM_EACH_CHANNEL,
        "rescale": RESCALE,
    }
)

############################## Model Configuration ##############################
CFG.MODEL = EasyDict()
CFG.MODEL.NAME = MODEL_ARCH.__name__
CFG.MODEL.ARCH = MODEL_ARCH
CFG.MODEL.PARAM = MODEL_PARAM
CFG.MODEL.FORWARD_FEATURES = [0, 1, 2]
CFG.MODEL.TARGET_FEATURES = [0]

############################## Metrics Configuration ##############################
CFG.METRICS = EasyDict()
CFG.METRICS.FUNCS = EasyDict(
    {
        "MAE": masked_mae,
        "RMSE": masked_rmse,
        "MAPE": masked_mape,
    }
)
CFG.METRICS.TARGET = "MAE"
CFG.METRICS.NULL_VAL = NULL_VAL

############################## Training Configuration ##############################
CFG.TRAIN = EasyDict()
CFG.TRAIN.NUM_EPOCHS = NUM_EPOCHS
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    "checkpoints",
    MODEL_ARCH.__name__,
    "_".join([DATA_NAME, "tiny", str(NUM_EPOCHS), str(INPUT_LEN), str(OUTPUT_LEN)]),
)
CFG.TRAIN.LOSS = masked_mae
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {
    "lr": 0.005,
}
CFG.TRAIN.LR_SCHEDULER = EasyDict()
desc = load_dataset_desc(DATA_NAME)
total_len = int(desc["num_time_steps"])
valid_len = int(total_len * TRAIN_VAL_TEST_RATIO[1])
test_len = int(total_len * TRAIN_VAL_TEST_RATIO[2])
train_len = total_len - valid_len - test_len
n_train_samples = train_len - INPUT_LEN - OUTPUT_LEN + 1
correct_steps = math.ceil(n_train_samples / BATCH_SIZE)
hyperd_steps = math.ceil(desc["num_time_steps"] * TRAIN_VAL_TEST_RATIO[0])
steps_per_epoch = hyperd_steps if SCHEDULER_MODE == "match_hyperd" else correct_steps

CFG.TRAIN.LR_SCHEDULER.TYPE = "OneCycleLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {
    "pct_start": 0.3,
    "epochs": NUM_EPOCHS,
    "steps_per_epoch": steps_per_epoch,
    "max_lr": CFG.TRAIN.OPTIM.PARAM["lr"],
}
CFG.TRAIN.CLIP_GRAD_PARAM = {
    "max_norm": 5.0,
}
CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.BATCH_SIZE = BATCH_SIZE
CFG.TRAIN.DATA.SHUFFLE = True

############################## Validation Configuration ##############################
CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
CFG.VAL.DATA = EasyDict()
CFG.VAL.DATA.BATCH_SIZE = BATCH_SIZE

############################## Test Configuration ##############################
CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 1
CFG.TEST.DATA = EasyDict()
CFG.TEST.DATA.BATCH_SIZE = BATCH_SIZE

############################## Evaluation Configuration ##############################
CFG.EVAL = EasyDict()
CFG.EVAL.HORIZONS = [3, 6, 12]
CFG.EVAL.USE_GPU = True
