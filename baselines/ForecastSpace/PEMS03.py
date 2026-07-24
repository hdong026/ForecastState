import os
import sys
import math
from easydict import EasyDict

sys.path.append(os.path.abspath(__file__ + '/../../..'))

from basicts.metrics import masked_mae, masked_mape, masked_rmse
from basicts.data import TimeSeriesForecastingDataset
from basicts.scaler import ZScoreScaler
from basicts.utils import get_regular_settings, load_dataset_desc

from .arch import ForecastSpace
from .runner import ForecastSpaceRunner

DATA_NAME = 'PEMS03'
regular_settings = get_regular_settings(DATA_NAME)
INPUT_LEN = regular_settings['INPUT_LEN']
OUTPUT_LEN = regular_settings['OUTPUT_LEN']
TRAIN_VAL_TEST_RATIO = regular_settings['TRAIN_VAL_TEST_RATIO']
NORM_EACH_CHANNEL = regular_settings['NORM_EACH_CHANNEL']
RESCALE = regular_settings['RESCALE']
NULL_VAL = regular_settings['NULL_VAL']

MODEL_ARCH = ForecastSpace
MODEL_PARAM = {
    "node_size": 358,
    "input_len": INPUT_LEN,
    "output_len": OUTPUT_LEN,
    "input_dim": 3,
    "patch_len": 3,
    "stride": 4,
    "td_size": 288,
    "dw_size": 7,
    "d_td": 32,
    "d_dw": 32,
    "d_d": 32,
    "d_spa": 32,
    "if_time_in_day": True,
    "if_day_in_week": True,
    "if_spatial": True,
    "num_layer": 2,
    "spatial_scheme": "C",
    "adj_mx_path": os.path.join("datasets", DATA_NAME, "adj_mx.pkl"),
    "use_gcn": True,
    "gcn_hidden_dim": 64,
    "use_dynamic_spatial": True,
    "dyn_hidden_dim": 64,
    "dyn_topk": 20,
    "dyn_tau": 0.5,
    "dyn_static_weight": 0.2,
    "use_adaptive_adj": True,
    "adp_hidden_dim": 32,
    "adp_topk": 20,
    "adp_tau": 0.5,
    "use_hybrid_graph": True,
    "hybrid_alpha": 0.2,
    "use_patch_branch": True,
    "use_downsample_branch": True,
    "use_linear_residual_branch": True,
    "patch_embedding_mode": "serial_concat",
    "patch_data_input_mode": "all",
    "post_spatial_mode": "adaptive_only",
    "spatial_placement": "final",
    "use_pre_temporal_spatial_enhancement": False,
    "use_prev_condition": True,
    "chain_lengths": [3, 6, 12],
    "chain_loss_weights": [0.2, 0.3, 1.0],
    "spatial_stage_loss_weights": [0.0, 0.0, 1.0],
    "spatial_graph_loss_weights": [0.0, 0.0, 0.0],
}
NUM_EPOCHS = 100

CFG = EasyDict()
CFG.DESCRIPTION = 'ForecastSpace G1_final_adaptive on PEMS03 12->12'
CFG.GPU_NUM = 1
CFG.RUNNER = ForecastSpaceRunner
CFG.ENV = EasyDict()
CFG.ENV.SEED = 1
CFG.DATASET = EasyDict()
CFG.DATASET.NAME = DATA_NAME
CFG.DATASET.TYPE = TimeSeriesForecastingDataset
CFG.DATASET.PARAM = EasyDict({
    'dataset_name': DATA_NAME,
    'train_val_test_ratio': TRAIN_VAL_TEST_RATIO,
    'input_len': INPUT_LEN,
    'output_len': OUTPUT_LEN,
})
CFG.SCALER = EasyDict()
CFG.SCALER.TYPE = ZScoreScaler
CFG.SCALER.PARAM = EasyDict({
    'dataset_name': DATA_NAME,
    'train_ratio': TRAIN_VAL_TEST_RATIO[0],
    'norm_each_channel': NORM_EACH_CHANNEL,
    'rescale': RESCALE,
})
CFG.MODEL = EasyDict()
CFG.MODEL.NAME = MODEL_ARCH.__name__
CFG.MODEL.ARCH = MODEL_ARCH
CFG.MODEL.PARAM = MODEL_PARAM
CFG.MODEL.FORWARD_FEATURES = [0, 1, 2]
CFG.MODEL.TARGET_FEATURES = [0]
CFG.METRICS = EasyDict()
CFG.METRICS.FUNCS = EasyDict({'MAE': masked_mae, 'RMSE': masked_rmse, 'MAPE': masked_mape})
CFG.METRICS.TARGET = 'MAE'
CFG.METRICS.NULL_VAL = NULL_VAL
CFG.TRAIN = EasyDict()
CFG.TRAIN.NUM_EPOCHS = NUM_EPOCHS
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join('results', 'ForecastSpace', DATA_NAME)
CFG.TRAIN.LOSS = masked_mae
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {"lr": 0.005}
CFG.TRAIN.LR_SCHEDULER = EasyDict()
desc = load_dataset_desc(DATA_NAME)
train_steps = math.ceil(desc["num_time_steps"] * TRAIN_VAL_TEST_RATIO[0])
CFG.TRAIN.LR_SCHEDULER.TYPE = "OneCycleLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {
    "pct_start": 0.3,
    "epochs": NUM_EPOCHS,
    "steps_per_epoch": train_steps,
    "max_lr": CFG.TRAIN.OPTIM.PARAM["lr"],
}
CFG.TRAIN.CLIP_GRAD_PARAM = {'max_norm': 5.0}
CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.BATCH_SIZE = 64
CFG.TRAIN.DATA.SHUFFLE = True
CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
CFG.VAL.DATA = EasyDict()
CFG.VAL.DATA.BATCH_SIZE = 64
CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 1
CFG.TEST.DATA = EasyDict()
CFG.TEST.DATA.BATCH_SIZE = 64
CFG.EVAL = EasyDict()
CFG.EVAL.HORIZONS = [3, 6, 12]
CFG.EVAL.USE_GPU = True
