"""Build BasicTS 1.1.0 config for KASA G1_final_adaptive reproduce on PEMS04 12->12."""

from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

import torch
from torch.optim.lr_scheduler import MultiStepLR

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basicts.configs import BasicTSForecastingConfig

from forecast_space.data import Pems04PklDataset
from forecast_space.models.kasa_g1.config import KasaG1ModelConfig
from forecast_space.scaler import PreprocessedFlowScaler
from forecast_space.taskflow import G1PklTaskFlow
from forecast_space.wrapper import KasaG1Wrapper, kasa_g1_chain_loss


def _load_flow_scaler_stats(dataset_dir: Path, input_len: int, output_len: int) -> dict:
    scaler_path = dataset_dir / f"scaler_in{input_len}_out{output_len}.pkl"
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    args = scaler["args"]
    return {
        "mean": torch.tensor(float(args["mean"])),
        "std": torch.tensor(float(args["std"])),
    }


def build_config(num_epochs: int = 100) -> BasicTSForecastingConfig:
    input_len = 12
    output_len = 12
    dataset_name = "PEMS04"
    data_dir = str(ROOT / "datasets" / dataset_name)
    scaler_stats = _load_flow_scaler_stats(Path(data_dir), input_len, output_len)
    flow_mean = float(scaler_stats["mean"])
    flow_std = float(scaler_stats["std"])

    model_config = KasaG1ModelConfig(
        num_nodes=307,
        input_len=input_len,
        output_len=output_len,
        input_dim=3,
        main_input_dim=3,
        patch_len=3,
        stride=4,
        adj_mx_path=os.path.join(data_dir, "adj_mx.pkl"),
        use_prev_condition=True,
        chain_lengths=[3, 6, 12],
        chain_loss_weights=[0.2, 0.3, 1.0],
        spatial_graph_loss_weights=[0.0, 0.0, 0.0],
        spatial_placement="final",
        post_spatial_mode="adaptive_only",
        use_extra_prior_input=False,
        variant_name="G1_final_adaptive",
        flow_mean=flow_mean,
        flow_std=flow_std,
        null_val=0.0,
    )

    return BasicTSForecastingConfig(
        model=KasaG1Wrapper,
        model_config=model_config,
        dataset_name=dataset_name,
        dataset_type=Pems04PklDataset,
        dataset_params={
            "dataset_name": dataset_name,
            "input_len": input_len,
            "output_len": output_len,
            "data_dir": data_dir,
        },
        taskflow=G1PklTaskFlow(),
        scaler=PreprocessedFlowScaler,
        norm_each_channel=False,
        rescale=True,
        stats=scaler_stats,
        null_val=0.0,
        null_to_num=0.0,
        gpus=None,
        seed=1,
        num_epochs=num_epochs,
        batch_size=32,
        loss=kasa_g1_chain_loss,
        optimizer_params={"lr": 0.002, "weight_decay": 0.0001},
        lr_scheduler=MultiStepLR,
        lr_scheduler_params={
            "milestones": [1, 35, 60, 80, 95],
            "gamma": 0.5,
        },
        ckpt_save_dir=str(
            ROOT / "checkpoints" / "G1_final_adaptive_PEMS04_12to12_kasa_reproduce"
        ),
        ckpt_save_strategy=list(range(1, num_epochs + 1)),
        target_metric="MAE",
        best_metric="min",
        eval_horizons=[3, 6, 12],
        eval_after_train=True,
        train_data_shuffle=True,
        train_data_num_workers=2,
        val_data_num_workers=2,
        test_data_num_workers=2,
    )


CFG = build_config()
