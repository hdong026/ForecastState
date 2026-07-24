"""ForecastStateGraphResolution model — paper Algorithm 1."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .diagnostics import assert_same_resolution, stage_shape_report, summarize_alphas
from .forecast_refinement import SharedHistoryEncoder, StageResidualPredictor
from .graph_operator import StageGraphOperator
from .graph_resolution import (
    build_structural_graph,
    load_or_build_graph_resolution,
)
from .kasa_backbone import KASABackboneBundle
from .resolution_space import ResolutionSpace, validate_resolution_schedule
from .temporal_refinement import InitialForecast


class ForecastStateGraphResolution(nn.Module):
    """
    history X
      -> forecast state at (h1, K1)
      -> ...
      -> full-resolution forecast state (H, 1)
      -> final prediction
    """

    def __init__(self, **model_args):
        super().__init__()
        self.num_nodes = int(model_args.get("num_nodes", model_args.get("node_size")))
        self.input_len = int(model_args["input_len"])
        self.pred_len = int(model_args.get("pred_len", model_args.get("output_len")))
        self.input_dim = int(model_args.get("input_dim", 3))
        self.output_dim = int(model_args.get("output_dim", 1))
        self.hidden_dim = int(model_args.get("hidden_dim", 64))
        self.num_encoder_layers = int(model_args.get("num_encoder_layers", 2))
        self.dropout = float(model_args.get("dropout", 0.1))
        self.temporal_lift_mode = str(model_args.get("temporal_lift_mode", "linear"))
        self.initial_forecast = str(model_args.get("initial_forecast", "kasa"))
        self.alpha_mode = str(model_args.get("alpha_mode", "softplus"))
        self.use_graph = bool(model_args.get("use_graph", True))
        self.use_kasa_stages = bool(model_args.get("use_kasa_stages", True))
        self.adp_topk = model_args.get("adp_topk", 8)
        self.adp_tau = float(model_args.get("adp_tau", 0.5))
        self.adp_embed_dim = int(model_args.get("adp_embed_dim", 32))
        self.lambda_init = float(model_args.get("lambda_init", 0.9))
        self.learnable_lambda = bool(model_args.get("learnable_lambda", True))
        self.dataset_name = str(model_args.get("dataset_name", "PEMS04"))
        self.adj_mx_path = model_args.get(
            "adj_mx_path", f"datasets/{self.dataset_name}/adj_mx.pkl"
        )
        self.distance_path = model_args.get("distance_path", None)
        self.distance_mode = str(model_args.get("distance_mode", "auto"))
        self.sigma_d = float(model_args.get("sigma_d", 0.5))
        self.lambda_d = float(model_args.get("lambda_d", 0.1))
        self.clustering_seed = int(model_args.get("clustering_seed", model_args.get("seed", 1)))
        self.cache_dir = model_args.get(
            "graph_cache_dir",
            f"datasets/{self.dataset_name}/forecast_state_graph_cache",
        )
        self.print_stage_shapes = bool(model_args.get("print_stage_shapes", True))
        self._printed_shapes = False

        schedule = model_args.get(
            "resolution_schedule",
            [
                {"h": 3, "capacity": 4},
                {"h": 6, "capacity": 2},
                {"h": 12, "capacity": 1},
            ],
        )
        self.resolution_schedule = validate_resolution_schedule(schedule, self.pred_len)
        self.num_stages = len(self.resolution_schedule)

        # KASA shared backbone (codebooks)
        self.kasa_bundle = KASABackboneBundle(
            num_nodes=self.num_nodes,
            input_len=self.input_len,
            patch_len=int(model_args.get("patch_len", 3)),
            stride=int(model_args.get("stride", 4)),
            td_size=int(model_args.get("td_size", 288)),
            dw_size=int(model_args.get("dw_size", 7)),
            d_td=int(model_args.get("d_td", 32)),
            d_dw=int(model_args.get("d_dw", 32)),
            d_d=int(model_args.get("d_d", 32)),
            d_spa=int(model_args.get("d_spa", 32)),
            num_layer=int(model_args.get("num_layer", 2)),
        )

        self.history_encoder = SharedHistoryEncoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_encoder_layers,
            dropout=self.dropout,
        )
        self.init_forecast_module = None
        self.init_kasa = None
        if self.initial_forecast == "kasa":
            self.init_kasa = self.kasa_bundle.make_step(self.pred_len)
        else:
            self.init_forecast_module = InitialForecast(
                mode=self.initial_forecast,
                input_len=self.input_len,
                pred_len=self.pred_len,
                num_nodes=self.num_nodes,
                output_dim=self.output_dim,
            )

        unique_caps = sorted({s["capacity"] for s in self.resolution_schedule}, reverse=True)
        self._graph_meta_by_cap: dict[int, dict] = {}
        for cap in unique_caps:
            meta = load_or_build_graph_resolution(
                dataset=self.dataset_name,
                adj_mx_path=self.adj_mx_path,
                capacity=cap,
                seed=self.clustering_seed,
                sigma_d=self.sigma_d,
                lambda_d=self.lambda_d,
                distance_path=self.distance_path,
                distance_mode=self.distance_mode,
                cache_dir=self.cache_dir,
            )
            self._graph_meta_by_cap[cap] = meta

        self.spaces = nn.ModuleList()
        self.graph_ops = nn.ModuleList()
        self.residual_predictors = nn.ModuleList()
        self.kasa_stages = nn.ModuleList()
        self.region_spa_codebooks = nn.ParameterList()
        self.residual_scales = nn.ParameterList()
        self.raw_alphas = nn.ParameterList()

        for stage_idx, stage in enumerate(self.resolution_schedule):
            h = stage["h"]
            cap = stage["capacity"]
            meta = self._graph_meta_by_cap[cap]
            c = torch.from_numpy(meta["C"])
            p = torch.from_numpy(meta["P"])
            space = ResolutionSpace(
                h=h,
                capacity=cap,
                full_h=self.pred_len,
                num_nodes=self.num_nodes,
                c_assign=c,
                p_proj=p,
                temporal_lift_mode=self.temporal_lift_mode,
            )
            self.spaces.append(space)

            w = meta["W"]
            a_str = build_structural_graph(meta["P"], w, meta["C"])
            topk = self.adp_topk
            if isinstance(topk, (list, tuple)):
                topk = int(topk[min(stage_idx, len(topk) - 1)])
            else:
                topk = int(topk)
            topk = min(max(1, topk), space.num_regions)
            gop = StageGraphOperator(
                num_regions=space.num_regions,
                a_str=torch.from_numpy(a_str),
                embed_dim=self.adp_embed_dim,
                topk=topk,
                tau=self.adp_tau,
                lambda_init=self.lambda_init,
                learnable_lambda=self.learnable_lambda,
            )
            self.graph_ops.append(gop)

            # Always keep a light residual head as fallback / complement
            pred = StageResidualPredictor(
                h=h,
                num_regions=space.num_regions,
                history_len=self.input_len,
                history_dim=self.hidden_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.output_dim,
                dropout=self.dropout,
                use_graph=self.use_graph,
                zero_init_head=True,
            )
            self.residual_predictors.append(pred)

            if self.use_kasa_stages:
                if space.num_regions == self.num_nodes:
                    self.region_spa_codebooks.append(
                        nn.Parameter(torch.zeros(1), requires_grad=False)
                    )
                    kasa_step = self.kasa_bundle.make_step(h)
                else:
                    spa = nn.Parameter(torch.empty(space.num_regions, self.kasa_bundle.d_spa))
                    nn.init.xavier_uniform_(spa)
                    self.region_spa_codebooks.append(spa)
                    kasa_step = self.kasa_bundle.make_step(h, spa_codebook=spa)
                self.kasa_stages.append(kasa_step)
                # zero residual scale so init ≈ KASA/linear baseline
                self.residual_scales.append(nn.Parameter(torch.zeros(1)))
            else:
                self.kasa_stages.append(nn.Identity())
                self.region_spa_codebooks.append(nn.Parameter(torch.zeros(1), requires_grad=False))
                self.residual_scales.append(nn.Parameter(torch.ones(1), requires_grad=False))
            if self.alpha_mode == "fixed":
                self.raw_alphas.append(nn.Parameter(torch.tensor(1.0), requires_grad=False))
            else:
                self.raw_alphas.append(nn.Parameter(torch.tensor(0.541324854)))

        self.last_diagnostics: dict[str, Any] = {}

    def _alpha(self, stage_idx: int) -> torch.Tensor:
        raw = self.raw_alphas[stage_idx]
        if self.alpha_mode == "softplus":
            return F.softplus(raw)
        return raw

    def _project_history_to_regions(self, hist: torch.Tensor, space: ResolutionSpace) -> torch.Tensor:
        """hist [B,P,N,C] -> [B,P,M,C] via graph projection."""
        return torch.einsum("mn,bpnc->bpmc", space.P, hist)

    def _init_forecast(self, history: torch.Tensor) -> torch.Tensor:
        if self.init_kasa is not None:
            return self.init_kasa(history, prev_forecast=None, spatial_codebook=self.kasa_bundle.spa_codebook)
        return self.init_forecast_module(history)

    def _stage_residual(
        self,
        s: int,
        history: torch.Tensor,
        hist_enc: torch.Tensor,
        z_bar: torch.Tensor,
        space: ResolutionSpace,
    ) -> torch.Tensor:
        graph_op = self.graph_ops[s] if self.use_graph else None
        hist_region_enc = torch.einsum("mn,bpnc->bpmc", space.P, hist_enc)
        light = self.residual_predictors[s](hist_region_enc, z_bar, graph_op)

        if not self.use_kasa_stages:
            return light

        hist_region = self._project_history_to_regions(history, space)
        # Ensure 3 channels for KASA (flow + time features projected)
        if hist_region.shape[-1] < 3:
            pad = torch.zeros(
                *hist_region.shape[:-1],
                3 - hist_region.shape[-1],
                device=hist_region.device,
                dtype=hist_region.dtype,
            )
            hist_region = torch.cat([hist_region, pad], dim=-1)
        else:
            hist_region = hist_region[..., :3]

        spa = (
            self.kasa_bundle.spa_codebook
            if space.num_regions == self.num_nodes
            else self.region_spa_codebooks[s]
        )
        kasa_out = self.kasa_stages[s](
            hist_region,
            prev_forecast=z_bar,
            spatial_codebook=spa,
        )
        # optional graph mix on KASA residual candidate
        if self.use_graph and graph_op is not None:
            kasa_out = kasa_out + 0.1 * graph_op.propagate(kasa_out)
        scale = self.residual_scales[s]
        return light + scale * kasa_out

    def forward(
        self,
        history_data: torch.Tensor,
        future_data: torch.Tensor = None,
        batch_seen: int = 0,
        epoch: int = 0,
        train: bool = False,
        return_all: bool = False,
        **kwargs,
    ):
        del future_data, batch_seen, epoch, kwargs
        history = history_data[..., : self.input_dim]
        hist_enc = self.history_encoder(history)

        y_hat = self._init_forecast(history)

        stage_full_preds = []
        stage_res_preds = []
        stage_projected_preds = []
        stage_infos = []

        for s, stage in enumerate(self.resolution_schedule):
            space: ResolutionSpace = self.spaces[s]
            z_bar = space.project(y_hat)
            r_s = self._stage_residual(s, history, hist_enc, z_bar, space)
            delta = space.lift(r_s)
            alpha = self._alpha(s)
            y_hat = y_hat + alpha * delta

            stage_full_preds.append(y_hat)
            stage_res_preds.append(r_s)
            stage_projected_preds.append(space.project(y_hat))
            info = {
                **space.diagnostics(),
                "alpha": float(alpha.detach().item()),
                "lambda": float(self.graph_ops[s].lambda_s.detach().item()) if self.use_graph else None,
                "topk": int(self.graph_ops[s].topk) if self.use_graph else None,
                "residual_scale": float(self.residual_scales[s].detach().item()),
            }
            stage_infos.append(info)

        self.last_diagnostics = {
            "stages": stage_infos,
            "alphas": summarize_alphas([self._alpha(i) for i in range(self.num_stages)]),
            "resolution_schedule": list(self.resolution_schedule),
        }

        out = {
            "pred": y_hat,
            "stage_full_preds": stage_full_preds,
            "stage_residuals": stage_res_preds,
            "stage_projected_preds": stage_projected_preds,
            "resolution_schedule": list(self.resolution_schedule),
            "diagnostics": self.last_diagnostics,
        }
        if return_all:
            return out
        return y_hat

    def project_target(self, future_target: torch.Tensor, stage_idx: int) -> torch.Tensor:
        y = future_target[..., : self.output_dim]
        return self.spaces[stage_idx].project(y)

    def scale_matched_pairs(self, out: dict, future_target: torch.Tensor):
        y = future_target[..., : self.output_dim]
        pairs = []
        for s in range(self.num_stages):
            pred_s = self.spaces[s].project(out["stage_full_preds"][s])
            tgt_s = self.spaces[s].project(y)
            assert_same_resolution(pred_s, tgt_s, s)
            mae = None
            if self.print_stage_shapes and not self._printed_shapes:
                with torch.no_grad():
                    mae = float((pred_s - tgt_s).abs().mean().item())
            report = stage_shape_report(
                s,
                self.resolution_schedule[s]["h"],
                self.resolution_schedule[s]["capacity"],
                pred_s,
                tgt_s,
                mae=mae,
            )
            pairs.append((pred_s, tgt_s, report))
        if self.print_stage_shapes and not self._printed_shapes:
            for _, _, rep in pairs:
                print(f"[FSGR stage shapes] {rep}")
            self._printed_shapes = True
        return pairs
