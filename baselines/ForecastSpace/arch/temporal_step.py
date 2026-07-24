from __future__ import annotations

from math import ceil
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class MultiLayerPerceptron(nn.Module):
    """Multi-Layer Perceptron with residual links."""

    def __init__(self, input_dim, hidden_dim) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.fc2 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.15)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        hidden = self.fc2(self.drop(self.act(self.fc1(input_data))))
        hidden = self.norm((hidden + input_data).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return hidden


class PatchEncoder(nn.Module):
    def __init__(
        self,
        td_size,
        dw_size,
        td_codebook,
        dw_codebook,
        spa_codebook,
        if_time_in_day,
        if_day_in_week,
        if_spatial,
        input_dim,
        patch_len,
        stride,
        d_d,
        d_td,
        d_dw,
        d_spa,
        output_len,
        num_layer,
        patch_data_input_mode="all",
        patch_embedding_mode="serial_concat",
        patch_feature_dim=None,
    ):
        super().__init__()
        self.td_codebook = td_codebook
        self.dw_codebook = dw_codebook
        self.spa_codebook = spa_codebook
        self.if_time_in_day = if_time_in_day
        self.if_day_in_week = if_day_in_week
        self.if_spatial = if_spatial
        self.output_len = output_len
        self.td_size = td_size
        self.dw_size = dw_size
        self.stride = stride
        self.patch_len = patch_len
        self.encoder_input_dim = input_dim
        self.patch_data_input_mode = str(patch_data_input_mode).lower()
        self.patch_embedding_mode = str(patch_embedding_mode).lower()

        if self.patch_data_input_mode == "flow_only":
            data_input_dim = 1
        else:
            data_input_dim = input_dim
        self.data_input_dim = data_input_dim

        self.data_embedding_layer = nn.Conv2d(
            in_channels=data_input_dim * patch_len,
            out_channels=d_d,
            kernel_size=(1, 1),
            bias=True,
        )

        self.hidden_dim = d_d + d_dw * int(self.if_day_in_week) * 2 + d_td * int(self.if_time_in_day) * 2
        self.temporal_encoder = nn.Sequential(
            *[
                MultiLayerPerceptron(
                    self.hidden_dim + d_spa * int(self.if_spatial),
                    self.hidden_dim + d_spa * int(self.if_spatial),
                )
                for _ in range(num_layer)
            ]
        )
        self.spatial_encoder = nn.Sequential(
            *[
                MultiLayerPerceptron(
                    d_d + d_spa * int(self.if_spatial),
                    d_d + d_spa * int(self.if_spatial),
                )
                for _ in range(num_layer)
            ]
        )
        self.data_encoder = nn.Sequential(*[MultiLayerPerceptron(d_d, d_d) for _ in range(num_layer)])
        self.projection1 = nn.Conv2d(
            in_channels=(self.hidden_dim + d_spa * int(self.if_spatial)) * self.stride + d_td + d_dw,
            out_channels=output_len,
            kernel_size=(1, 1),
            bias=True,
        )

    def _embed_serial_concat(self, patch_input: torch.Tensor) -> torch.Tensor:
        if self.patch_data_input_mode == "flow_only":
            data_channel_indices = [0]
        else:
            data_channel_indices = list(range(self.encoder_input_dim))
        data_channels = [patch_input[..., i] for i in data_channel_indices]
        data_emb_input = torch.concat(data_channels, dim=2)
        data_emb = self.data_embedding_layer(data_emb_input.permute(0, 2, 1, 3)).permute(0, 2, 3, 1)
        return data_emb

    def forward(self, patch_input, spatial_codebook=None):
        batch_size, num, _, _, _ = patch_input.shape

        if self.if_day_in_week:
            day_in_week_data = patch_input[..., 2]
            day_start_idx = day_in_week_data[:, :, 0, :].long().clamp(0, self.dw_size - 1)
            day_end_idx = day_in_week_data[:, :, -1, :].long().clamp(0, self.dw_size - 1)
            day_in_week_start_emb = self.dw_codebook[day_start_idx]
            day_in_week_end_emb = self.dw_codebook[day_end_idx]
            future_day_in_week_emb = day_in_week_end_emb[:, -1, :, :].permute(0, 2, 1).unsqueeze(-1)
        else:
            day_in_week_start_emb = day_in_week_end_emb = future_day_in_week_emb = None

        if self.if_time_in_day:
            time_in_day_data = patch_input[..., 1]
            time_start_idx = torch.clamp((time_in_day_data[:, :, 0, :] * self.td_size).long(), 0, self.td_size - 1)
            time_end_idx = torch.clamp((time_in_day_data[:, :, -1, :] * self.td_size).long(), 0, self.td_size - 1)
            time_in_day_start_emb = self.td_codebook[time_start_idx]
            time_in_day_end_emb = self.td_codebook[time_end_idx]
            future_time_idx = (
                (time_in_day_data[:, -1, -1, :] * self.td_size + self.output_len) % self.td_size
            ).long()
            future_time_idx = torch.clamp(future_time_idx, 0, self.td_size - 1)
            future_time_in_day_emb = self.td_codebook[future_time_idx].permute(0, 2, 1).unsqueeze(-1)
        else:
            time_in_day_start_emb = time_in_day_end_emb = future_time_in_day_emb = None

        if self.if_spatial:
            if spatial_codebook is None:
                spatial_codebook = self.spa_codebook
            spatial_emb = spatial_codebook.unsqueeze(0).expand(batch_size, -1, -1).unsqueeze(1).expand(-1, num, -1, -1)
        else:
            spatial_emb = None

        data_emb = self._embed_serial_concat(patch_input)
        data_emb = self.data_encoder(data_emb.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        if self.if_spatial:
            hidden_input = torch.concat((data_emb, spatial_emb), dim=-1)
        else:
            hidden_input = data_emb
        hidden = hidden_input.permute(0, 3, 1, 2)
        hidden = self.spatial_encoder(hidden).permute(0, 2, 3, 1)
        hidden = torch.concat(
            (time_in_day_start_emb, day_in_week_start_emb, hidden, time_in_day_end_emb, day_in_week_end_emb),
            dim=-1,
        ).permute(0, 3, 1, 2)
        hidden = self.temporal_encoder(hidden)
        batch, channels, patches, nodes = hidden.shape
        hidden = hidden.permute(0, 2, 3, 1).reshape(batch, patches * channels, nodes).unsqueeze(-1)
        hidden = torch.concat((hidden, future_time_in_day_emb, future_day_in_week_emb), dim=1)
        return self.projection1(hidden)


class DownsampEncoder(nn.Module):
    def __init__(
        self,
        td_size,
        dw_size,
        td_codebook,
        dw_codebook,
        spa_codebook,
        if_time_in_day,
        if_day_in_week,
        if_spatial,
        input_dim,
        patch_len,
        stride,
        d_d,
        d_td,
        d_dw,
        d_spa,
        output_len,
        num_layer,
    ):
        super().__init__()
        self.td_codebook = td_codebook
        self.dw_codebook = dw_codebook
        self.spa_codebook = spa_codebook
        self.if_time_in_day = if_time_in_day
        self.if_day_in_week = if_day_in_week
        self.if_spatial = if_spatial
        self.output_len = output_len
        self.td_size = td_size
        self.dw_size = dw_size
        self.stride = stride
        self.encoder_input_dim = input_dim

        self.data_embedding_layer = nn.Conv2d(
            in_channels=input_dim * patch_len,
            out_channels=d_d,
            kernel_size=(1, 1),
            bias=True,
        )
        self.hidden_dim = d_d + d_dw * int(self.if_day_in_week) * 2 + d_td * int(self.if_time_in_day) * 2
        self.temporal_encoder = nn.Sequential(
            *[
                MultiLayerPerceptron(
                    self.hidden_dim + d_spa * int(self.if_spatial),
                    self.hidden_dim + d_spa * int(self.if_spatial),
                )
                for _ in range(num_layer)
            ]
        )
        self.spatial_encoder = nn.Sequential(
            *[
                MultiLayerPerceptron(
                    d_d + d_spa * int(self.if_spatial),
                    d_d + d_spa * int(self.if_spatial),
                )
                for _ in range(num_layer)
            ]
        )
        self.data_encoder = nn.Sequential(*[MultiLayerPerceptron(d_d, d_d) for _ in range(num_layer)])
        self.projection1 = nn.Conv2d(
            in_channels=(self.hidden_dim + d_spa * int(self.if_spatial)) * self.stride + d_td + d_dw,
            out_channels=output_len,
            kernel_size=(1, 1),
            bias=True,
        )

    def forward(self, patch_input, spatial_codebook=None):
        batch_size, num, _, _, _ = patch_input.shape

        if self.if_time_in_day:
            time_in_day_data = patch_input[..., 1]
            time_start_idx = torch.clamp((time_in_day_data[:, :, 0, :] * self.td_size).long(), 0, self.td_size - 1)
            time_end_idx = torch.clamp((time_in_day_data[:, :, -1, :] * self.td_size).long(), 0, self.td_size - 1)
            time_in_day_start_emb = self.td_codebook[time_start_idx]
            time_in_day_end_emb = self.td_codebook[time_end_idx]
            future_time_idx = ((time_in_day_data[:, -1, -1, :] * self.td_size + self.output_len) % self.td_size).long()
            future_time_idx = torch.clamp(future_time_idx, 0, self.td_size - 1)
            future_time_in_day_emb = self.td_codebook[future_time_idx].permute(0, 2, 1).unsqueeze(-1)
        else:
            time_in_day_start_emb = time_in_day_end_emb = future_time_in_day_emb = None

        if self.if_day_in_week:
            day_in_week_data = patch_input[..., 2]
            day_start_idx = day_in_week_data[:, :, 0, :].long().clamp(0, self.dw_size - 1)
            day_end_idx = day_in_week_data[:, :, -1, :].long().clamp(0, self.dw_size - 1)
            day_in_week_start_emb = self.dw_codebook[day_start_idx]
            day_in_week_end_emb = self.dw_codebook[day_end_idx]
            future_day_in_week_emb = day_in_week_end_emb[:, -1, :, :].permute(0, 2, 1).unsqueeze(-1)
        else:
            day_in_week_start_emb = day_in_week_end_emb = future_day_in_week_emb = None

        if self.if_spatial:
            if spatial_codebook is None:
                spatial_codebook = self.spa_codebook
            spatial_emb = spatial_codebook.unsqueeze(0).expand(batch_size, -1, -1).unsqueeze(1).expand(-1, num, -1, -1)
        else:
            spatial_emb = None

        data_channels = [patch_input[..., i] for i in range(self.encoder_input_dim)]
        data_emb = self.data_embedding_layer(torch.concat(data_channels, dim=2).permute(0, 2, 1, 3)).permute(0, 2, 3, 1)
        data_emb = self.data_encoder(data_emb.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        if self.if_spatial:
            hidden_input = torch.concat((data_emb, spatial_emb), dim=-1)
        else:
            hidden_input = data_emb
        hidden = hidden_input.permute(0, 3, 1, 2)
        hidden = self.spatial_encoder(hidden).permute(0, 2, 3, 1)
        hidden = torch.concat(
            (time_in_day_start_emb, day_in_week_start_emb, hidden, time_in_day_end_emb, day_in_week_end_emb),
            dim=-1,
        ).permute(0, 3, 1, 2)
        hidden = self.temporal_encoder(hidden)
        batch, channels, patches, nodes = hidden.shape
        hidden = hidden.permute(0, 2, 3, 1).reshape(batch, patches * channels, nodes).unsqueeze(-1)
        hidden = torch.concat((hidden, future_time_in_day_emb, future_day_in_week_emb), dim=1)
        return self.projection1(hidden)


def interpolate_forecast(forecast: torch.Tensor, target_len: int) -> torch.Tensor:
    batch_size, forecast_len, num_nodes, channels = forecast.shape
    x = forecast.permute(0, 2, 3, 1).reshape(batch_size * num_nodes, channels, forecast_len)
    x = F.interpolate(x, size=target_len, mode="linear", align_corners=False)
    return x.reshape(batch_size, num_nodes, channels, target_len).permute(0, 3, 1, 2)


class KASATemporalStep(nn.Module):
    """One KASA temporal forecasting step without post spatial refinement."""

    def __init__(
        self,
        output_len: int,
        input_len: int,
        patch_len: int,
        stride: int,
        td_size: int,
        dw_size: int,
        td_codebook,
        dw_codebook,
        spa_codebook,
        if_time_in_day: bool,
        if_day_in_week: bool,
        if_spatial: bool,
        d_d: int,
        d_td: int,
        d_dw: int,
        d_spa: int,
        num_layer: int,
        use_patch_branch: bool = True,
        use_downsample_branch: bool = True,
        use_linear_residual_branch: bool = True,
        patch_data_input_mode: str = "all",
        patch_embedding_mode: str = "serial_concat",
        patch_feature_dim=None,
        use_prev_condition: bool = True,
        latent_cond_dim: int = 0,
    ):
        super().__init__()
        self.output_len = output_len
        self.input_len = input_len
        self.patch_len = patch_len
        self.stride = stride
        self.use_patch_branch = use_patch_branch
        self.use_downsample_branch = use_downsample_branch
        self.use_linear_residual_branch = use_linear_residual_branch
        self.use_prev_condition = use_prev_condition
        self.latent_cond_dim = int(latent_cond_dim)
        self.base_encoder_input_dim = 3
        self.cond_encoder_input_dim = 3 + (1 if self.latent_cond_dim <= 0 else self.latent_cond_dim)

        encoder_kwargs = dict(
            td_size=td_size,
            dw_size=dw_size,
            td_codebook=td_codebook,
            dw_codebook=dw_codebook,
            spa_codebook=spa_codebook,
            if_time_in_day=if_time_in_day,
            if_day_in_week=if_day_in_week,
            if_spatial=if_spatial,
            patch_len=patch_len,
            stride=stride,
            d_d=d_d,
            d_td=d_td,
            d_dw=d_dw,
            d_spa=d_spa,
            output_len=output_len,
            num_layer=num_layer,
        )
        patch_kwargs = dict(
            encoder_kwargs,
            patch_data_input_mode=patch_data_input_mode,
            patch_embedding_mode=patch_embedding_mode,
            patch_feature_dim=patch_feature_dim,
        )

        self.patch_encoder = PatchEncoder(input_dim=self.base_encoder_input_dim, **patch_kwargs)
        self.downsamp_encoder = DownsampEncoder(input_dim=self.base_encoder_input_dim, **encoder_kwargs)
        self.patch_encoder_cond = PatchEncoder(input_dim=self.cond_encoder_input_dim, **patch_kwargs)
        self.downsamp_encoder_cond = DownsampEncoder(input_dim=self.cond_encoder_input_dim, **encoder_kwargs)
        self.residual = nn.Conv2d(
            in_channels=input_len,
            out_channels=output_len,
            kernel_size=(1, 1),
            bias=True,
        )

    def _build_step_input(
        self,
        history_data: torch.Tensor,
        prev_forecast: Optional[torch.Tensor],
    ) -> tuple:
        x_main = history_data[..., :3]
        if prev_forecast is not None and self.use_prev_condition:
            cond = interpolate_forecast(prev_forecast, self.input_len)
            return torch.cat([x_main, cond], dim=-1), True
        return x_main, False

    def forward(
        self,
        history_data: torch.Tensor,
        prev_forecast: Optional[torch.Tensor] = None,
        spatial_codebook=None,
    ) -> torch.Tensor:
        step_input, use_cond = self._build_step_input(history_data, prev_forecast)

        in_len_add = ceil(1.0 * self.input_len / self.stride) * self.stride - self.input_len
        if in_len_add:
            main_input_aug = torch.cat(
                (step_input[:, -1:, :, :].expand(-1, in_len_add, -1, -1), step_input),
                dim=1,
            )
        else:
            main_input_aug = step_input

        downsamp_input = [main_input_aug[:, i :: self.stride, :, :] for i in range(self.stride)]
        downsamp_input = torch.stack(downsamp_input, dim=1)
        patch_input = main_input_aug.unfold(dimension=1, size=self.patch_len, step=self.patch_len).permute(0, 1, 4, 2, 3)

        if use_cond:
            patch_encoder = self.patch_encoder_cond
            downsamp_encoder = self.downsamp_encoder_cond
        else:
            patch_encoder = self.patch_encoder
            downsamp_encoder = self.downsamp_encoder

        branch_outputs = []
        if self.use_patch_branch:
            branch_outputs.append(patch_encoder(patch_input, spatial_codebook=spatial_codebook))
        if self.use_downsample_branch:
            branch_outputs.append(downsamp_encoder(downsamp_input, spatial_codebook=spatial_codebook))
        if self.use_linear_residual_branch:
            branch_outputs.append(self.residual(history_data[..., 0:1].permute(0, 1, 2, 3)))
        if not branch_outputs:
            raise ValueError("At least one temporal branch must be enabled.")
        return sum(branch_outputs)
