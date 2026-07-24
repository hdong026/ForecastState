"""HyperD STFE frequency encoder without the original final fc head.

Copied from baselines/HyperD/arch/STFE.py with only the prediction head removed.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HyperDSTFEEncoder(nn.Module):
    """
    HyperD STFE before the original final fc head.

    The frequency-domain computation must remain equivalent
    to baselines/HyperD/arch/STFE.py.
    """

    def __init__(
        self,
        num_nodes: int,
        seq_len: int,
        embed_size: int,
        hidden_size: int,
    ) -> None:
        super().__init__()

        self.scale = 0.02
        self.feature_size = num_nodes
        self.seq_length = seq_len
        self.embed_size = embed_size
        self.sparsity_threshold = 0.01

        self.embeddings = nn.Parameter(
            torch.randn(1, embed_size)
        )

        self.spatial_r1 = nn.Parameter(
            self.scale * torch.randn(embed_size, hidden_size)
        )
        self.spatial_i1 = nn.Parameter(
            self.scale * torch.randn(embed_size, hidden_size)
        )
        self.spatial_rb1 = nn.Parameter(
            self.scale * torch.randn(hidden_size)
        )
        self.spatial_ib1 = nn.Parameter(
            self.scale * torch.randn(hidden_size)
        )
        self.spatial_r2 = nn.Parameter(
            self.scale * torch.randn(hidden_size, embed_size)
        )
        self.spatial_i2 = nn.Parameter(
            self.scale * torch.randn(hidden_size, embed_size)
        )
        self.spatial_rb2 = nn.Parameter(
            self.scale * torch.randn(embed_size)
        )
        self.spatial_ib2 = nn.Parameter(
            self.scale * torch.randn(embed_size)
        )

        self.temporal_r1 = nn.Parameter(
            self.scale * torch.randn(embed_size, hidden_size)
        )
        self.temporal_i1 = nn.Parameter(
            self.scale * torch.randn(embed_size, hidden_size)
        )
        self.temporal_rb1 = nn.Parameter(
            self.scale * torch.randn(hidden_size)
        )
        self.temporal_ib1 = nn.Parameter(
            self.scale * torch.randn(hidden_size)
        )
        self.temporal_r2 = nn.Parameter(
            self.scale * torch.randn(hidden_size, embed_size)
        )
        self.temporal_i2 = nn.Parameter(
            self.scale * torch.randn(hidden_size, embed_size)
        )
        self.temporal_rb2 = nn.Parameter(
            self.scale * torch.randn(embed_size)
        )
        self.temporal_ib2 = nn.Parameter(
            self.scale * torch.randn(embed_size)
        )

    @property
    def output_dim(self) -> int:
        return self.seq_length * self.embed_size

    def tokenEmb(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(3)
        return x * self.embeddings

    def C_MLP_s(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.fft.rfft(
            x,
            dim=2,
            norm="ortho",
        )
        y = self.C_MLP(
            x,
            self.spatial_r1,
            self.spatial_i1,
            self.spatial_r2,
            self.spatial_i2,
            self.spatial_rb1,
            self.spatial_rb2,
            self.spatial_ib1,
            self.spatial_ib2,
        )
        return torch.fft.irfft(
            y,
            n=self.feature_size,
            dim=2,
            norm="ortho",
        )

    def C_MLP_t(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = torch.fft.rfft(
            x,
            dim=2,
            norm="ortho",
        )
        y = self.C_MLP(
            x,
            self.temporal_r1,
            self.temporal_i1,
            self.temporal_r2,
            self.temporal_i2,
            self.temporal_rb1,
            self.temporal_rb2,
            self.temporal_ib1,
            self.temporal_ib2,
        )
        x = torch.fft.irfft(
            y,
            n=self.seq_length,
            dim=2,
            norm="ortho",
        )
        return x.transpose(1, 2)

    def C_MLP(
        self,
        x,
        r1,
        i1,
        r2,
        i2,
        rb1,
        rb2,
        ib1,
        ib2,
    ):
        o1_real = F.relu(
            torch.einsum("bijd,df->bijf", x.real, r1)
            - torch.einsum("bijd,df->bijf", x.imag, i1)
            + rb1
        )
        o1_imag = F.relu(
            torch.einsum("bijd,df->bijf", x.imag, r1)
            + torch.einsum("bijd,df->bijf", x.real, i1)
            + ib1
        )
        o2_real = F.relu(
            torch.einsum("bijf,fd->bijd", o1_real, r2)
            - torch.einsum("bijf,fd->bijd", o1_imag, i2)
            + rb2
        )
        o2_imag = F.relu(
            torch.einsum("bijf,fd->bijd", o1_imag, r2)
            + torch.einsum("bijf,fd->bijd", o1_real, i2)
            + ib2
        )
        y = torch.stack(
            [o2_real, o2_imag],
            dim=-1,
        )
        y = F.softshrink(
            y,
            lambd=self.sparsity_threshold,
        )
        return torch.view_as_complex(y)

    def forward(
        self,
        residual_history: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            residual_history: [B, P, N]

        Returns:
            encoded: [B, N, P * embed_size]
        """
        if residual_history.ndim != 3:
            raise ValueError(
                "residual_history must have shape [B, P, N], "
                f"got {tuple(residual_history.shape)}."
            )

        batch_size, seq_len, num_nodes = residual_history.shape

        if seq_len != self.seq_length:
            raise ValueError(
                f"Expected seq_len={self.seq_length}, got {seq_len}."
            )

        if num_nodes != self.feature_size:
            raise ValueError(
                f"Expected num_nodes={self.feature_size}, got {num_nodes}."
            )

        x = self.tokenEmb(residual_history)
        bias = x

        x = self.C_MLP_s(x)
        x = self.C_MLP_t(x)
        x = x + bias

        return x.transpose(1, 2).reshape(
            batch_size,
            num_nodes,
            -1,
        )
