"""Stage-wise mixed structural / adaptive graph operators.

Paper: Eq. (18)–(22).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def mask_topk(logits: torch.Tensor, topk: int) -> torch.Tensor:
    n = logits.shape[-1]
    k = int(topk)
    if k <= 0:
        raise ValueError("topk must be positive")
    if k >= n:
        return logits
    idx = torch.topk(logits, k=k, dim=-1).indices
    keep = torch.zeros_like(logits, dtype=torch.bool)
    keep.scatter_(-1, idx, True)
    return logits.masked_fill(~keep, float("-inf"))


class StageGraphOperator(nn.Module):
    """A_s = λ A_str + (1-λ) A_adp for one resolution stage."""

    def __init__(
        self,
        num_regions: int,
        a_str: torch.Tensor,
        embed_dim: int = 32,
        topk: int = 8,
        tau: float = 0.5,
        lambda_init: float = 0.9,
        learnable_lambda: bool = True,
    ):
        super().__init__()
        self.num_regions = int(num_regions)
        self.topk = min(int(topk), self.num_regions)
        self.tau = float(tau)
        self.register_buffer("A_str", a_str.float())
        self.emb_src = nn.Parameter(torch.empty(self.num_regions, embed_dim))
        self.emb_dst = nn.Parameter(torch.empty(self.num_regions, embed_dim))
        nn.init.xavier_uniform_(self.emb_src)
        nn.init.xavier_uniform_(self.emb_dst)
        # favor structural graph initially via large positive logit -> sigmoid ≈ lambda_init
        init_logit = math.log(lambda_init / max(1e-6, 1.0 - lambda_init))
        if learnable_lambda:
            self.lambda_logit = nn.Parameter(torch.tensor(float(init_logit)))
        else:
            self.register_buffer("lambda_logit", torch.tensor(float(init_logit)))

    @property
    def lambda_s(self) -> torch.Tensor:
        return torch.sigmoid(self.lambda_logit)

    def build_adaptive(self) -> torch.Tensor:
        src = F.normalize(self.emb_src, p=2, dim=-1)
        dst = F.normalize(self.emb_dst, p=2, dim=-1)
        logits = src @ dst.t()
        logits = mask_topk(logits, self.topk)
        return torch.softmax(logits / max(self.tau, 1e-6), dim=-1)

    def forward(self) -> torch.Tensor:
        a_adp = self.build_adaptive()
        lam = self.lambda_s
        return lam * self.A_str + (1.0 - lam) * a_adp

    def propagate(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B,T,M,C] -> same shape via A_s."""
        a = self.forward()
        return torch.einsum("ij,btjc->btic", a, x)
