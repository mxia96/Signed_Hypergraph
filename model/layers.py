from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Parameter


class HypergraphConv(nn.Module):
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.2,
                 symmetric_norm: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.symmetric_norm = symmetric_norm
        self.lin = nn.Linear(in_features, out_features)
        torch.nn.init.xavier_normal_(self.lin.weight)
        self.bias = Parameter(torch.zeros(out_features))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor, H: Tensor) -> Tensor:
        x = self.dropout(self.lin(x))

        dv = torch.sum(torch.abs(H), dim=2)
        de = torch.sum(torch.abs(H), dim=1)
        Ht = torch.transpose(H, 1, 2)

        de_inv = torch.diag_embed(1.0 / de)
        de_inv[de_inv == float("inf")] = 0

        if self.symmetric_norm:
            dv_isqrt = torch.diag_embed(torch.pow(dv, -0.5))
            dv_isqrt[dv_isqrt == float("inf")] = 0
            x = dv_isqrt @ H @ de_inv @ Ht @ dv_isqrt @ x
        else:
            dv_inv = torch.diag_embed(1.0 / dv)
            dv_inv[dv_inv == float("inf")] = 0
            x = dv_inv @ H @ de_inv @ Ht @ x

        return x + self.bias


def _topk(x: Tensor, ratio: float) -> Tensor:
    num_nodes = x.shape[1]
    _, indices = torch.sort(x, dim=-1, descending=True)
    k = int(math.ceil(ratio * num_nodes))
    return indices[:, :k]


class GroupwiseTopKPooling(nn.Module):
    def __init__(self, in_channels: int, num_regions: int, ratio: float = 0.8,
                 momentum: float = 0.9, nonlinearity=torch.tanh):
        super().__init__()
        self.in_channels = in_channels
        self.num_regions = num_regions
        self.ratio = ratio
        self.momentum = momentum
        self.nonlinearity = nonlinearity
        self.register_buffer("running_score", torch.ones(1, num_regions))
        self.weight = Parameter(torch.Tensor(1, in_channels))
        nn.init.uniform_(self.weight)

    def forward(self, x: Tensor):
        score = (x * self.weight).sum(dim=-1)
        score = self.nonlinearity(score / self.weight.norm(p=2, dim=-1))

        if self.training:
            mean_score = score.mean(dim=0, keepdim=True)
            self.running_score.mul_(self.momentum)
            self.running_score.add_((1 - self.momentum) * mean_score)

        perm = _topk(self.running_score, self.ratio).squeeze(0)
        x = x * score.unsqueeze(-1)
        return x[:, perm, :], perm, score
