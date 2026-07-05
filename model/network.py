from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import HypergraphConv, GroupwiseTopKPooling


def _ceil(x):
    return int(math.ceil(x))


def make_activation(name):
    name = (name or "none").lower()
    table = {"none": nn.Identity, "identity": nn.Identity, "relu": nn.ReLU,
             "tanh": nn.Tanh, "gelu": nn.GELU, "leaky_relu": nn.LeakyReLU,
             "elu": nn.ELU, "sigmoid": nn.Sigmoid}
    if name not in table:
        raise ValueError(f"Unknown activation: {name}")
    return table[name]()


class MLP(nn.Sequential):
    # equivalent to torchvision.ops.MLP (same layer sequence), avoids the dependency
    def __init__(self, in_channels, hidden_channels, norm_layer=None,
                 activation_layer=nn.ReLU, dropout=0.0, bias=True):
        layers = []
        in_dim = in_channels
        for hidden_dim in hidden_channels[:-1]:
            layers.append(nn.Linear(in_dim, hidden_dim, bias=bias))
            if norm_layer is not None:
                layers.append(norm_layer(hidden_dim))
            layers.append(activation_layer())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, hidden_channels[-1], bias=bias))
        layers.append(nn.Dropout(dropout))
        super().__init__(*layers)


class fMRIHyperPool(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        out = cfg.output_features
        self.lin0 = nn.Linear(cfg.input_features, out)
        self.input_act = make_activation(cfg.input_activation)

        self.pools = nn.ModuleList()
        self.down_convs = nn.ModuleList()
        for _ in range(cfg.depth):
            self.pools.append(GroupwiseTopKPooling(
                out, num_regions=cfg.num_roi, ratio=cfg.pool_ratio,
                momentum=cfg.pool_momentum))
            self.down_convs.append(HypergraphConv(
                out, out, dropout=cfg.conv_dropout, symmetric_norm=cfg.symmetric_norm))

        kept = _ceil(cfg.pool_ratio * cfg.num_roi)
        self.decoder = MLP(in_channels=out * kept,
                           hidden_channels=cfg.hidden_channels_mlp,
                           norm_layer=nn.BatchNorm1d, activation_layer=nn.ReLU,
                           dropout=cfg.mlp_dropout)
        self.head = nn.Linear(cfg.hidden_channels_mlp[-1], 1)
        torch.nn.init.xavier_uniform_(self.head.weight)

    def forward(self, f, H):
        x = self.input_act(self.lin0(f))
        for i in range(self.cfg.depth):
            x, perm, _ = self.pools[i](x)
            H = H[:, perm]
            x = F.relu(self.down_convs[i](x, H))
        x = x.reshape(len(f), -1)
        x = self.decoder(x)
        return self.head(x)


class fMRIHyper(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        out = cfg.output_features
        self.lin0 = nn.Linear(cfg.input_features, out)
        self.input_act = make_activation(cfg.input_activation)
        self.down_convs = nn.ModuleList([
            HypergraphConv(out, out, dropout=cfg.conv_dropout,
                           symmetric_norm=cfg.symmetric_norm)
            for _ in range(cfg.depth)])
        self.decoder = MLP(in_channels=out * 2,
                           hidden_channels=cfg.hidden_channels_mlp,
                           norm_layer=nn.BatchNorm1d, activation_layer=nn.ReLU,
                           dropout=cfg.mlp_dropout)
        self.head = nn.Linear(cfg.hidden_channels_mlp[-1], 1)
        torch.nn.init.xavier_uniform_(self.head.weight)

    def forward(self, f, H):
        x = self.input_act(self.lin0(f))
        for conv in self.down_convs:
            x = F.relu(conv(x, H))
        x = torch.cat([torch.mean(x, dim=1), torch.max(x, dim=1)[0]], dim=1)
        x = self.decoder(x)
        return self.head(x)


def build_model(cfg):
    if cfg.model_type == "pool":
        return fMRIHyperPool(cfg)
    if cfg.model_type == "nopool":
        return fMRIHyper(cfg)
    raise ValueError(f"Unknown model_type: {cfg.model_type}")
