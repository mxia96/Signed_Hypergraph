from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import List, Optional

import yaml


@dataclass
class ModelConfig:
    fmri_path: str = "data/x.npy"
    hypergraph_path: str = "data/H.npy"
    label_path: str = "data/y.npy"
    test_ratio: float = 0.2
    split_seed: int = 42

    model_type: str = "pool"
    num_roi: int = 68
    input_features: int = 68
    output_features: int = 32
    input_activation: str = "tanh"   # none | relu | tanh | gelu | leaky_relu | elu | sigmoid
    depth: int = 1
    hidden_channels_mlp: List[int] = field(default_factory=lambda: [64, 32])
    conv_dropout: float = 0.2
    mlp_dropout: float = 0.0
    symmetric_norm: bool = False

    pool_ratio: float = 0.8
    pool_momentum: float = 0.9

    use_pca: bool = False
    pca_components: int = 10

    optimizer: str = "sgd"
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 0.001
    scheduler_step: int = 10
    scheduler_gamma: float = 0.1
    epochs: int = 60
    batch_size: int = 60
    grad_clip: Optional[float] = None

    device: str = "cuda"
    seed: int = 42
    log_every: int = 10

    @classmethod
    def from_yaml(cls, path: str) -> "ModelConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        unknown = set(raw) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**raw)

    def to_yaml(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump({f.name: getattr(self, f.name) for f in fields(self)},
                           f, sort_keys=False)
