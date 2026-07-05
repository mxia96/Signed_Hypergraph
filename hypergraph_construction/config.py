from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Optional, Union

import yaml


@dataclass
class ConstructionConfig:
    signal_path: str = "data/x.npy"
    label_path: str = "data/y.npy"
    output_path: str = "data/H.npy"

    num_hyperedges: int = 20

    alpha: float = 0.1
    beta: float = 0.1
    lam: float = 0.02

    num_iters: int = 200
    learning_rate: float = 0.01
    step_scale_by_loss: bool = False
    convergence_tol: Optional[float] = 1e-3
    row_normalize: bool = True

    laplacian_mode: str = "full"
    laplacian_k_cols: int = 5
    laplacian_eps: float = 1e-2
    laplacian_normalizer: Union[float, str] = 116.0
    laplacian_use_sqrt: bool = True

    reg3_mode: str = "cosine"

    init_method: str = "kmeans"
    init_random_state: int = 0

    seed: int = 3
    verbose: bool = True

    @classmethod
    def from_yaml(cls, path: str) -> "ConstructionConfig":
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
