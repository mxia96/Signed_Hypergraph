from __future__ import annotations

import argparse

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from .config import ModelConfig
from .dataset import fMRIDataset, apply_pca
from .engine import fit
from .network import build_model


def resolve_device(name):
    if name == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable -> falling back to CPU.")
        return torch.device("cpu")
    return torch.device(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = ModelConfig.from_yaml(args.config)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = resolve_device(cfg.device)

    features = np.load(cfg.fmri_path, allow_pickle=True)
    H = np.load(cfg.hypergraph_path)
    labels = np.load(cfg.label_path)
    print(f"features {features.shape} | H {H.shape} | labels {labels.shape}")

    idx = np.arange(len(labels))
    train_idx, test_idx = train_test_split(
        idx, test_size=cfg.test_ratio, random_state=cfg.split_seed, stratify=labels)

    if cfg.use_pca:
        features = apply_pca(features, cfg.pca_components, fit_idx=train_idx)
        if cfg.input_features != cfg.pca_components:
            print(f"[info] use_pca=True -> input_features {cfg.input_features} -> {cfg.pca_components}")
            cfg.input_features = cfg.pca_components

    train_ds = fMRIDataset(features[train_idx], H[train_idx], labels[train_idx])
    test_ds = fMRIDataset(features[test_idx], H[test_idx], labels[test_idx])
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size)

    model = build_model(cfg).to(device)
    best = fit(model, device, train_loader, test_loader, cfg)
    print("\n=== Best (by accuracy) ===")
    print(f"epoch {best['epoch']} | ACC {best['acc']*100:.1f} | AUC {best['auc']*100:.1f} "
          f"| SEN {best['sen']*100:.1f} | SPE {best['spe']*100:.1f}")


if __name__ == "__main__":
    main()
