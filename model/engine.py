from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix


def build_optimizer(model, cfg):
    if cfg.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg.lr,
                                weight_decay=cfg.weight_decay)
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=cfg.lr,
                               momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    raise ValueError(f"Unknown optimizer: {cfg.optimizer}")


def train_one_epoch(model, device, loader, optimizer, cfg):
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    total_loss, n_batches = 0.0, 0
    for labels, feats, H in loader:
        labels = labels.float().to(device)
        feats = feats.float().to(device)
        H = H.float().to(device)
        labels = torch.where(labels < 0.1, torch.zeros_like(labels), labels)

        optimizer.zero_grad()
        logits = model(feats, H)
        loss = criterion(logits, labels)
        loss.backward()
        if cfg.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, device, loader):
    model.eval()
    all_logits, all_labels = [], []
    for labels, feats, H in loader:
        labels = labels.float().to(device)
        feats = feats.float().to(device)
        H = H.float().to(device)
        labels = torch.where(labels < 0.1, torch.zeros_like(labels), labels)
        logits = model(feats, H)
        all_logits.extend(logits.cpu().flatten().tolist())
        all_labels.extend(labels.cpu().flatten().tolist())

    y = np.array(all_labels)
    logits = np.array(all_logits)
    pred = (1.0 / (1.0 + np.exp(-logits)) >= 0.5).astype(int)
    acc = accuracy_score(y, pred)
    auc = roc_auc_score(y, logits) if len(np.unique(y)) > 1 else float("nan")
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sen = tp / (tp + fn) if (tp + fn) else 0.0
    spe = tn / (tn + fp) if (tn + fp) else 0.0
    return {"acc": acc, "auc": auc, "sen": sen, "spe": spe}


def fit(model, device, train_loader, test_loader, cfg):
    optimizer = build_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg.scheduler_step, gamma=cfg.scheduler_gamma)

    best = {"acc": 0.0}
    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(model, device, train_loader, optimizer, cfg)
        metrics = evaluate(model, device, test_loader)
        scheduler.step()
        if epoch % cfg.log_every == 0 or epoch == cfg.epochs:
            print(f"epoch {epoch:3d} | loss {train_loss:.4f} | "
                  f"acc {metrics['acc']:.3f} auc {metrics['auc']:.3f} "
                  f"sen {metrics['sen']:.3f} spe {metrics['spe']:.3f}")
        if metrics["acc"] >= best["acc"]:
            best = {**metrics, "epoch": epoch}
    return best
