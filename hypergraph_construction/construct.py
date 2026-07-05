from __future__ import annotations

import numpy as np

from .initialization import build_initial_H
from .losses import total_loss_and_grad


def _row_normalize(H):
    norm = np.linalg.norm(H, axis=2, keepdims=True)
    norm = np.where(norm > 0, norm, 1.0)
    return H / norm


def construct_hypergraph(signal, cfg):
    P = np.asarray(signal, dtype=np.float64)
    N = P.shape[0]

    H_single = build_initial_H(P, cfg)
    H = np.stack([H_single] * N, axis=0)
    if cfg.row_normalize:
        H = _row_normalize(H)

    prev_loss = None
    for it in range(cfg.num_iters):
        loss, grad, parts = total_loss_and_grad(H, P, cfg)

        step = cfg.learning_rate * (loss if cfg.step_scale_by_loss else 1.0)
        H = H - step * grad

        if cfg.row_normalize:
            H = _row_normalize(H)

        if cfg.verbose and (it % 20 == 0 or it == cfg.num_iters - 1):
            print(f"[iter {it:4d}] loss={loss:.6f}  "
                  f"fid={parts['fidelity']:.4f} lap={parts['reg_laplacian']:.4f} "
                  f"l21={parts['reg_l21']:.4f} he={parts['reg_hyperedge']:.4f}")

        if cfg.convergence_tol is not None and prev_loss is not None:
            if abs(prev_loss - loss) <= cfg.convergence_tol:
                if cfg.verbose:
                    print(f"Converged at iter {it}.")
                break
        prev_loss = loss

    return H
