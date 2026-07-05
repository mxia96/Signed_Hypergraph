import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hypergraph_construction.config import ConstructionConfig
from hypergraph_construction.losses import total_loss_and_grad


def _make_data(seed=0, N=3, V=6, E=4):
    rng = np.random.default_rng(seed)
    H = rng.standard_normal((N, V, E))
    H = H / np.linalg.norm(H, axis=2, keepdims=True)
    Praw = rng.standard_normal((N, V, V))
    P = np.clip((Praw + Praw.transpose(0, 2, 1)) / 2, -1, 1)
    for n in range(N):
        np.fill_diagonal(P[n], 1.0)
    return H, P


def _finite_diff(cfg, H, P, eps=1e-6):
    g = np.zeros_like(H)
    it = np.nditer(H, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        Hp = H.copy(); Hp[idx] += eps
        Hm = H.copy(); Hm[idx] -= eps
        lp, _, _ = total_loss_and_grad(Hp, P, cfg)
        lm, _, _ = total_loss_and_grad(Hm, P, cfg)
        g[idx] = (lp - lm) / (2 * eps)
        it.iternext()
    return g


def _check(laplacian_mode, reg3_mode="cosine"):
    cfg = ConstructionConfig(alpha=0.7, beta=0.2, lam=0.13,
                             laplacian_mode=laplacian_mode,
                             laplacian_normalizer=116.0,
                             reg3_mode=reg3_mode)
    H, P = _make_data()
    _, g_analytic, _ = total_loss_and_grad(H, P, cfg)
    g_fd = _finite_diff(cfg, H, P)
    rel = np.abs(g_analytic - g_fd).max() / (np.abs(g_fd).max() + 1e-12)
    print(f"laplacian_mode={laplacian_mode:8s} reg3_mode={reg3_mode:15s} "
          f"max rel err = {rel:.2e}")
    assert rel < 1e-5, f"gradient mismatch ({laplacian_mode}, {reg3_mode}): {rel}"


def test_gradients_first_k():
    _check("first_k")


def test_gradients_full():
    _check("full")


def test_gradients_reg3_offdiag():
    _check("first_k", reg3_mode="cosine_offdiag")


if __name__ == "__main__":
    test_gradients_first_k()
    test_gradients_full()
    test_gradients_reg3_offdiag()
    print("All gradient checks passed.")
