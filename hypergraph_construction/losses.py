from __future__ import annotations

import numpy as np


def vertex_degree(H_abs):
    return np.sum(H_abs, axis=1)


def hyperedge_degree(H_abs):
    return np.sum(H_abs, axis=0)


def fidelity_term(H, P):
    N = H.shape[0]
    val = 0.0
    grad = np.zeros_like(H)
    for n in range(N):
        R = H[n] @ H[n].T - P[n]
        fn = np.linalg.norm(R)
        val += fn
        if fn > 0:
            grad[n] = 2.0 * R @ H[n] / fn
    return val / N, grad / N


def l21_term(H):
    s = np.sqrt(np.sum(H ** 2, axis=0))
    val = float(np.sum(s))
    s_safe = np.where(s > 0, s, 1.0)
    grad = H / s_safe[None, :, :]
    return val, grad


def hyperedge_similarity_term(H, exclude_diagonal=False):
    N = H.shape[0]
    val = 0.0
    grad = np.zeros_like(H)
    for n in range(N):
        W = H[n].T
        r = np.linalg.norm(W, axis=1, keepdims=True)
        r_safe = np.where(r > 0, r, 1.0)
        Nn = W / r_safe
        Sim = Nn @ Nn.T
        S = Sim.copy()
        if exclude_diagonal:
            np.fill_diagonal(S, 0.0)
        fro = np.linalg.norm(S)
        val += fro
        if fro > 0:
            U = S / fro
            M = U @ Nn
            wm = np.sum(U * Sim, axis=1, keepdims=True)
            gW = (2.0 / r_safe) * (M - wm * Nn)
            grad[n] = gW.T
    return val / N, grad / N


def laplacian_term(H, P, mode="first_k", k_cols=5, eps=1e-2,
                   normalizer=116.0, use_sqrt=True):
    N = H.shape[0]
    val = 0.0
    grad = np.zeros_like(H)
    for n in range(N):
        Hn = H[n]
        absH = np.abs(Hn)
        signH = np.sign(Hn)
        dv = vertex_degree(absH)
        de = hyperedge_degree(absH)
        a = dv ** -0.5
        Dh1 = 1.0 / de

        Pk = P[n] if mode == "full" else P[n][:, 0:k_cols]
        B = Pk @ Pk.T
        C = (a[:, None] * B) * a[None, :]

        HD = Hn * Dh1[None, :]
        K = HD @ Hn.T
        g = float(np.sum(C * K))
        t = float(np.trace(Pk.T @ Pk)) - g

        # gradient of g = tr(M B) wrt H: explicit-H + vertex-degree + hyperedge-degree channels
        CH = C @ Hn
        grad1 = 2.0 * CH * Dh1[None, :]
        Q = np.sum(Hn * CH, axis=0)
        grad3 = -(Q / de ** 2)[None, :] * signH
        Vmat = B @ (a[:, None] * Hn)
        phi = np.sum(HD * Vmat, axis=1)
        grad2 = -(dv ** -1.5 * phi)[:, None] * signH
        dt = -(grad1 + grad2 + grad3)

        if mode == "full" or not use_sqrt:
            val += t
            grad[n] = dt
        else:
            sroot = np.sqrt((t + eps) / normalizer)
            val += sroot
            grad[n] = dt / (2.0 * normalizer * sroot)
    return val / N, grad / N


def total_loss_and_grad(H, P, cfg):
    fid_v, fid_g = fidelity_term(H, P)
    lap_v, lap_g = laplacian_term(
        H, P,
        mode=cfg.laplacian_mode,
        k_cols=cfg.laplacian_k_cols,
        eps=cfg.laplacian_eps,
        normalizer=(H.shape[1] if cfg.laplacian_normalizer == "num_regions"
                    else float(cfg.laplacian_normalizer)),
        use_sqrt=cfg.laplacian_use_sqrt,
    )
    l21_v, l21_g = l21_term(H)
    he_v, he_g = hyperedge_similarity_term(
        H, exclude_diagonal=(cfg.reg3_mode == "cosine_offdiag"))

    loss = fid_v + cfg.alpha * lap_v + cfg.lam * l21_v + cfg.beta * he_v
    grad = fid_g + cfg.alpha * lap_g + cfg.lam * l21_g + cfg.beta * he_g
    parts = {"fidelity": fid_v, "reg_laplacian": lap_v,
             "reg_l21": l21_v, "reg_hyperedge": he_v}
    return loss, grad, parts
