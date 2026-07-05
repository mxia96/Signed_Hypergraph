from __future__ import annotations

import numpy as np


def signed_kmedoids_init(signal, num_hyperedges, random_state=0):
    try:
        from sklearn_extra.cluster import KMedoids
    except ImportError as exc:
        raise ImportError(
            "init_method='kmedoids_signed' requires scikit-learn-extra "
            "(pip install scikit-learn-extra). Use init_method='kmeans' instead."
        ) from exc

    signal_concat = np.mean(signal, axis=0)
    correlation_matrix = np.corrcoef(signal_concat, rowvar=True)
    num_ROI = signal.shape[1]
    num_time_points = signal_concat.shape[1]

    kmedoids = KMedoids(n_clusters=num_hyperedges, metric="precomputed",
                        random_state=random_state)
    kmedoids.fit(1 - np.abs(signal_concat))

    mean_centroid = np.zeros([num_hyperedges, num_time_points])
    cluster_labels = kmedoids.labels_
    for i in range(num_hyperedges):
        idx = np.where(kmedoids.labels_ == i)[0]
        correlation_idx = correlation_matrix[np.ix_(idx, idx)]
        sum_cor = np.sum(np.abs(correlation_idx), axis=0)
        order = np.argsort(sum_cor)
        for j in range(num_ROI):
            if cluster_labels[j] == i:
                mean_centroid[i, :] += np.sign(correlation_matrix[order[-1], j]) * signal_concat[j, :]

    cor_center_signal = np.corrcoef(mean_centroid, signal_concat)[
        0:num_ROI, num_ROI:num_ROI + num_hyperedges]
    return cor_center_signal


def kmeans_init(signal, num_hyperedges, random_state=0):
    from sklearn.cluster import KMeans

    signal_concat = np.mean(signal, axis=0)
    num_ROI, T = signal_concat.shape
    km = KMeans(n_clusters=num_hyperedges, random_state=random_state, n_init=10)
    labels = km.fit_predict(signal_concat)

    centers = np.zeros([num_hyperedges, T])
    for j in range(num_hyperedges):
        members = np.where(labels == j)[0]
        if len(members) > 0:
            centers[j, :] = signal_concat[members, :].mean(axis=0)

    H_init = np.corrcoef(signal_concat, centers)[0:num_ROI, num_ROI:num_ROI + num_hyperedges]
    return np.nan_to_num(H_init)


def random_init(signal, num_hyperedges, random_state=0):
    rng = np.random.default_rng(random_state)
    return rng.standard_normal((signal.shape[1], num_hyperedges))


def build_initial_H(signal, cfg):
    if cfg.init_method == "kmedoids_signed":
        H_init = signed_kmedoids_init(signal, cfg.num_hyperedges, cfg.init_random_state)
    elif cfg.init_method == "kmeans":
        H_init = kmeans_init(signal, cfg.num_hyperedges, cfg.init_random_state)
    elif cfg.init_method == "random":
        H_init = random_init(signal, cfg.num_hyperedges, cfg.init_random_state)
    else:
        raise ValueError(f"Unknown init_method: {cfg.init_method}")
    return H_init.astype(np.float64)
