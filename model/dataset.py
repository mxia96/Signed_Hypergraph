from __future__ import annotations

import numpy as np
from torch.utils.data import Dataset


class fMRIDataset(Dataset):
    def __init__(self, features, hypergraph, labels):
        self.features = np.asarray(features, dtype=np.float32)
        self.hypergraph = np.asarray(hypergraph, dtype=np.float32)
        self.labels = np.asarray(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = np.array([self.labels[idx]], dtype=np.float32)
        return label, self.features[idx], self.hypergraph[idx]


def apply_pca(features, n_components, fit_idx=None):
    from sklearn.decomposition import PCA

    N, V, D = features.shape
    flat = features.reshape(N * V, D)
    fit_mask = None
    if fit_idx is not None:
        node_mask = np.zeros(N, dtype=bool)
        node_mask[fit_idx] = True
        fit_mask = np.repeat(node_mask, V)
    pca = PCA(n_components=n_components)
    pca.fit(flat[fit_mask] if fit_mask is not None else flat)
    reduced = pca.transform(flat).reshape(N, V, n_components)
    return reduced.astype(np.float32)
