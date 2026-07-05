from __future__ import annotations

import argparse

import numpy as np

from .config import ConstructionConfig
from .construct import construct_hypergraph


def evaluate_svm(H, labels, n_splits=10, seed=42):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

    features = H.reshape(len(H), -1)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs, aucs, sens, spes = [], [], [], []
    for tr, te in skf.split(features, labels):
        clf = Pipeline([("clf", SVC(gamma="auto", kernel="linear", C=1.0))])
        clf.fit(features[tr], labels[tr])
        pred = clf.predict(features[te])
        score = clf.decision_function(features[te])
        accs.append(accuracy_score(labels[te], pred))
        aucs.append(roc_auc_score(labels[te], score))
        tn, fp, fn, tp = confusion_matrix(labels[te], pred, labels=[0, 1]).ravel()
        sens.append(tp / (tp + fn) if (tp + fn) else 0.0)
        spes.append(tn / (tn + fp) if (tn + fp) else 0.0)
    print("\n=== Linear SVM (%d-fold) ===" % n_splits)
    print(f"ACC {np.mean(accs)*100:.1f} +/- {np.std(accs)*100:.1f}")
    print(f"AUC {np.mean(aucs)*100:.1f} +/- {np.std(aucs)*100:.1f}")
    print(f"SEN {np.mean(sens)*100:.1f} +/- {np.std(sens)*100:.1f}")
    print(f"SPE {np.mean(spes)*100:.1f} +/- {np.std(spes)*100:.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--eval-svm", action="store_true")
    args = ap.parse_args()

    cfg = ConstructionConfig.from_yaml(args.config)
    np.random.seed(cfg.seed)

    signal = np.load(cfg.signal_path, allow_pickle=True).astype(np.float64)
    print(f"Loaded signal {signal.shape} from {cfg.signal_path}")

    H = construct_hypergraph(signal, cfg)
    np.save(cfg.output_path, H)
    print(f"Saved hypergraph incidence {H.shape} -> {cfg.output_path}")

    if args.eval_svm:
        labels = np.load(cfg.label_path)
        evaluate_svm(H, labels)


if __name__ == "__main__":
    main()
