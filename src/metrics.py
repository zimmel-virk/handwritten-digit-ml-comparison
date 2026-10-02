"""NumPy-only evaluation metrics used across the project."""

import numpy as np


def confusion_matrix_numpy(y_true, y_pred, n_classes=10):
    y_true = np.asarray(y_true).astype(int).ravel()
    y_pred = np.asarray(y_pred).astype(int).ravel()
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def accuracy_numpy(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def classification_metrics_numpy(y_true, y_pred, n_classes=10):
    cm = confusion_matrix_numpy(y_true, y_pred, n_classes)
    precisions, recalls, f1s, supports = [], [], [], []

    for c in range(n_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        support = cm[c, :].sum()

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        supports.append(support)

    precisions = np.asarray(precisions)
    recalls = np.asarray(recalls)
    f1s = np.asarray(f1s)
    supports = np.asarray(supports)

    return {
        "accuracy": accuracy_numpy(y_true, y_pred),
        "macro_precision": float(precisions.mean()),
        "macro_recall": float(recalls.mean()),
        "macro_f1": float(f1s.mean()),
        "weighted_f1": float(np.sum(f1s * supports) / np.sum(supports)),
        "confusion_matrix": cm,
    }
