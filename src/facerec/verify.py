"""Face verification: are two photos the same person? Decided by comparing the distance between
their embeddings with a threshold, and scored with LFW's 10-fold protocol."""

import numpy as np
from sklearn.metrics import roc_curve


def accuracy_at(distances: np.ndarray, same: np.ndarray, threshold: float) -> float:
    return float(np.mean((distances < threshold) == same))


def best_threshold(distances: np.ndarray, same: np.ndarray) -> float:
    """The threshold with the highest accuracy on these pairs (a midpoint between two distances)."""
    d = np.sort(np.unique(distances))
    candidates = np.concatenate([[d[0] - 1e-6], (d[:-1] + d[1:]) / 2, [d[-1] + 1e-6]])
    # Accuracy at every candidate at once: pairs below the threshold are called "same".
    calls = distances[None, :] < candidates[:, None]
    accuracy = (calls == same[None, :]).mean(axis=1)
    return float(candidates[np.argmax(accuracy)])


def ten_fold(distances: np.ndarray, same: np.ndarray, folds: np.ndarray) -> dict:
    """LFW's protocol: for each fold, choose the threshold on the other nine and test on it."""
    thresholds, accuracies = [], []
    for k in np.unique(folds):
        test = folds == k
        t = best_threshold(distances[~test], same[~test])
        thresholds.append(t)
        accuracies.append(accuracy_at(distances[test], same[test], t))
    return {
        "accuracy": float(np.mean(accuracies)),
        "accuracy_std": float(np.std(accuracies)),
        "threshold": float(np.mean(thresholds)),
        "fold_accuracy": accuracies,
    }


def tar_at_far(distances: np.ndarray, same: np.ndarray, far: float) -> tuple[float, float]:
    """The share of same-person pairs accepted (true accept rate) when only `far` of
    different-person pairs may be accepted, and the threshold that achieves it."""
    fpr, tpr, thresholds = roc_curve(same, -distances)
    i = np.searchsorted(fpr, far, side="right") - 1
    return float(tpr[i]), float(-thresholds[i])
