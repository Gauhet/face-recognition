"""Grouping an unlabelled photo album by person, and scoring the groups against the true names."""

import numpy as np
from sklearn.cluster import AgglomerativeClustering


def cluster(embeddings: np.ndarray, threshold: float, linkage: str = "average") -> np.ndarray:
    """A group number for each face. Groups merge while their faces are, on average, closer than
    the threshold, so the number of people doesn't need to be known."""
    if len(embeddings) < 2:
        return np.zeros(len(embeddings), dtype=int)
    model = AgglomerativeClustering(n_clusters=None, distance_threshold=threshold, linkage=linkage)
    return model.fit_predict(embeddings)


def pairwise_scores(true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """Over all pairs of photos: precision is the share of pairs put in the same group that are the
    same person, recall the share of same-person pairs put in the same group."""
    same_true = true[:, None] == true[None, :]
    same_pred = pred[:, None] == pred[None, :]
    upper = np.triu(np.ones_like(same_true), k=1)
    tp = (same_true & same_pred & upper).sum()
    predicted, actual = (same_pred & upper).sum(), (same_true & upper).sum()
    precision = tp / predicted if predicted else 1.0
    recall = tp / actual if actual else 1.0
    return {"precision": float(precision), "recall": float(recall), "f1": f1(precision, recall)}


def bcubed(true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """B-cubed scores, averaged over photos: for each photo, how much of its group is the same
    person (precision) and how much of that person ended up in its group (recall). Unlike pair
    counts, one large group doesn't dominate."""
    same_true = true[:, None] == true[None, :]
    same_pred = pred[:, None] == pred[None, :]
    both = (same_true & same_pred).sum(axis=1)
    precision = float(np.mean(both / same_pred.sum(axis=1)))
    recall = float(np.mean(both / same_true.sum(axis=1)))
    return {"precision": precision, "recall": recall, "f1": f1(precision, recall)}


def f1(precision: float, recall: float) -> float:
    return float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0


def make_album(
    people_photos: dict[str, list[int]], people: list[str], max_photos: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """A pretend photo album: up to max_photos photos of each person. Returns photo indices and a
    person number for each."""
    indices, labels = [], []
    for k, name in enumerate(people):
        photos = rng.permutation(people_photos[name])[:max_photos]
        indices += photos.tolist()
        labels += [k] * len(photos)
    return np.array(indices), np.array(labels)
