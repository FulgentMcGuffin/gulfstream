"""Shared HDBSCAN / OPTICS helpers for regime detection, dimred, and metrics."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.cluster import HDBSCAN, OPTICS
from sklearn.metrics import pairwise_distances

logger = logging.getLogger(__name__)

NOISE_LABEL = -1


def fit_hdbscan(
    X: np.ndarray,
    *,
    min_cluster_size: int = 5,
    min_samples: int | None = None,
    metric: str = "euclidean",
    cluster_selection_epsilon: float = 0.0,
    cluster_selection_method: str = "eom",
    allow_single_cluster: bool = False,
    **kwargs: Any,
) -> tuple[HDBSCAN, np.ndarray]:
    """Fit HDBSCAN; returns ``(model, labels)`` with noise labeled ``-1``."""
    X = np.asarray(X, dtype=float)
    model = HDBSCAN(
        min_cluster_size=int(min_cluster_size),
        min_samples=min_samples,
        metric=metric,
        cluster_selection_epsilon=float(cluster_selection_epsilon),
        cluster_selection_method=cluster_selection_method,
        allow_single_cluster=bool(allow_single_cluster),
        store_centers="centroid",
        copy=True,
        **{k: v for k, v in kwargs.items() if k not in ("store_centers", "copy")},
    )
    labels = np.asarray(model.fit_predict(X), dtype=int)
    return model, labels


def fit_optics(
    X: np.ndarray,
    *,
    min_samples: int = 5,
    max_eps: float = np.inf,
    metric: str = "minkowski",
    p: int = 2,
    cluster_method: str = "xi",
    xi: float = 0.05,
    min_cluster_size: int | None = None,
    eps: float | None = None,
    **kwargs: Any,
) -> tuple[OPTICS, np.ndarray]:
    """Fit OPTICS; returns ``(model, labels)`` with noise labeled ``-1``."""
    X = np.asarray(X, dtype=float)
    model = OPTICS(
        min_samples=int(min_samples),
        max_eps=float(max_eps),
        metric=metric,
        p=int(p),
        cluster_method=cluster_method,
        xi=float(xi),
        min_cluster_size=min_cluster_size,
        eps=eps,
        **kwargs,
    )
    labels = np.asarray(model.fit_predict(X), dtype=int)
    return model, labels


def cluster_centroids(X: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(centroids, cluster_ids)`` for non-noise labels (sorted)."""
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels, dtype=int)
    ids = np.unique(labels)
    ids = ids[ids != NOISE_LABEL]
    if ids.size == 0:
        return np.zeros((0, X.shape[1]), dtype=float), ids
    cents = np.vstack([X[labels == cid].mean(axis=0) for cid in ids])
    return cents, ids


def distances_to_centroids(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Euclidean distances from each row of ``X`` to each centroid."""
    X = np.asarray(X, dtype=float)
    if centroids.size == 0:
        return np.zeros((X.shape[0], 1), dtype=float)
    return pairwise_distances(X, centroids, metric="euclidean")


def density_embedding(
    X: np.ndarray,
    labels: np.ndarray,
    *,
    model: Any | None = None,
) -> np.ndarray:
    """Distances-to-centers embedding (k-means-style) for density clusters.

    Prefer ``model.centroids_`` when HDBSCAN stored them; otherwise compute
    means of non-noise members. If no clusters are found, return a single
    zero column so downstream dimred still has a valid frame.
    """
    X = np.asarray(X, dtype=float)
    centroids = getattr(model, "centroids_", None) if model is not None else None
    if centroids is None or np.asarray(centroids).size == 0:
        centroids, _ = cluster_centroids(X, labels)
    else:
        centroids = np.asarray(centroids, dtype=float)
    return distances_to_centroids(X, centroids)


def adaptive_min_cluster_size(n_samples: int, requested: int | None = None) -> int:
    """Clamp ``min_cluster_size`` so small matrices (e.g. regime means) still work."""
    if n_samples < 2:
        return 2
    default = max(2, min(5, n_samples))
    if requested is None:
        return default
    return int(max(2, min(int(requested), n_samples)))


def cluster_labels(
    X: np.ndarray,
    algorithm: str,
    *,
    n_clusters: int | None = None,
    random_state: int | None = 0,
    hdbscan_kwargs: dict | None = None,
    optics_kwargs: dict | None = None,
) -> np.ndarray:
    """Fit ``kmeans`` / ``hdbscan`` / ``optics`` and return integer labels."""
    from sklearn.cluster import KMeans

    X = np.asarray(X, dtype=float)
    algo = algorithm.lower()
    if algo == "kmeans":
        k = n_clusters or min(max(2, X.shape[0] // 2), X.shape[0])
        k = max(1, min(int(k), X.shape[0]))
        return np.asarray(
            KMeans(n_clusters=k, n_init=10, random_state=random_state).fit_predict(X),
            dtype=int,
        )
    if algo == "hdbscan":
        kw = dict(hdbscan_kwargs or {})
        kw["min_cluster_size"] = adaptive_min_cluster_size(
            X.shape[0], kw.get("min_cluster_size")
        )
        _model, labels = fit_hdbscan(X, **kw)
        return labels
    if algo == "optics":
        kw = dict(optics_kwargs or {})
        ms = kw.get("min_samples", adaptive_min_cluster_size(X.shape[0]))
        kw["min_samples"] = max(2, min(int(ms), X.shape[0]))
        if "min_cluster_size" in kw and kw["min_cluster_size"] is not None:
            kw["min_cluster_size"] = adaptive_min_cluster_size(
                X.shape[0], kw["min_cluster_size"]
            )
        _model, labels = fit_optics(X, **kw)
        return labels
    raise ValueError(
        f"Unknown clustering algorithm {algorithm!r}. "
        "Supported: kmeans, hdbscan, optics."
    )
