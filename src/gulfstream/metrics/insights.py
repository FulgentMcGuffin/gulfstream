"""Post-information visualization: L2 heatmaps, distances, clustering."""

from __future__ import annotations

import logging
import os
from typing import Literal

import numpy as np
import polars as pl
from plotnine import ggplot
from sklearn.metrics import pairwise_distances

from gulfstream.common import frames, plotting, utils
from gulfstream.common.results import SegmentResults
from gulfstream.dimred import density as dens
from gulfstream.metrics import evaluation as evaluation_tools

logger = logging.getLogger(__name__)

_DEFAULT_REGIME_CLUSTER_ALGORITHMS = ["kmeans"]
_SUPPORTED_REGIME_CLUSTER_ALGORITHMS = ("kmeans", "hdbscan", "optics")


def _select_feature_columns(df: pl.DataFrame, params: dict) -> list[str]:
    metrics = params.get("metrics", {})
    requested = metrics.get("requested_features_for_distances") or []
    if isinstance(requested, dict):
        requested = list(requested.keys())
    feat_cols = frames.feature_columns(df)
    cols = [c for c in requested if c in feat_cols]
    if cols:
        return cols
    n_top = int(metrics.get("num_top_features_for_distances") or min(5, len(feat_cols)))
    preferred = [
        c for c in feat_cols if "vol" not in c.lower() and "corr" not in c.lower()
    ]
    pool = preferred if preferred else list(feat_cols)
    return pool[:n_top]


def draw_error_heatmaps(
    loss_matrix: np.ndarray,
    df: pl.DataFrame,
    bkpts: list[int],
    *,
    title: str,
    cbar_label: str,
    mode: Literal["display", "write", "display_and_write"] = "write",
    img_dir: str | None = None,
    gallery_filename: str = "avg_feature_L2",
    feature_names: list[str] | None = None,
) -> ggplot:
    """Draw a feature×regime L2 heatmap and optionally save it."""
    n_regimes = loss_matrix.shape[1]
    feat_cols = frames.feature_columns(df)
    feature_names = feature_names or [str(c) for c in feat_cols[: loss_matrix.shape[0]]]
    ylabels = [f[:40] for f in feature_names]
    xlabels = [f"R{i}" for i in range(n_regimes)]
    plot = plotting.ggplot_heatmap(
        loss_matrix,
        title=title,
        x_labels=xlabels,
        y_labels=ylabels,
        fill_label=cbar_label,
        cmap="viridis",
    )
    path = None
    if img_dir:
        os.makedirs(img_dir, exist_ok=True)
        path = os.path.join(
            img_dir, utils.img_gallery_filename(gallery_filename).lstrip("/")
        )
    plotting.emit_ggplot(
        plot,
        path=path,
        mode=mode,
        width=max(6, n_regimes),
        height=max(4, len(ylabels) * 0.35),
        log_label="heatmap",
    )
    return plot


def _regime_feature_means(
    df: pl.DataFrame, bkpts: list[int], cols: list[str]
) -> np.ndarray:
    n = df.height
    edges = evaluation_tools._regime_edges(bkpts, n)
    means = []
    sub = frames.to_numpy(frames.select_features(df, cols))
    for i in range(len(edges) - 1):
        a, b = edges[i], edges[i + 1]
        if b <= a:
            means.append(np.zeros(len(cols)))
        else:
            means.append(sub[a:b].mean(axis=0))
    return np.vstack(means) if means else np.zeros((0, len(cols)))


def _mahalanobis_distance_matrix(
    means: np.ndarray, df: pl.DataFrame, cols: list[str]
) -> np.ndarray:
    if means.shape[0] == 0:
        return np.zeros((0, 0))
    cov = np.cov(frames.to_numpy(frames.select_features(df, cols)), rowvar=False)
    if cov.ndim == 0:
        cov = np.array([[float(cov) + 1e-6]])
    cov = cov + np.eye(cov.shape[0]) * 1e-6
    try:
        inv = np.linalg.pinv(cov)
    except Exception:
        inv = np.eye(cov.shape[0])
    n = means.shape[0]
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            d = means[i] - means[j]
            dist[i, j] = float(np.sqrt(max(0.0, d @ inv @ d)))
    return dist


def _draw_matrix(
    matrix: np.ndarray,
    *,
    title: str,
    gallery_key: str,
    mode: str,
    img_dir: str | None,
) -> None:
    if matrix.size == 0:
        return
    labels = [str(i) for i in range(matrix.shape[0])]
    plot = plotting.ggplot_heatmap(
        matrix,
        title=title,
        x_labels=labels,
        y_labels=labels,
        fill_label="distance",
        cmap="magma",
        annotate=matrix.shape[0] <= 8,
    )
    path = None
    if img_dir:
        os.makedirs(img_dir, exist_ok=True)
        path = os.path.join(
            img_dir, utils.img_gallery_filename(gallery_key).lstrip("/")
        )
    plotting.emit_ggplot(
        plot, path=path, mode=mode, width=6, height=5, log_label="matrix"
    )


def _draw_clusters(
    means: np.ndarray,
    labels: np.ndarray,
    *,
    title: str,
    gallery_key: str,
    mode: str,
    img_dir: str | None,
) -> None:
    if means.shape[0] == 0:
        return
    if means.shape[1] >= 2:
        xy = means[:, :2]
    else:
        xy = np.column_stack([means[:, 0], np.zeros(len(means))])
    plot = plotting.ggplot_clusters_2d(xy, labels, title=title)
    path = None
    if img_dir:
        os.makedirs(img_dir, exist_ok=True)
        path = os.path.join(
            img_dir, utils.img_gallery_filename(gallery_key).lstrip("/")
        )
    plotting.emit_ggplot(
        plot, path=path, mode=mode, width=6, height=5, log_label="clusters"
    )


def produce_all_post_information_visualization_tools(
    df: pl.DataFrame,
    params: dict,
    res: SegmentResults,
) -> None:
    """Emit L2 heatmaps, pairwise regime distances, and simple clustering plots."""
    metrics = params.get("metrics", {})
    mode = metrics.get("mode", "write")
    img_dir = metrics.get("image_dir") or metrics.get("dir")
    bkpts = list(res.bkpts or [])

    loss = evaluation_tools.avg_features_loss(df, bkpts)
    n_show = min(int(metrics.get("num_features") or 10), loss.shape[0])
    order = np.argsort(-loss.sum(axis=1))[:n_show]
    feat_cols = frames.feature_columns(df)
    feature_names = [str(feat_cols[i]) for i in order]
    loss_top = loss[order, :]

    draw_error_heatmaps(
        loss_top,
        df,
        bkpts,
        title="Average daily L2 error by feature × regime",
        cbar_label="avg daily L2",
        mode=mode,
        img_dir=img_dir,
        gallery_filename="avg_feature_L2",
        feature_names=feature_names,
    )
    if loss_top.shape[1] > 0:
        aggregated = loss_top.mean(axis=0, keepdims=True)
        draw_error_heatmaps(
            aggregated,
            df,
            bkpts,
            title="Aggregated avg L2 across top features",
            cbar_label="avg daily L2",
            mode=mode,
            img_dir=img_dir,
            gallery_filename="aggregated_feature_L2",
            feature_names=["aggregated"],
        )

    cols = _select_feature_columns(df, params)
    if not cols:
        logger.warning("No feature columns available for distance insights.")
        return
    means = _regime_feature_means(df, bkpts, cols)
    if means.shape[0] < 2:
        logger.info("Fewer than 2 regimes; skipping distance/cluster insights.")
        return

    euc = pairwise_distances(means, metric="euclidean")
    mah = _mahalanobis_distance_matrix(means, df, cols)
    _draw_matrix(
        euc,
        title="Regime pairwise Euclidean distance",
        gallery_key="avg_pairwise_euclidean_matrix",
        mode=mode,
        img_dir=img_dir,
    )
    _draw_matrix(
        mah,
        title="Regime pairwise Mahalanobis distance",
        gallery_key="mahalanobis_matrix",
        mode=mode,
        img_dir=img_dir,
    )

    n_clusters = min(max(2, means.shape[0] // 2), means.shape[0])
    algorithms = metrics.get("regime_cluster_algorithms") or list(
        _DEFAULT_REGIME_CLUSTER_ALGORITHMS
    )
    if isinstance(algorithms, str):
        algorithms = [algorithms]
    hdbscan_kw = {
        "min_cluster_size": metrics.get("hdbscan_min_cluster_size"),
        "min_samples": metrics.get("hdbscan_min_samples"),
        "metric": metrics.get("hdbscan_metric", "euclidean"),
    }
    hdbscan_kw = {k: v for k, v in hdbscan_kw.items() if v is not None}
    optics_kw = {
        "min_samples": metrics.get("optics_min_samples"),
        "min_cluster_size": metrics.get("optics_min_cluster_size"),
        "metric": metrics.get("optics_metric", "minkowski"),
        "xi": metrics.get("optics_xi"),
    }
    optics_kw = {k: v for k, v in optics_kw.items() if v is not None}

    for algo in algorithms:
        algo_l = str(algo).lower()
        if algo_l not in _SUPPORTED_REGIME_CLUSTER_ALGORITHMS:
            logger.warning(
                "Skipping unknown regime_cluster_algorithms entry %r (supported: %s).",
                algo,
                ", ".join(_SUPPORTED_REGIME_CLUSTER_ALGORITHMS),
            )
            continue
        try:
            labels = dens.cluster_labels(
                means,
                algo_l,
                n_clusters=n_clusters,
                random_state=0,
                hdbscan_kwargs=hdbscan_kw,
                optics_kwargs=optics_kw,
            )
        except Exception:
            logger.exception("%s clustering (euclidean) failed", algo_l)
            labels = np.arange(means.shape[0])
        suffix = "" if algo_l == "kmeans" else f"_{algo_l}"
        _draw_clusters(
            means,
            labels,
            title=f"Regime clusters ({algo_l}, Euclidean on means)",
            gallery_key=f"avg_pairwise_euclidean_clusters{suffix}",
            mode=mode,
            img_dir=img_dir,
        )
        try:
            labels_m = dens.cluster_labels(
                means,
                algo_l,
                n_clusters=n_clusters,
                random_state=1,
                hdbscan_kwargs=hdbscan_kw,
                optics_kwargs=optics_kw,
            )
        except Exception:
            labels_m = labels
        _draw_clusters(
            means,
            labels_m,
            title=f"Regime clusters ({algo_l}, Mahalanobis-oriented)",
            gallery_key=f"mahalanobis_clusters{suffix}",
            mode=mode,
            img_dir=img_dir,
        )

    params.setdefault("_insights", {})["euclidean_dist"] = euc
    params["_insights"]["mahalanobis_dist"] = mah
    params["_insights"]["regime_means"] = means
    params["_insights"]["distance_features"] = cols
