"""Shared utilities for the gulfstream regime pipeline."""
from __future__ import annotations

import logging
import os
from typing import Literal, Union

import numpy as np
import polars as pl
import yaml
from sklearn.metrics import pairwise_distances

from gulfstream.common import frames

logger = logging.getLogger(__name__)

IMG_EXT = ".png"
HTML_EXT = ".html"
TXT_EXT = ".txt"
PKL_EXT = ".pkl"
SHEET_EXT = ".csv"
CLUSTERS_PREFIX = "30"
MATRIX_PREFIX = "31"


def read_config_yaml(filename: str, img_dir: str, log_dir: str) -> dict:
    """Read YAML config, substituting ${LOG_DIR} and ${IMG_DIR}."""
    with open(filename, encoding="utf-8") as f:
        config_text = f.read()
    config_text = config_text.replace("${LOG_DIR}", log_dir)
    config_text = config_text.replace("${IMG_DIR}", img_dir)
    return yaml.safe_load(config_text)


def _calculate_bandwidth(
    df: pl.DataFrame,
    gamma: Union[
        float,
        Literal[
            "median",
            "sk_scale",
            "total_variance",
            "unscaled_max_variance",
            "max_variance",
        ],
    ],
) -> float:
    """Calculate RBF kernel bandwidth gamma."""
    X = frames.to_numpy(df)
    n_feat = frames.n_features(df)
    feat_var = X.var(axis=0) if X.size else np.array([])
    if gamma == "median":
        d = pairwise_distances(X)
        med = float(np.median(d))
        if med == 0:
            logger.error("Median pairwise distance is 0.")
            med = 1e-8
        return 1.0 / (2 * med**2)
    if gamma == "sk_scale":
        return 1.0 / (n_feat * float(X.var()))
    if gamma == "total_variance":
        return 1.0 / (2 * float(feat_var.sum()))
    if gamma == "max_variance":
        return 1.0 / (n_feat * float(feat_var.max()))
    if gamma == "unscaled_max_variance":
        return 1.0 / (2 * float(feat_var.max()))
    if not isinstance(gamma, (int, float)):
        raise TypeError(f"Unknown value of gamma {gamma}.")
    return float(gamma)


def map_labels_to_ordered_integers(labels: list) -> list:
    """Map contiguous label groups to ordered integers starting at 0."""
    mapped: list = []
    current_mapping: dict = {}
    current_index = 0
    for i, label in enumerate(labels):
        if i == 0 or label != labels[i - 1]:
            if label not in current_mapping:
                current_mapping[label] = current_index
                current_index += 1
        mapped.append(current_mapping[label])
    return mapped


def _map_labels_to_ordered_integers(labels: list) -> list:
    """Alias used by legacy regime detection methods."""
    return map_labels_to_ordered_integers(labels)


def _convert_labels_to_bkpts(labels: list) -> list:
    bkpts = []
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            bkpts.append(i)
    return bkpts


def _convert_breakpoints_to_labels(length: int, bk: list) -> list:
    """Convert breakpoints to regime labels of the given series length."""
    labels = np.zeros(length, dtype=int)
    current_label = 0
    bk_set = set(bk)
    for i in range(length):
        if i in bk_set:
            current_label += 1
        labels[i] = current_label
    return list(labels)


def _convert_bkpts_to_labels(bkpts: list[int], length: int) -> list[int]:
    labels = np.zeros(length, dtype=int)
    edges = [0] + sorted(b for b in bkpts if 0 < b < length) + [length]
    for i in range(len(edges) - 1):
        labels[edges[i] : edges[i + 1]] = i
    return labels.tolist()


def _img_gallery_filename(name: str, **kwargs) -> str:
    lookup = {
        "bkpt_tree": f"/01_proc_bkpt_tree{IMG_EXT}",
        "regime_plot": f"/02_regime_plot{IMG_EXT}",
        "avg_feature_L2": f"/04_avg_L2{IMG_EXT}",
        "aggregated_feature_L2": f"/05_aggregated_L2{IMG_EXT}",
        "avg_pairwise_euclidean_clusters": f"/{CLUSTERS_PREFIX}_euclidean_clusters{IMG_EXT}",
        "mahalanobis_clusters": f"/{CLUSTERS_PREFIX}_mahalanobis_clusters{IMG_EXT}",
        "avg_pairwise_euclidean_matrix": f"/{MATRIX_PREFIX}_euclidean_matrix{IMG_EXT}",
        "mahalanobis_matrix": f"/{MATRIX_PREFIX}_mahalanobis_matrix{IMG_EXT}",
        "unproc_tree": f"/40_unproc_bkpt_tree{IMG_EXT}",
        "transition_distance": f"/32_transition_distance{IMG_EXT}",
        "transition_tree": f"/33_transition_tree{IMG_EXT}",
        "persistence_summary": f"/34_persistence_summary{IMG_EXT}",
        "stability_summary": f"/35_stability_summary{IMG_EXT}",
        "shallow_exp_tree": f"/03_shallow_exp_tree{IMG_EXT}",
        "accurate_exp_tree": f"/03_accurate_exp_tree{IMG_EXT}",
        "shallow_exp_tree_pickle": f"/shallow_exp_tree{PKL_EXT}",
        "accurate_exp_tree_pickle": f"/accurate_exp_tree{PKL_EXT}",
    }
    if name in lookup:
        return lookup[name]
    if name.startswith("retrain") and name.endswith("avg_feature_L2"):
        return f"/{name}{IMG_EXT}"
    if name.endswith("exp_tree"):
        return f"/03_{name}{IMG_EXT}"
    if name.endswith("exp_tree_pickle"):
        return f"/{name}{PKL_EXT}"
    return f"/{name}{IMG_EXT}"


def _generate_gallery(image_dir: str) -> None:
    """Write a simple HTML gallery of PNGs in image_dir."""
    if not image_dir or not os.path.isdir(image_dir):
        return
    pngs = []
    for root, _, files in os.walk(image_dir):
        for f in files:
            if f.lower().endswith(".png"):
                rel = os.path.relpath(os.path.join(root, f), image_dir)
                pngs.append(rel.replace("\\", "/"))
    pngs.sort()
    lines = [
        "<html><head><title>Gulfstream gallery</title></head><body>",
        "<h1>Regime outputs</h1>",
    ]
    for p in pngs:
        lines.append(f'<div><h3>{p}</h3><img src="{p}" style="max-width:100%"/></div>')
    lines.append("</body></html>")
    with open(os.path.join(image_dir, "gallery.html"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
