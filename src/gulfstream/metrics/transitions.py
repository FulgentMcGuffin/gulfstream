"""Regime transition matrices from distances and explainability trees."""
from __future__ import annotations

import logging
import os
import pickle

import numpy as np
import polars as pl
from sklearn.tree import DecisionTreeClassifier

from gulfstream.common import plotting, utils

logger = logging.getLogger(__name__)


def generate_distance_based_transition_probabilities(
    dist_matrix: np.ndarray | None = None,
    *,
    gamma: float = 1.0,
    regimes_df: pl.DataFrame | None = None,
    df: pl.DataFrame | None = None,
    **kwargs,
) -> pl.DataFrame:
    """Transition probs ∝ exp(-gamma * distance); rows normalized."""
    if dist_matrix is None:
        raise ValueError("dist_matrix is required for distance-based transitions.")
    mat = np.asarray(dist_matrix, dtype=float)
    transition = np.exp(-gamma * mat)
    # Zero self-loops optional; keep soft self mass then renormalize.
    row_sums = transition.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    transition = transition / row_sums
    cols = [str(i) for i in range(transition.shape[0])]
    data = {c: transition[:, i] for i, c in enumerate(cols)}
    return pl.DataFrame(data)


def generate_tree_based_distribution(tree_filename: str) -> np.ndarray:
    """Leaf-mass distribution over regimes from a pickled decision tree."""
    with open(tree_filename, "rb") as f:
        clf: DecisionTreeClassifier = pickle.load(f)
    tree = clf.tree_
    # Weighted regime counts from training samples reaching each leaf.
    # value shape: (n_nodes, 1, n_classes)
    n_classes = len(clf.classes_)
    masses = np.zeros(n_classes, dtype=float)
    for node in range(tree.node_count):
        if tree.children_left[node] == -1:  # leaf
            values = tree.value[node][0]
            masses += values
    total = masses.sum()
    if total <= 0:
        return np.ones(n_classes) / max(n_classes, 1)
    # Map class indices to regime ids via clf.classes_
    full = np.zeros(int(max(clf.classes_)) + 1, dtype=float)
    for i, regime in enumerate(clf.classes_):
        full[int(regime)] = masses[i] / total
    return full


def generate_tree_based_transition_probabilities(
    tree_filename: str, **kwargs
) -> pl.DataFrame:
    """Spatially homogeneous transitions from shallow-tree leaf masses."""
    probs = generate_tree_based_distribution(tree_filename)
    nonzero_idx = [i for i, v in enumerate(probs) if v != 0.0]
    nonzero_probs = [probs[i] for i in nonzero_idx]
    if not nonzero_idx:
        return pl.DataFrame()
    transition_mat = np.vstack([nonzero_probs] * len(nonzero_probs))
    cols = [str(i) for i in nonzero_idx]
    data = {c: transition_mat[:, j] for j, c in enumerate(cols)}
    return pl.DataFrame(data)


def _draw_transition(
    mat: pl.DataFrame,
    *,
    title: str,
    gallery_key: str,
    img_dir: str | None,
    mode: str,
) -> None:
    if mat is None or mat.height == 0:
        return
    arr = mat.to_numpy()
    labels = list(mat.columns)
    plot = plotting.ggplot_heatmap(
        arr,
        title=title,
        x_labels=labels,
        y_labels=labels,
        fill_label="P",
        cmap="Blues",
        annotate=arr.shape[0] <= 10,
    )
    path = None
    if img_dir:
        os.makedirs(img_dir, exist_ok=True)
        path = os.path.join(img_dir, utils.img_gallery_filename(gallery_key).lstrip("/"))
    plotting.emit_ggplot(
        plot, path=path, mode=mode, width=6, height=5, log_label="transition matrix"
    )


def produce_transition_matrices(
    df: pl.DataFrame,
    params: dict,
    *,
    gamma: float = 1.0,
) -> None:
    """Write distance- and tree-based transition heatmaps using cached insights."""
    metrics = params.get("metrics", {})
    mode = metrics.get("mode", "write")
    img_dir = metrics.get("image_dir") or metrics.get("dir")
    insights = params.get("_insights") or {}
    dist = insights.get("euclidean_dist")
    if dist is not None:
        t_dist = generate_distance_based_transition_probabilities(dist, gamma=gamma)
        _draw_transition(
            t_dist,
            title="Distance-based regime transitions",
            gallery_key="transition_distance",
            img_dir=img_dir,
            mode=mode,
        )
        params.setdefault("_transitions", {})["distance"] = t_dist

    shallow = (params.get("_explainability") or {}).get("shallow_tree_path")
    if shallow and os.path.isfile(shallow):
        t_tree = generate_tree_based_transition_probabilities(shallow)
        _draw_transition(
            t_tree,
            title="Tree-based regime transitions (shallow)",
            gallery_key="transition_tree",
            img_dir=img_dir,
            mode=mode,
        )
        params.setdefault("_transitions", {})["tree"] = t_tree
