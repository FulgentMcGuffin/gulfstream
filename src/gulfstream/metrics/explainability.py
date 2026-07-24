"""Explainability decision trees for regime labels.

Tree diagrams use sklearn ``plot_tree`` (matplotlib axes). Other gulfstream
gallery plots prefer plotnine via ``gulfstream.common.plotting``.
"""
from __future__ import annotations

import logging
import os
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.tree import DecisionTreeClassifier, plot_tree

from gulfstream.features import names as feature_name_resolution
from gulfstream.common import frames
from gulfstream.common import utils
from gulfstream.common.results import SegmentResults
from gulfstream.detection.time_index import bkpts_to_labels

logger = logging.getLogger(__name__)


def _get_explainability_df(
    df: pl.DataFrame,
    explainability_features: dict | list[str] | None,
) -> pl.DataFrame:
    """Return a dataframe restricted to explainability feature columns."""
    if explainability_features is None:
        return df.clone()
    cols = feature_name_resolution.get_column_names(explainability_features)
    if not cols or cols == ["__auto__"]:
        return df.clone()
    feat_cols = set(frames.feature_columns(df))
    present = [c for c in cols if c in feat_cols]
    if not present:
        logger.warning("No explainability features found in df; using all columns.")
        return df.clone()
    return frames.select_features(df, present)


def _regime_labels(df: pl.DataFrame, res: SegmentResults) -> np.ndarray:
    if res.labels is not None and len(res.labels) == df.height:
        return np.asarray(res.labels, dtype=int)
    return np.asarray(bkpts_to_labels(list(res.bkpts or []), df.height), dtype=int)


def _scale_features(X: np.ndarray, feature_names: list[str], bps_decimals: bool) -> np.ndarray:
    if not bps_decimals:
        return X
    out = X.copy()
    for i, c in enumerate(feature_names):
        name = str(c).lower()
        if "corr" in name or "vol" in name:
            continue
        out[:, i] = out[:, i] * 100.0
    return out


def _fit_accurate_tree(
    X: np.ndarray,
    y: np.ndarray,
    target_accuracy: float,
) -> DecisionTreeClassifier:
    """Grow a tree until train accuracy meets target (or max depth)."""
    best = None
    best_acc = -1.0
    for depth in range(1, 21):
        clf = DecisionTreeClassifier(max_depth=depth, random_state=0)
        clf.fit(X, y)
        acc = float(clf.score(X, y))
        best = clf
        best_acc = acc
        if acc >= target_accuracy:
            break
    logger.info(
        "Accurate tree depth=%s train_acc=%.3f target=%.3f",
        best.get_depth(),
        best_acc,
        target_accuracy,
    )
    return best


def _fit_shallow_tree(
    X: np.ndarray, y: np.ndarray, max_depth: int = 3
) -> DecisionTreeClassifier:
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=1)
    clf.fit(X, y)
    logger.info(
        "Shallow tree depth=%s train_acc=%.3f", clf.get_depth(), clf.score(X, y)
    )
    return clf


def _save_tree_plot(
    clf: DecisionTreeClassifier,
    feature_names: list[str],
    *,
    title: str,
    gallery_key: str,
    img_dir: str | None,
    mode: str,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 8))
    plot_tree(
        clf,
        feature_names=feature_names,
        class_names=[str(c) for c in clf.classes_],
        filled=True,
        rounded=True,
        fontsize=7,
        ax=ax,
    )
    ax.set_title(title)
    fig.tight_layout()
    if mode in ("write", "display_and_write") and img_dir:
        os.makedirs(img_dir, exist_ok=True)
        path = os.path.join(img_dir, utils.img_gallery_filename(gallery_key).lstrip("/"))
        fig.savefig(path, dpi=120)
        logger.info("Saved explainability tree to %s", path)
    plt.close(fig)


def _save_tree_pickle(
    clf: DecisionTreeClassifier, gallery_key: str, img_dir: str | None
) -> str | None:
    if not img_dir:
        return None
    os.makedirs(img_dir, exist_ok=True)
    path = os.path.join(img_dir, utils.img_gallery_filename(gallery_key).lstrip("/"))
    with open(path, "wb") as f:
        pickle.dump(clf, f)
    logger.info("Pickled tree to %s", path)
    return path


def produce_all_explainability_tools(
    df: pl.DataFrame,
    params: dict,
    res: SegmentResults,
) -> None:
    metrics = params.get("metrics", {})
    mode = metrics.get("mode", "write")
    img_dir = metrics.get("image_dir") or metrics.get("dir")
    exp_df = _get_explainability_df(df, metrics.get("explainability_features"))
    feat_cols = frames.feature_columns(exp_df)
    n_feat = min(int(metrics.get("num_features") or 8), len(feat_cols))
    feature_names = feat_cols[:n_feat]
    exp_df = frames.select_features(exp_df, feature_names)
    bps = bool(metrics.get("exp_tree_bps_decimals", False))
    X = _scale_features(frames.to_numpy(exp_df), feature_names, bps)
    y = _regime_labels(df, res)
    if len(np.unique(y)) < 2:
        logger.warning("Fewer than 2 regimes; skipping explainability trees.")
        return

    targets = metrics.get("exp_tree_accuracy") or [0.9]
    shallow = _fit_shallow_tree(X, y, max_depth=3)
    _save_tree_plot(
        shallow,
        feature_names,
        title="Shallow explainability tree (transitions)",
        gallery_key="shallow_exp_tree",
        img_dir=img_dir,
        mode=mode,
    )
    shallow_path = _save_tree_pickle(shallow, "shallow_exp_tree_pickle", img_dir)

    accurate_paths = []
    for target in targets:
        clf = _fit_accurate_tree(X, y, float(target))
        key = f"acc{int(float(target) * 100)}_exp_tree"
        _save_tree_plot(
            clf,
            feature_names,
            title=f"Accurate explainability tree (target={target})",
            gallery_key=key,
            img_dir=img_dir,
            mode=mode,
        )
        pkl_key = f"acc{int(float(target) * 100)}_exp_tree_pickle"
        path = _save_tree_pickle(clf, pkl_key, img_dir)
        accurate_paths.append(path)

    params.setdefault("_explainability", {})
    params["_explainability"]["shallow_tree_path"] = shallow_path
    params["_explainability"]["accurate_tree_paths"] = accurate_paths
    params["_explainability"]["feature_names"] = list(feature_names)
