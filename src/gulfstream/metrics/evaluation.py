"""Evaluation helpers for regime L2 losses and breakpoint matching."""
from __future__ import annotations

import logging

import numpy as np
import polars as pl
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import directed_hausdorff

from gulfstream.common import frames

logger = logging.getLogger(__name__)


def _regime_edges(bkpts: list[int], length: int) -> list[int]:
    return [0] + sorted(b for b in bkpts if 0 < b < length) + [length]


def _local_loss(df: pl.DataFrame, start: int, end: int) -> float:
    """Total L2 loss of rows [start:end) versus column-wise mean."""
    if end <= start:
        return 0.0
    segment = frames.to_numpy(frames.slice_rows(df, start, end))
    mean = segment.mean(axis=0, keepdims=True)
    return float(np.linalg.norm(segment - mean) ** 2)


def _avg_features_loss(df: pl.DataFrame, bkpts: list[int]) -> np.ndarray:
    """Average daily per-feature L2 to regime mean.

    Returns
    -------
    np.ndarray
        Shape ``(n_features, n_regimes)``.
    """
    n = df.height
    edges = _regime_edges(bkpts, n)
    n_regimes = len(edges) - 1
    n_features = frames.n_features(df)
    out = np.zeros((n_features, n_regimes), dtype=float)
    values = frames.to_numpy(df)
    for j in range(n_regimes):
        a, b = edges[j], edges[j + 1]
        if b <= a:
            continue
        seg = values[a:b]
        mean = seg.mean(axis=0, keepdims=True)
        # Mean squared deviation per feature, averaged over days.
        out[:, j] = ((seg - mean) ** 2).mean(axis=0)
    return out


def _match(true_bkpts: list[int], pred_bkpts: list[int]) -> dict:
    """Hungarian matching from true breakpoints to predicted ones.

    Returns mapping ``true_bkpt -> (matched_pred, distance)``.
    Unmatched true breakpoints map to ``(None, inf)``.
    """
    true_bkpts = list(true_bkpts)
    pred_bkpts = list(pred_bkpts)
    if not true_bkpts:
        return {}
    if not pred_bkpts:
        return {t: (None, float("inf")) for t in true_bkpts}

    cost = np.zeros((len(true_bkpts), len(pred_bkpts)), dtype=float)
    for i, t in enumerate(true_bkpts):
        for j, p in enumerate(pred_bkpts):
            cost[i, j] = abs(t - p)
    row_ind, col_ind = linear_sum_assignment(cost)
    mapping: dict = {t: (None, float("inf")) for t in true_bkpts}
    for r, c in zip(row_ind, col_ind):
        mapping[true_bkpts[r]] = (pred_bkpts[c], float(cost[r, c]))
    return mapping


def _euclidean_mapping(pred_bkpts: list[int], true_bkpts: list[int]) -> dict:
    """Map each predicted breakpoint to the nearest true breakpoint."""
    pred_bkpts = list(pred_bkpts)
    true_bkpts = list(true_bkpts)
    if not pred_bkpts:
        return {}
    if not true_bkpts:
        return {p: (None, float("inf")) for p in pred_bkpts}
    mapping = {}
    true_arr = np.asarray(true_bkpts, dtype=float)
    for p in pred_bkpts:
        dists = np.abs(true_arr - p)
        j = int(np.argmin(dists))
        mapping[p] = (true_bkpts[j], float(dists[j]))
    return mapping


def _directed_hausdorff_wrapper(
    true_bkpts: list[int],
    pred_bkpts: list[int],
    length: int,
) -> float:
    """Directed Hausdorff distance between breakpoint sets (as 1-D points)."""
    if not true_bkpts and not pred_bkpts:
        return 0.0
    if not true_bkpts or not pred_bkpts:
        return float(length)
    u = np.asarray(true_bkpts, dtype=float).reshape(-1, 1)
    v = np.asarray(pred_bkpts, dtype=float).reshape(-1, 1)
    return float(directed_hausdorff(u, v)[0])


def recovery_rate(
    baseline_bkpts: list[int],
    other_bkpts: list[int],
    *,
    tolerance: int = 5,
) -> float:
    """Fraction of baseline breakpoints recovered within ``tolerance`` days."""
    if not baseline_bkpts:
        return 1.0
    mapping = _match(baseline_bkpts, other_bkpts)
    hits = sum(
        1
        for _, (matched, dist) in mapping.items()
        if matched is not None and dist <= tolerance
    )
    return hits / len(baseline_bkpts)
