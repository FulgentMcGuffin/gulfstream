"""Evaluation helpers for regime L2 losses and breakpoint matching."""
from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import polars as pl
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import directed_hausdorff
from sklearn.metrics import adjusted_rand_score

from gulfstream.common import frames
from gulfstream.common.utils import convert_bkpts_to_labels

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


def avg_features_loss(df: pl.DataFrame, bkpts: list[int]) -> np.ndarray:
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
    mapping = match_breakpoints(baseline_bkpts, other_bkpts)
    hits = sum(
        1
        for _, (matched, dist) in mapping.items()
        if matched is not None and dist <= tolerance
    )
    return hits / len(baseline_bkpts)


def breakpoint_precision_recall_f1(
    true_bkpts: list[int],
    pred_bkpts: list[int],
    *,
    tolerance: int = 5,
) -> dict[str, float]:
    """Breakpoint precision / recall / F1 at a matching tolerance."""
    true_bkpts = list(true_bkpts)
    pred_bkpts = list(pred_bkpts)
    if not true_bkpts and not pred_bkpts:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred_bkpts:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    if not true_bkpts:
        return {"precision": 0.0, "recall": 1.0 if not pred_bkpts else 0.0, "f1": 0.0}

    mapping = match_breakpoints(true_bkpts, pred_bkpts)
    tp = sum(
        1
        for _, (matched, dist) in mapping.items()
        if matched is not None and dist <= tolerance
    )
    precision = tp / len(pred_bkpts)
    recall = tp / len(true_bkpts)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def covering_metric(
    true_bkpts: list[int],
    pred_bkpts: list[int],
    length: int,
) -> float:
    """Segmentation covering score in [0, 1] (Arbelaez et al. / ruptures-style).

    For each true regime segment, weight the best-overlapping predicted segment
    by the true segment length, then average over the series.
    """
    if length <= 0:
        return 1.0
    true_edges = _regime_edges(true_bkpts, length)
    pred_edges = _regime_edges(pred_bkpts, length)
    true_segs = list(zip(true_edges[:-1], true_edges[1:]))
    pred_segs = list(zip(pred_edges[:-1], pred_edges[1:]))
    score = 0.0
    for ts, te in true_segs:
        tlen = te - ts
        if tlen <= 0:
            continue
        best = 0.0
        for ps, pe in pred_segs:
            overlap = max(0, min(te, pe) - max(ts, ps))
            best = max(best, overlap / tlen)
        score += tlen * best
    return float(score / length)


def adjusted_rand_index_labels(
    true_labels: Sequence[int],
    pred_labels: Sequence[int],
) -> float:
    """Adjusted Rand index between two per-timestep regime labelings.

    Label values need not match across runs (permutation-invariant). Chance
    agreement scores near 0; identical partitions score 1.0.
    """
    y_true = np.asarray(true_labels, dtype=int).ravel()
    y_pred = np.asarray(pred_labels, dtype=int).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Label length mismatch: true={y_true.shape[0]} pred={y_pred.shape[0]}"
        )
    if y_true.size == 0:
        return 1.0
    return float(adjusted_rand_score(y_true, y_pred))


def adjusted_rand_index(
    true_bkpts: list[int],
    pred_bkpts: list[int],
    length: int,
) -> float:
    """Adjusted Rand index between segmentations induced by breakpoints.

    Converts each breakpoint list to contiguous regime labels of length
    ``length``, then scores with :func:`adjusted_rand_index_labels`.
    """
    if length <= 0:
        return 1.0
    true_labels = convert_bkpts_to_labels(list(true_bkpts), length)
    pred_labels = convert_bkpts_to_labels(list(pred_bkpts), length)
    return adjusted_rand_index_labels(true_labels, pred_labels)


def v_measure_labels(
    true_labels: Sequence[int],
    pred_labels: Sequence[int],
    *,
    beta: float = 1.0,
) -> dict[str, float]:
    """V-measure / homogeneity / completeness between labelings."""
    from sklearn.metrics import (
        completeness_score,
        homogeneity_score,
        v_measure_score,
    )

    y_true = np.asarray(true_labels, dtype=int).ravel()
    y_pred = np.asarray(pred_labels, dtype=int).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Label length mismatch: true={y_true.shape[0]} pred={y_pred.shape[0]}"
        )
    if y_true.size == 0:
        return {"v_measure": 1.0, "homogeneity": 1.0, "completeness": 1.0}
    return {
        "v_measure": float(v_measure_score(y_true, y_pred, beta=beta)),
        "homogeneity": float(homogeneity_score(y_true, y_pred)),
        "completeness": float(completeness_score(y_true, y_pred)),
    }


def v_measure(
    true_bkpts: list[int],
    pred_bkpts: list[int],
    length: int,
    *,
    beta: float = 1.0,
) -> dict[str, float]:
    """V-measure between segmentations induced by breakpoints."""
    if length <= 0:
        return {"v_measure": 1.0, "homogeneity": 1.0, "completeness": 1.0}
    return v_measure_labels(
        convert_bkpts_to_labels(list(true_bkpts), length),
        convert_bkpts_to_labels(list(pred_bkpts), length),
        beta=beta,
    )


def normalized_mutual_info_labels(
    true_labels: Sequence[int],
    pred_labels: Sequence[int],
) -> float:
    """Normalised mutual information between two labelings."""
    from sklearn.metrics import normalized_mutual_info_score

    y_true = np.asarray(true_labels, dtype=int).ravel()
    y_pred = np.asarray(pred_labels, dtype=int).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Label length mismatch: true={y_true.shape[0]} pred={y_pred.shape[0]}"
        )
    if y_true.size == 0:
        return 1.0
    return float(normalized_mutual_info_score(y_true, y_pred))


def normalized_mutual_info(
    true_bkpts: list[int],
    pred_bkpts: list[int],
    length: int,
) -> float:
    """NMI between segmentations induced by breakpoints."""
    if length <= 0:
        return 1.0
    return normalized_mutual_info_labels(
        convert_bkpts_to_labels(list(true_bkpts), length),
        convert_bkpts_to_labels(list(pred_bkpts), length),
    )


def temporal_hamming_labels(
    true_labels: Sequence[int],
    pred_labels: Sequence[int],
) -> dict[str, float]:
    """Temporal Hamming / annotation error after optimal label permutation.

    Finds the permutation of predicted labels that maximises agreement with
    true labels (Hungarian matching on the contingency table), then reports
    Hamming distance and annotation error rate (fraction disagreeing).
    """
    from scipy.optimize import linear_sum_assignment

    y_true = np.asarray(true_labels, dtype=int).ravel()
    y_pred = np.asarray(pred_labels, dtype=int).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Label length mismatch: true={y_true.shape[0]} pred={y_pred.shape[0]}"
        )
    n = y_true.size
    if n == 0:
        return {"hamming": 0.0, "annotation_error": 0.0, "accuracy": 1.0}

    true_ids = sorted(set(y_true.tolist()))
    pred_ids = sorted(set(y_pred.tolist()))
    # Contingency: rows=true, cols=pred
    C = np.zeros((len(true_ids), len(pred_ids)), dtype=float)
    t_index = {lab: i for i, lab in enumerate(true_ids)}
    p_index = {lab: i for i, lab in enumerate(pred_ids)}
    for t, p in zip(y_true, y_pred):
        C[t_index[int(t)], p_index[int(p)]] += 1.0
    # Maximise agreement → minimise cost = -C (pad to square)
    n_r, n_c = C.shape
    m = max(n_r, n_c)
    cost = np.zeros((m, m), dtype=float)
    cost[:n_r, :n_c] = -C
    row_ind, col_ind = linear_sum_assignment(cost)
    remap = {}
    for r, c in zip(row_ind, col_ind):
        if r < n_r and c < n_c:
            remap[pred_ids[c]] = true_ids[r]
    aligned = np.array([remap.get(int(p), -999) for p in y_pred], dtype=int)
    disagree = int(np.sum(aligned != y_true))
    err = disagree / n
    return {
        "hamming": float(disagree),
        "annotation_error": float(err),
        "accuracy": float(1.0 - err),
    }


def temporal_hamming(
    true_bkpts: list[int],
    pred_bkpts: list[int],
    length: int,
) -> dict[str, float]:
    """Temporal Hamming between segmentations induced by breakpoints."""
    if length <= 0:
        return {"hamming": 0.0, "annotation_error": 0.0, "accuracy": 1.0}
    return temporal_hamming_labels(
        convert_bkpts_to_labels(list(true_bkpts), length),
        convert_bkpts_to_labels(list(pred_bkpts), length),
    )


def fdr_control_breakpoints(
    candidates: list[int],
    pvalues: Sequence[float],
    *,
    alpha: float = 0.05,
) -> dict:
    """Benjamini–Hochberg FDR control across candidate breakpoints.

    Parameters
    ----------
    candidates :
        Breakpoint indices (same order as ``pvalues``).
    pvalues :
        Per-candidate p-values from a breakpoint test.
    alpha :
        Target false-discovery rate.

    Returns
    -------
    dict
        ``kept`` / ``rejected`` breakpoint lists, BH ``threshold``, and
        boolean ``mask`` aligned with ``candidates``.
    """
    cands = list(candidates)
    pvals = np.asarray(pvalues, dtype=float).ravel()
    if len(cands) != len(pvals):
        raise ValueError(
            f"candidates/pvalues length mismatch: {len(cands)} vs {len(pvals)}"
        )
    m = len(pvals)
    if m == 0:
        return {
            "kept": [],
            "rejected": [],
            "threshold": 0.0,
            "mask": [],
            "alpha": float(alpha),
        }

    order = np.argsort(pvals)
    ranked = pvals[order]
    # Largest k with p_(k) <= (k/m) * alpha
    thresh = 0.0
    cutoff = -1
    for k in range(m):
        bh = ((k + 1) / m) * float(alpha)
        if ranked[k] <= bh:
            cutoff = k
            thresh = bh
    mask = np.zeros(m, dtype=bool)
    if cutoff >= 0:
        mask[order[: cutoff + 1]] = True
    kept = [cands[i] for i in range(m) if mask[i]]
    rejected = [cands[i] for i in range(m) if not mask[i]]
    return {
        "kept": kept,
        "rejected": rejected,
        "threshold": float(thresh),
        "mask": mask.tolist(),
        "alpha": float(alpha),
    }


match_breakpoints = _match
