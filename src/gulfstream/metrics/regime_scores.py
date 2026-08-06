"""Feature × regime score matrices for Graph 2 targeted retrain.

Higher cell values mean a regime/feature is a better retrain candidate.
All scorers return ``np.ndarray`` of shape ``(n_features, n_regimes)``.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np
import polars as pl

from gulfstream.common import frames
from gulfstream.common.options import RetrainScoreMethod, values
from gulfstream.metrics.evaluation import _regime_edges, avg_features_loss

logger = logging.getLogger(__name__)

ScoreFn = Callable[..., np.ndarray]


def _mse_to_mean(df: pl.DataFrame, bkpts: list[int], **_kwargs: Any) -> np.ndarray:
    """Mean squared deviation from regime mean (classical L2 heatmap)."""
    return avg_features_loss(df, bkpts)


def _mad_to_median(df: pl.DataFrame, bkpts: list[int], **_kwargs: Any) -> np.ndarray:
    """Mean absolute deviation from regime median (robust L1 twin of L2)."""
    n = df.height
    edges = _regime_edges(bkpts, n)
    n_regimes = len(edges) - 1
    n_features = frames.n_features(df)
    out = np.zeros((n_features, n_regimes), dtype=float)
    values_arr = frames.to_numpy(df)
    for j in range(n_regimes):
        a, b = edges[j], edges[j + 1]
        if b <= a:
            continue
        seg = values_arr[a:b]
        med = np.median(seg, axis=0, keepdims=True)
        out[:, j] = np.abs(seg - med).mean(axis=0)
    return out


def _differenced(values: np.ndarray, order: int) -> np.ndarray:
    out = values
    for _ in range(max(0, order)):
        if out.shape[0] < 2:
            return out[:0]
        out = np.diff(out, axis=0)
    return out


def _mse_on_diff(df: pl.DataFrame, bkpts: list[int], **kwargs: Any) -> np.ndarray:
    """MSE-to-mean on first differences / returns (``score.diff_order``, default 1)."""
    order = int(kwargs.get("diff_order", 1))
    n = df.height
    edges = _regime_edges(bkpts, n)
    n_regimes = len(edges) - 1
    n_features = frames.n_features(df)
    out = np.zeros((n_features, n_regimes), dtype=float)
    values_arr = frames.to_numpy(df)
    for j in range(n_regimes):
        a, b = edges[j], edges[j + 1]
        if b <= a:
            continue
        seg = _differenced(values_arr[a:b], order)
        if seg.shape[0] == 0:
            continue
        mean = seg.mean(axis=0, keepdims=True)
        out[:, j] = ((seg - mean) ** 2).mean(axis=0)
    return out


def _factor_residual(df: pl.DataFrame, bkpts: list[int], **kwargs: Any) -> np.ndarray:
    """Within-regime PCA reconstruction residual variance per feature."""
    from sklearn.decomposition import PCA

    n_components = int(kwargs.get("n_components", 1))
    n = df.height
    edges = _regime_edges(bkpts, n)
    n_regimes = len(edges) - 1
    n_features = frames.n_features(df)
    out = np.zeros((n_features, n_regimes), dtype=float)
    values_arr = frames.to_numpy(df)
    for j in range(n_regimes):
        a, b = edges[j], edges[j + 1]
        if b <= a:
            continue
        seg = values_arr[a:b]
        t, p = seg.shape
        if t < 2 or p < 2:
            out[:, j] = seg.var(axis=0)
            continue
        n_comp = max(1, min(n_components, p - 1, t - 1))
        try:
            pca = PCA(n_components=n_comp)
            scores = pca.fit_transform(seg)
            recon = pca.inverse_transform(scores)
            out[:, j] = ((seg - recon) ** 2).mean(axis=0)
        except Exception:
            logger.debug("factor_residual PCA failed for regime %s; using variance.", j)
            out[:, j] = seg.var(axis=0)
    return out


def _hotelling_within(df: pl.DataFrame, bkpts: list[int], **_kwargs: Any) -> np.ndarray:
    """Per-feature squared standardized mean shift between regime halves."""
    n = df.height
    edges = _regime_edges(bkpts, n)
    n_regimes = len(edges) - 1
    n_features = frames.n_features(df)
    out = np.zeros((n_features, n_regimes), dtype=float)
    values_arr = frames.to_numpy(df)
    eps = 1e-8
    for j in range(n_regimes):
        a, b = edges[j], edges[j + 1]
        if b - a < 4:
            continue
        mid = (a + b) // 2
        left, right = values_arr[a:mid], values_arr[mid:b]
        if left.shape[0] < 2 or right.shape[0] < 2:
            continue
        n_l, n_r = left.shape[0], right.shape[0]
        mean_l = left.mean(axis=0)
        mean_r = right.mean(axis=0)
        var_l = left.var(axis=0, ddof=1)
        var_r = right.var(axis=0, ddof=1)
        pooled = ((n_l - 1) * var_l + (n_r - 1) * var_r) / max(n_l + n_r - 2, 1)
        out[:, j] = (mean_l - mean_r) ** 2 / (pooled + eps)
    return out


def _cusum_intensity(df: pl.DataFrame, bkpts: list[int], **kwargs: Any) -> np.ndarray:
    """Per-feature max |CUSUM| inside each regime (Crosier-style, first half = IC)."""
    k = float(kwargs.get("cusum_k", 0.5))
    n = df.height
    edges = _regime_edges(bkpts, n)
    n_regimes = len(edges) - 1
    n_features = frames.n_features(df)
    out = np.zeros((n_features, n_regimes), dtype=float)
    values_arr = frames.to_numpy(df)
    for j in range(n_regimes):
        a, b = edges[j], edges[j + 1]
        if b - a < 4:
            continue
        mid = (a + b) // 2
        ref = values_arr[a:mid]
        if ref.shape[0] < 2:
            continue
        mu = ref.mean(axis=0)
        cov = np.cov(ref, rowvar=False)
        if cov.ndim == 0:
            cov = np.array([[float(cov)]])
        try:
            eigvals, eigvecs = np.linalg.eigh(cov)
            eigvals = np.clip(eigvals, 1e-8, None)
            whitener = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        except np.linalg.LinAlgError:
            continue
        stream = values_arr[a:b]
        c = np.zeros(n_features, dtype=float)
        max_abs = np.zeros(n_features, dtype=float)
        for row in stream:
            z = whitener @ (row - mu)
            cand = c + z
            mag = float(np.linalg.norm(cand))
            if mag <= k:
                c = np.zeros_like(c)
            else:
                c = (1.0 - k / mag) * cand
                max_abs = np.maximum(max_abs, np.abs(c))
        out[:, j] = max_abs
    return out


def _downsample_rows(seg: np.ndarray, max_rows: int) -> np.ndarray:
    """Stride-downsample a regime to at most ``max_rows`` (preserves order)."""
    t = seg.shape[0]
    if max_rows <= 0 or t <= max_rows:
        return seg
    step = int(np.ceil(t / max_rows))
    return seg[::step]


def _candidate_splits(length: int, n_splits: int, min_side: int) -> list[int]:
    """Temporal split indices ``s`` with both sides at least ``min_side`` long."""
    lo = int(min_side)
    hi = int(length - min_side)
    if hi <= lo:
        return []
    n_splits = max(1, int(n_splits))
    if n_splits == 1:
        return [(lo + hi) // 2]
    return sorted({int(x) for x in np.linspace(lo, hi - 1, n_splits)})


def _best_univariate_split_scores(
    seg: np.ndarray,
    stat_fn,
    *,
    n_splits: int,
    min_side: int,
    max_rows: int,
) -> np.ndarray:
    """Max two-sample statistic over mid-window splits, per feature column."""
    seg = _downsample_rows(seg, max_rows)
    t, p = seg.shape
    scores = np.zeros(p, dtype=float)
    splits = _candidate_splits(t, n_splits, min_side)
    if not splits:
        return scores
    for feat in range(p):
        col = seg[:, feat : feat + 1]
        best = 0.0
        for s in splits:
            left, right = col[:s], col[s:]
            if left.shape[0] < min_side or right.shape[0] < min_side:
                continue
            try:
                val = float(stat_fn(left, right))
            except Exception:
                continue
            if np.isfinite(val):
                best = max(best, val)
        scores[feat] = best
    return scores


def _energy_split(df: pl.DataFrame, bkpts: list[int], **kwargs: Any) -> np.ndarray:
    """Best mid-window univariate energy distance inside each regime.

    Knobs (``retrain.score``): ``n_splits`` (default 5), ``min_side`` (10),
    ``max_rows`` (80) for smoke-friendly downsampling.
    """
    from gulfstream.detection.stat_tests import _energy_distance

    n_splits = int(kwargs.get("n_splits", 5))
    min_side = int(kwargs.get("min_side", 10))
    max_rows = int(kwargs.get("max_rows", 80))

    n = df.height
    edges = _regime_edges(bkpts, n)
    n_regimes = len(edges) - 1
    n_features = frames.n_features(df)
    out = np.zeros((n_features, n_regimes), dtype=float)
    values_arr = frames.to_numpy(df)
    need = 2 * min_side
    for j in range(n_regimes):
        a, b = edges[j], edges[j + 1]
        if b - a < need:
            continue
        out[:, j] = _best_univariate_split_scores(
            values_arr[a:b],
            lambda x, y: _energy_distance(x, y),
            n_splits=n_splits,
            min_side=min_side,
            max_rows=max_rows,
        )
    return out


def _mmd_split(df: pl.DataFrame, bkpts: list[int], **kwargs: Any) -> np.ndarray:
    """Best mid-window univariate RBF-MMD² inside each regime.

    Knobs (``retrain.score``): ``n_splits`` (5), ``min_side`` (10), ``max_rows`` (80),
    ``mmd_estimator`` in ``{linear, biased, unbiased}`` (default ``linear``),
    optional ``gamma`` (else median heuristic per split).
    """
    from gulfstream.detection.stat_tests import (
        _biased_mmd2,
        _linear_mmd2,
        _median_gamma,
        _unbiased_mmd2,
    )

    n_splits = int(kwargs.get("n_splits", 5))
    min_side = int(kwargs.get("min_side", 10))
    max_rows = int(kwargs.get("max_rows", 80))
    estimator = str(kwargs.get("mmd_estimator", "linear")).lower()
    gamma_fixed = kwargs.get("gamma")
    if estimator == "biased":
        mmd_fn = _biased_mmd2
    elif estimator == "unbiased":
        mmd_fn = _unbiased_mmd2
    else:
        mmd_fn = _linear_mmd2

    def _stat(x: np.ndarray, y: np.ndarray) -> float:
        g = float(gamma_fixed) if gamma_fixed is not None else _median_gamma(x, y)
        return float(mmd_fn(x, y, g))

    n = df.height
    edges = _regime_edges(bkpts, n)
    n_regimes = len(edges) - 1
    n_features = frames.n_features(df)
    out = np.zeros((n_features, n_regimes), dtype=float)
    values_arr = frames.to_numpy(df)
    need = 2 * min_side
    for j in range(n_regimes):
        a, b = edges[j], edges[j + 1]
        if b - a < need:
            continue
        out[:, j] = _best_univariate_split_scores(
            values_arr[a:b],
            _stat,
            n_splits=n_splits,
            min_side=min_side,
            max_rows=max_rows,
        )
    return out


_REGISTRY: dict[str, ScoreFn] = {
    RetrainScoreMethod.MSE_TO_MEAN: _mse_to_mean,
    RetrainScoreMethod.MAD_TO_MEDIAN: _mad_to_median,
    RetrainScoreMethod.MSE_ON_DIFF: _mse_on_diff,
    RetrainScoreMethod.FACTOR_RESIDUAL: _factor_residual,
    RetrainScoreMethod.HOTELLING_WITHIN: _hotelling_within,
    RetrainScoreMethod.CUSUM_INTENSITY: _cusum_intensity,
    RetrainScoreMethod.ENERGY_SPLIT: _energy_split,
    RetrainScoreMethod.MMD_SPLIT: _mmd_split,
}

# Plot / gallery metadata (title fragments for Graph 2 heatmaps).
SCORE_META: dict[str, dict[str, str]] = {
    RetrainScoreMethod.MSE_TO_MEAN: {
        "title": "Average daily L2 loss per feature in each regime (in bps)",
        "cbar": "average L2 dist to mean (measured in bps)",
        # Keep classical gallery filename so existing HTML galleries still match.
        "gallery_suffix": "avg_feature_L2",
        "log_name": "L2",
    },
    RetrainScoreMethod.MAD_TO_MEDIAN: {
        "title": "Mean abs. deviation from regime median per feature",
        "cbar": "MAD to median",
        "gallery_suffix": "avg_feature_MAD",
        "log_name": "MAD",
    },
    RetrainScoreMethod.MSE_ON_DIFF: {
        "title": "MSE of differenced series to regime mean per feature",
        "cbar": "MSE on diffs",
        "gallery_suffix": "avg_feature_MSE_diff",
        "log_name": "MSE-on-diff",
    },
    RetrainScoreMethod.FACTOR_RESIDUAL: {
        "title": "Within-regime PCA residual variance per feature",
        "cbar": "factor residual MSE",
        "gallery_suffix": "avg_feature_factor_resid",
        "log_name": "factor-residual",
    },
    RetrainScoreMethod.HOTELLING_WITHIN: {
        "title": "Within-regime Hotelling-style feature shift (half vs half)",
        "cbar": "std. mean-shift²",
        "gallery_suffix": "avg_feature_hotelling",
        "log_name": "hotelling-within",
    },
    RetrainScoreMethod.CUSUM_INTENSITY: {
        "title": "Within-regime CUSUM intensity per feature",
        "cbar": "max |CUSUM|",
        "gallery_suffix": "avg_feature_cusum",
        "log_name": "CUSUM",
    },
    RetrainScoreMethod.ENERGY_SPLIT: {
        "title": "Best mid-window energy distance per feature (within regime)",
        "cbar": "max energy split",
        "gallery_suffix": "avg_feature_energy_split",
        "log_name": "energy-split",
    },
    RetrainScoreMethod.MMD_SPLIT: {
        "title": "Best mid-window MMD² per feature (within regime)",
        "cbar": "max MMD split",
        "gallery_suffix": "avg_feature_mmd_split",
        "log_name": "MMD-split",
    },
}


def known_score_methods() -> list[str]:
    return values(RetrainScoreMethod)


def score_meta(method: str | RetrainScoreMethod) -> dict[str, str]:
    key = str(method)
    if key not in SCORE_META:
        return {
            "title": f"Retrain score ({key}) per feature × regime",
            "cbar": key,
            "gallery_suffix": f"avg_feature_{key}",
            "log_name": key,
        }
    return SCORE_META[key]


def score_feature_regime(
    df: pl.DataFrame,
    bkpts: list[int],
    method: str | RetrainScoreMethod = RetrainScoreMethod.MSE_TO_MEAN,
    **kwargs: Any,
) -> np.ndarray:
    """Compute a feature×regime score matrix (higher = worse / refine first)."""
    key = str(method)
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown retrain score_method {key!r}. "
            f"Choose one of: {', '.join(known_score_methods())}"
        )
    matrix = _REGISTRY[key](df, bkpts, **kwargs)
    if matrix.ndim != 2:
        raise ValueError(f"Score method {key!r} returned non-2D array.")
    expected_rows = frames.n_features(df)
    if matrix.shape[0] != expected_rows:
        raise ValueError(
            f"Score method {key!r} returned shape {matrix.shape}, "
            f"expected ({expected_rows}, n_regimes)."
        )
    return np.asarray(matrix, dtype=float)
