"""Shared helpers for dimensionality-reduction generators."""
from __future__ import annotations

import numpy as np
import polars as pl

from gulfstream.common import frames
from gulfstream.common.results import DimredResults


def _need_dimred(method: str, params: dict) -> bool:
    return method in params.get("algo", {}).get("dimred", [])


def _rank_by_explained_variance(*, eigvals, threshold, row, column) -> int:
    eigvals = np.asarray(eigvals, dtype=float)
    total = eigvals.sum()
    if total <= 0:
        return 1
    cum = np.cumsum(eigvals) / total
    rank = int(np.searchsorted(cum, threshold) + 1)
    return max(1, min(rank, len(eigvals)))


def _rank_by_entropy(*, eigvals, threshold, row, column) -> int:
    eigvals = np.asarray(eigvals, dtype=float)
    eigvals = np.clip(eigvals, 0, None)
    total = eigvals.sum()
    if total <= 0:
        return 1
    p = eigvals / total
    p = p[p > 0]
    entropy = float(-np.sum(p * np.log(p)))
    max_entropy = float(np.log(len(eigvals))) if len(eigvals) else 1.0
    # Lower relative entropy → fewer components; map threshold to rank fraction.
    frac = threshold if threshold is not None else 0.9
    rank = max(1, int(np.ceil(frac * len(eigvals) * (entropy / max_entropy if max_entropy else 1))))
    return min(rank, len(eigvals))


def _rank_by_svht(*, eigvals, threshold=None, row, column) -> int:
    """Singular value hard thresholding (Gavish–Donoho style heuristic)."""
    eigvals = np.asarray(eigvals, dtype=float)
    if len(eigvals) == 0:
        return 1
    beta = min(row, column) / max(row, column) if max(row, column) else 1.0
    omega = 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43
    median = float(np.median(np.sqrt(np.clip(eigvals, 0, None))))
    cutoff = (omega * median) ** 2
    rank = int(np.sum(eigvals > cutoff))
    return max(1, rank)


def _user_specified_rank_generator(params: dict):
    for rank in params["algo"].get("rank", []):
        yield {"rank_selection_method": "user_specified", "rank": rank}


def _explained_variance_param_generator(params: dict):
    for thr in params["algo"].get("threshold", []):
        yield {"rank_selection_method": "explained_variance", "threshold": thr}


def _entropy_param_generator(params: dict):
    for thr in params["algo"].get("threshold", []):
        yield {"rank_selection_method": "entropy", "threshold": thr}


def _svht_param_generator(params: dict):
    yield {"rank_selection_method": "svht"}


def _most_expensive_user_specified(params, thresholds, ranks):
    return {**params, "rank": max(ranks) if ranks else 1}


def _most_expensive_threshold(params, thresholds, ranks):
    return {**params, "threshold": max(thresholds) if thresholds else 0.9}


def _most_expensive_svht(params, thresholds, ranks):
    return dict(params)


MOST_EXPENSIVE_HANDLERS = {
    "user_specified": _most_expensive_user_specified,
    "explained_variance": _most_expensive_threshold,
    "entropy": _most_expensive_threshold,
    "svht": _most_expensive_svht,
}


def _slice_feature_cols(df: pl.DataFrame, rank: int) -> pl.DataFrame:
    cols = frames.feature_columns(df)[:rank]
    return frames.select_features(df, cols)


def _slice_user_specified(res: DimredResults, params: dict):
    ranks = sorted(params["algo"].get("rank", []))
    n_feat = frames.n_features(res.df)
    for rank in ranks:
        if rank > n_feat:
            continue
        sliced = DimredResults(
            df=_slice_feature_cols(res.df, rank),
            dimred=res.dimred,
            rank=rank,
            rank_selection_method="user_specified",
            eigvals=res.eigvals,
            kernel_params=res.kernel_params,
            fpca_smooth_window=res.fpca_smooth_window,
            model=res.model,
        )
        yield sliced


def _slice_threshold(res: DimredResults, params: dict):
    method = res.rank_selection_method
    thresholds = sorted(params["algo"].get("threshold", []))
    n_feat = frames.n_features(res.df)
    eigvals = res.eigvals if res.eigvals is not None else np.ones(n_feat)
    handlers = {
        "explained_variance": _rank_by_explained_variance,
        "entropy": _rank_by_entropy,
    }
    handler = handlers.get(method, _rank_by_explained_variance)
    for thr in thresholds:
        rank = handler(
            eigvals=eigvals,
            threshold=thr,
            row=res.df.height,
            column=n_feat,
        )
        rank = min(rank, n_feat)
        yield DimredResults(
            df=_slice_feature_cols(res.df, rank),
            dimred=res.dimred,
            rank=rank,
            rank_selection_method=method,
            threshold=thr,
            eigvals=eigvals,
            kernel_params=res.kernel_params,
            fpca_smooth_window=res.fpca_smooth_window,
            model=res.model,
        )


def _slice_svht(res: DimredResults, params: dict):
    yield res


SLICE_HANDLERS = {
    "user_specified": _slice_user_specified,
    "explained_variance": _slice_threshold,
    "entropy": _slice_threshold,
    "svht": _slice_svht,
}
