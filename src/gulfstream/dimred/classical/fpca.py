"""Functional PCA dimred on discrete curves.

Treats each row as a curve over the feature (grid) domain: lightly smooth
along that axis, then perform PCA on the smoothed panel — a discrete FPCA
approximation suitable for tenor / maturity-style features.
"""
from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.decomposition import PCA

from gulfstream.common import frames
from gulfstream.common.options import DimredMethod
from gulfstream.common.results import DimredResults
from gulfstream.dimred.classical import _common as common


def _smooth_along_features(X: np.ndarray, window: int) -> np.ndarray:
    """Moving-average smooth along axis=1 (feature / maturity domain)."""
    w = int(window)
    if w <= 1 or X.shape[1] < 3:
        return X
    w = min(w, X.shape[1] if X.shape[1] % 2 == 1 else X.shape[1] - 1)
    if w < 3:
        return X
    if w % 2 == 0:
        w -= 1
    pad = w // 2
    kernel = np.ones(w, dtype=float) / w
    out = np.empty_like(X, dtype=float)
    for i in range(X.shape[0]):
        padded = np.pad(X[i], (pad, pad), mode="edge")
        out[i] = np.convolve(padded, kernel, mode="valid")
    return out


def _fpca_dimred(
    df: pl.DataFrame,
    *,
    rank: int | None = None,
    rank_selection_method: str = "explained_variance",
    threshold: float | None = 0.9,
    fpca_smooth_window: int = 3,
    **kwargs,
) -> DimredResults:
    X = frames.to_numpy(df)
    Xs = _smooth_along_features(X, fpca_smooth_window)
    n_feat = Xs.shape[1]

    if rank_selection_method == "user_specified":
        if rank is None:
            raise ValueError("Must specify 'rank' for FPCA user_specified.")
        n_comp = max(1, min(int(rank), n_feat, Xs.shape[0] - 1))
        p = PCA(n_components=n_comp).fit(Xs)
        scores = p.transform(Xs)
        return DimredResults(
            df=frames.with_same_dates(scores, df),
            dimred=DimredMethod.FPCA,
            rank=n_comp,
            rank_selection_method="user_specified",
            eigvals=p.explained_variance_,
            fpca_smooth_window=int(fpca_smooth_window),
            model={"pca": p, "smooth_window": fpca_smooth_window},
        )

    # Fit full PCA then slice by threshold / entropy / svht
    p = PCA().fit(Xs)
    eigvals = p.explained_variance_
    if rank_selection_method == "explained_variance":
        thr = 0.9 if threshold is None else float(threshold)
        rank_sel = common._rank_by_explained_variance(
            eigvals=eigvals, threshold=thr, row=df.height, column=n_feat
        )
    elif rank_selection_method == "entropy":
        rank_sel = common._rank_by_entropy(
            eigvals=eigvals, threshold=threshold, row=df.height, column=n_feat
        )
    elif rank_selection_method == "svht":
        rank_sel = common._rank_by_svht(
            eigvals=eigvals, row=df.height, column=n_feat
        )
    else:
        raise ValueError(f"Unknown rank_selection_method {rank_selection_method}")

    rank_sel = max(1, min(rank_sel, n_feat, Xs.shape[0] - 1))
    scores = (Xs - p.mean_) @ p.components_[:rank_sel].T
    return DimredResults(
        df=frames.with_same_dates(scores, df),
        dimred=DimredMethod.FPCA,
        rank=rank_sel,
        rank_selection_method=rank_selection_method,
        threshold=threshold,
        eigvals=eigvals,
        fpca_smooth_window=int(fpca_smooth_window),
        model={"pca": p, "smooth_window": fpca_smooth_window},
    )


def _fpca_generator(df: pl.DataFrame, params: dict):
    if not common._need_dimred(DimredMethod.FPCA, params):
        return
    methods = params["algo"].get("rank_selection_method", ["explained_variance"])
    smooths = params["algo"].get("fpca_smooth_window", [3])
    for method in methods:
        expensive = common.MOST_EXPENSIVE_HANDLERS.get(method)
        if not expensive:
            raise ValueError(f"Unknown rank_selection_method {method}.")
        handler_params = {
            "dimred": DimredMethod.FPCA,
            "rank_selection_method": method,
        }
        ranks = params["algo"].get("rank", [])
        thresholds = params["algo"].get("threshold", [])
        most = expensive(params=handler_params, thresholds=thresholds, ranks=ranks)
        for sw in smooths:
            res = _fpca_dimred(df, fpca_smooth_window=int(sw), **most)
            slice_handler = common.SLICE_HANDLERS.get(method)
            if not slice_handler:
                raise ValueError(f"Unknown rank_selection_method {method}.")
            # Re-tag dimred after slicing
            for sliced in slice_handler(res, params):
                sliced.dimred = DimredMethod.FPCA
                yield sliced
