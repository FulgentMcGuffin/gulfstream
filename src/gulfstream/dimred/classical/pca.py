"""PCA dimensionality reduction."""
from __future__ import annotations

from typing import Literal

import numpy as np
import polars as pl
from sklearn.decomposition import PCA

from gulfstream.common import frames
from gulfstream.common.results import DimredResults
from gulfstream.dimred.classical import _common as common


def _pca_dimred(
    df: pl.DataFrame,
    *,
    rank_selection_method: Literal[
        "entropy", "explained_variance", "svht", "user_specified"
    ],
    rank: int | None = None,
    threshold: float | None = None,
    **kwargs,
) -> DimredResults:
    X = frames.to_numpy(df)
    n_feat = frames.n_features(df)
    if rank_selection_method == "user_specified":
        if rank is None:
            raise ValueError("Must specify 'rank' if using 'user_specified'.")
        p = PCA(n_components=rank, svd_solver="auto").fit(X)
        df_dimred = frames.with_same_dates(p.transform(X), df)
        return DimredResults(
            df=df_dimred,
            dimred="pca",
            rank_selection_method="user_specified",
            rank=rank,
            eigvals=p.explained_variance_,
            model=p,
        )
    if rank_selection_method == "explained_variance":
        if threshold is None:
            raise ValueError("Must specify 'threshold' for explained_variance.")
        p = PCA(n_components=threshold, svd_solver="auto").fit(X)
        df_dimred = frames.with_same_dates(p.transform(X), df)
        return DimredResults(
            df=df_dimred,
            dimred="pca",
            rank=frames.n_features(df_dimred),
            rank_selection_method="explained_variance",
            threshold=threshold,
            eigvals=p.explained_variance_,
            model=p,
        )
    handlers = {
        "entropy": common._rank_by_entropy,
        "svht": common._rank_by_svht,
    }
    handler = handlers.get(rank_selection_method)
    if handler is None:
        raise ValueError(f"Unknown rank_selection_method {rank_selection_method}.")
    p = PCA(svd_solver="auto").fit(X)
    eigvals = p.explained_variance_
    rank = handler(
        eigvals=eigvals,
        threshold=threshold,
        row=df.height,
        column=n_feat,
    )
    x_centered = X - p.mean_
    components = p.components_[:rank]
    df_dimred = frames.with_same_dates(np.dot(x_centered, components.T), df)
    return DimredResults(
        df=df_dimred,
        dimred="pca",
        rank=rank,
        rank_selection_method=rank_selection_method,
        threshold=threshold,
        eigvals=eigvals,
        model=p,
    )


def _pca_generator(df: pl.DataFrame, params: dict):
    if not common._need_dimred("pca", params):
        return
    for method in params["algo"].get("rank_selection_method", []):
        expensive = common.MOST_EXPENSIVE_HANDLERS.get(method)
        if not expensive:
            raise ValueError(f"Unknown rank_selection_method {method}.")
        handler_params = {"dimred": "pca", "rank_selection_method": method}
        ranks = params["algo"].get("rank", [])
        thresholds = params["algo"].get("threshold", [])
        most = expensive(params=handler_params, thresholds=thresholds, ranks=ranks)
        res = _pca_dimred(df, **most)
        slice_handler = common.SLICE_HANDLERS.get(method)
        if not slice_handler:
            raise ValueError(f"Unknown rank_selection_method {method}.")
        yield from slice_handler(res, params)
