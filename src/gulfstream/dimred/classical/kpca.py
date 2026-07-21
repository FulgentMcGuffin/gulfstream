"""Kernel PCA dimensionality reduction."""
from __future__ import annotations

from typing import Literal

import numpy as np
import polars as pl
from sklearn.decomposition import KernelPCA

from gulfstream.common import frames
from gulfstream.common import utils
from gulfstream.common.results import DimredResults
from gulfstream.dimred.classical import _common as common


def _kpca_dimred(
    df: pl.DataFrame,
    *,
    kpca_kernel_params: dict | None = None,
    rank_selection_method: Literal[
        "entropy", "explained_variance", "svht", "user_specified"
    ],
    rank: int | None = None,
    threshold: float | None = None,
    **kwargs,
) -> DimredResults:
    kpca_kernel_params = kpca_kernel_params or {"kernel": "rbf", "gamma": "median"}
    required = {
        "linear": [],
        "poly": ["degree", "coef0", "gamma"],
        "rbf": ["gamma"],
        "sigmoid": ["coef0", "gamma"],
        "cosine": [],
    }
    model = kpca_kernel_params.get("kernel")
    if not model:
        raise KeyError("'kernel' must be provided for kernel PCA.")
    if model not in required:
        raise ValueError(f"Unknown kernel {model}.")
    missing = [p for p in required[model] if p not in kpca_kernel_params]
    if missing:
        raise ValueError(f"Missing required parameters for {model}: {missing}.")

    processed = dict(kpca_kernel_params)
    if model == "rbf":
        processed["gamma"] = utils._calculate_bandwidth(df, processed["gamma"])
    processed = {k: v for k, v in processed.items() if k in required[model] or k == "kernel"}
    kernel = processed.pop("kernel")
    X = frames.to_numpy(df)
    n_feat = frames.n_features(df)

    if rank_selection_method == "user_specified":
        if rank is None:
            raise ValueError("Must specify 'rank' for user_specified.")
        p = KernelPCA(
            n_components=rank,
            kernel=kernel,
            **{k: v for k, v in processed.items() if k != "kernel"},
        ).fit(X)
        df_dimred = frames.with_same_dates(p.transform(X), df)
        processed["kernel"] = kernel
        return DimredResults(
            df=df_dimred,
            dimred="kpca",
            rank_selection_method="user_specified",
            rank=frames.n_features(df_dimred),
            kernel_params=processed,
            eigvals=getattr(p, "lambdas_", None),
            model=p,
        )

    handlers = {
        "entropy": common._rank_by_entropy,
        "explained_variance": common._rank_by_explained_variance,
        "svht": common._rank_by_svht,
    }
    handler = handlers.get(rank_selection_method)
    if handler is None:
        raise ValueError(f"Unknown rank_selection_method {rank_selection_method}.")

    fit_kwargs = {k: v for k, v in processed.items() if k != "kernel"}
    # Fit full then slice by rank.
    max_comp = min(df.height, n_feat)
    p = KernelPCA(n_components=max_comp, kernel=kernel, **fit_kwargs).fit(X)
    eigvals = getattr(p, "lambdas_", None)
    if eigvals is None:
        eigvals = np.ones(max_comp)
    rank = handler(
        eigvals=eigvals,
        threshold=threshold,
        row=df.height,
        column=n_feat,
    )
    transformed = p.transform(X)[:, :rank]
    df_dimred = frames.with_same_dates(transformed, df)
    processed["kernel"] = kernel
    return DimredResults(
        df=df_dimred,
        dimred="kpca",
        rank=rank,
        rank_selection_method=rank_selection_method,
        threshold=threshold,
        eigvals=eigvals,
        kernel_params=processed,
        model=p,
    )


def _kpca_generator(df: pl.DataFrame, params: dict):
    if not common._need_dimred("kpca", params):
        return
    kernel_list = params["algo"].get("kpca_kernel_params", [{"kernel": "rbf", "gamma": "median"}])
    for kparams in kernel_list:
        for method in params["algo"].get("rank_selection_method", []):
            expensive = common.MOST_EXPENSIVE_HANDLERS.get(method)
            if not expensive:
                raise ValueError(f"Unknown rank_selection_method {method}.")
            handler_params = {
                "dimred": "kpca",
                "rank_selection_method": method,
                "kpca_kernel_params": kparams,
            }
            ranks = params["algo"].get("rank", [])
            thresholds = params["algo"].get("threshold", [])
            most = expensive(params=handler_params, thresholds=thresholds, ranks=ranks)
            res = _kpca_dimred(df, **most)
            slice_handler = common.SLICE_HANDLERS.get(method)
            if not slice_handler:
                raise ValueError(f"Unknown rank_selection_method {method}.")
            yield from slice_handler(res, params)
