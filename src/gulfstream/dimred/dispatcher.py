"""Dimensionality reduction dispatcher."""
from __future__ import annotations

from typing import Iterator

import polars as pl

from gulfstream.dimred.classical import dmd as dmd_mod
from gulfstream.common import frames
from gulfstream.dimred.classical import kpca as kpca_mod
from gulfstream.dimred import model_based as model_dimred
from gulfstream.dimred.classical import pca as pca_mod
from gulfstream.dimred.classical import tsne as tsne_mod
from gulfstream.dimred.classical import umap as umap_mod
from gulfstream.common.results import DimredResults


def _raw_dimred(df: pl.DataFrame, **kwargs) -> DimredResults:
    return DimredResults(df=df.clone(), dimred="raw", rank=frames.n_features(df))


def _raw_generator(df: pl.DataFrame, params: dict):
    if "raw" in params.get("algo", {}).get("dimred", []):
        yield _raw_dimred(df)


_GENERATORS = {
    "pca": pca_mod._pca_generator,
    "kpca": kpca_mod._kpca_generator,
    "raw": _raw_generator,
    "dmd": dmd_mod._dmd_generator,
    "tsne": tsne_mod._tsne_generator,
    "umap": umap_mod._umap_generator,
    **model_dimred.GENERATORS,
}


def _dimred_generator(df: pl.DataFrame, params: dict) -> Iterator[DimredResults]:
    for method in params.get("algo", {}).get("dimred", []):
        gen = _GENERATORS.get(method)
        if gen is None:
            raise ValueError(
                f"Unknown dimred method {method}. "
                "Supported: pca/kpca/raw/dmd/tsne/umap/"
                "bayesian_gmm/hmm/kmeans/hdbscan/optics/msar/wasserstein/ruptures/tft."
            )
        yield from gen(df, params)


def _get_dimred_param_dict(res: DimredResults) -> dict:
    out = {
        "dimred": res.dimred,
        "rank": res.rank,
        "rank_selection_method": res.rank_selection_method,
        "threshold": res.threshold,
    }
    if res.kernel_params is not None:
        out["kernel_params"] = res.kernel_params
    if res.dmd_stride is not None:
        out["dmd_stride"] = res.dmd_stride
    if res.dmd_rolling_window is not None:
        out["dmd_rolling_window"] = res.dmd_rolling_window
    if res.tsne_perplexity is not None:
        out["tsne_perplexity"] = res.tsne_perplexity
    if res.tsne_n_iter is not None:
        out["tsne_n_iter"] = res.tsne_n_iter
    if res.umap_num_neighbors is not None:
        out["umap_num_neighbors"] = res.umap_num_neighbors
    if res.umap_min_dist is not None:
        out["umap_min_dist"] = res.umap_min_dist
    if res.umap_metric is not None:
        out["umap_metric"] = res.umap_metric
    return {k: v for k, v in out.items() if v is not None}
