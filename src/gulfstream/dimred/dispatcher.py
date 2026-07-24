"""Dimensionality reduction dispatcher."""
from __future__ import annotations

from typing import Iterator

import polars as pl

from gulfstream.dimred.classical import dmd as dmd_mod
from gulfstream.common import frames
from gulfstream.common.options import DimredMethod, values as option_values
from gulfstream.dimred.classical import autoencoder as autoencoder_mod
from gulfstream.dimred.classical import diebold_li as diebold_li_mod
from gulfstream.dimred.classical import dynamic_factor as dynamic_factor_mod
from gulfstream.dimred.classical import fpca as fpca_mod
from gulfstream.dimred.classical import ica as ica_mod
from gulfstream.dimred.classical import kpca as kpca_mod
from gulfstream.dimred import model_based as model_dimred
from gulfstream.dimred.classical import nelson_siegel as nelson_siegel_mod
from gulfstream.dimred.classical import pca as pca_mod
from gulfstream.dimred.classical import sparse_robust_pca as sparse_robust_pca_mod
from gulfstream.dimred.classical import tsne as tsne_mod
from gulfstream.dimred.classical import umap as umap_mod
from gulfstream.common.results import DimredResults


def _raw_dimred(df: pl.DataFrame, **kwargs) -> DimredResults:
    return DimredResults(df=df.clone(), dimred=DimredMethod.RAW, rank=frames.n_features(df))


def _raw_generator(df: pl.DataFrame, params: dict):
    if DimredMethod.RAW in params.get("algo", {}).get("dimred", []):
        yield _raw_dimred(df)


GENERATORS = {
    DimredMethod.PCA: pca_mod._pca_generator,
    DimredMethod.KPCA: kpca_mod._kpca_generator,
    DimredMethod.RAW: _raw_generator,
    DimredMethod.DMD: dmd_mod._dmd_generator,
    DimredMethod.TSNE: tsne_mod._tsne_generator,
    DimredMethod.UMAP: umap_mod._umap_generator,
    DimredMethod.ICA: ica_mod._ica_generator,
    DimredMethod.FPCA: fpca_mod._fpca_generator,
    DimredMethod.NELSON_SIEGEL: nelson_siegel_mod._nelson_siegel_generator,
    DimredMethod.DYNAMIC_FACTOR: dynamic_factor_mod._dynamic_factor_generator,
    DimredMethod.DIEBOLD_LI: diebold_li_mod._diebold_li_generator,
    DimredMethod.SPARSE_PCA: sparse_robust_pca_mod._sparse_pca_generator,
    DimredMethod.ROBUST_PCA: sparse_robust_pca_mod._robust_pca_generator,
    DimredMethod.AUTOENCODER: autoencoder_mod._autoencoder_generator,
    **model_dimred.GENERATORS,
}


def dimred_generator(df: pl.DataFrame, params: dict) -> Iterator[DimredResults]:
    for method in params.get("algo", {}).get("dimred", []):
        gen = GENERATORS.get(method)
        if gen is None:
            raise ValueError(
                f"Unknown dimred method {method}. "
                f"Supported: {'/'.join(option_values(DimredMethod))}."
            )
        yield from gen(df, params)


def get_dimred_param_dict(res: DimredResults) -> dict:
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
    if res.ica_max_iter is not None:
        out["ica_max_iter"] = res.ica_max_iter
    if res.ns_lambda is not None:
        out["ns_lambda"] = res.ns_lambda
    if res.factor_order is not None:
        out["factor_order"] = res.factor_order
    if res.fpca_smooth_window is not None:
        out["fpca_smooth_window"] = res.fpca_smooth_window
    return {k: v for k, v in out.items() if v is not None}


# Temporary aliases removed in the rename sweep.
