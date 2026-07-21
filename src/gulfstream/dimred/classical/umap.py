"""Uniform manifold approximation and projection (optional dependency).

Requires ``umap-learn``. If import fails (common on Python 3.13), calling
UMAP helpers raises ImportError with an install hint.
"""
from __future__ import annotations

import polars as pl

from gulfstream.common import frames
from gulfstream.common.results import DimredResults
from gulfstream.dimred.classical import _common as common

try:
    import umap
except ImportError as _umap_exc:  # pragma: no cover
    umap = None
    _UMAP_IMPORT_ERROR = _umap_exc
else:
    _UMAP_IMPORT_ERROR = None


def _require_umap() -> None:
    if umap is None:
        raise ImportError(
            "umap-learn is not installed or failed to import. "
            "Install with `uv add umap-learn` (may be unavailable on Python 3.13). "
            f"Original error: {_UMAP_IMPORT_ERROR}"
        ) from _UMAP_IMPORT_ERROR


def _umap_dimred(
    df: pl.DataFrame,
    rank_selection_method: str,
    rank: int,
    umap_num_neighbors: float = 15.0,
    umap_min_dist: float = 0.1,
    umap_metric: str = "euclidean",
    random_state: int | None = None,
    **kwargs,
) -> DimredResults:
    """Uniform manifold approximation and projection."""
    _require_umap()
    n_feat = frames.n_features(df)
    if rank_selection_method != "user_specified":
        raise ValueError(f"Unknown rank_selection_method {rank_selection_method}.")
    if rank is None:
        raise ValueError("Must specify 'rank' if using 'user_specified' rank selection.")
    if rank > min(df.height, n_feat) or rank < 1:
        raise ValueError(
            "'rank' must be positive and not exceed number of rows and columns of raw data."
        )
    model = umap.UMAP(
        n_components=rank,
        n_neighbors=umap_num_neighbors,
        min_dist=umap_min_dist,
        metric=umap_metric,
        random_state=random_state,
    )
    x_dimred = model.fit_transform(frames.to_numpy(df))
    df_dimred = frames.with_same_dates(x_dimred, df)
    return DimredResults(
        df=df_dimred,
        dimred="umap",
        rank=rank,
        rank_selection_method=rank_selection_method,
        umap_num_neighbors=umap_num_neighbors,
        umap_min_dist=umap_min_dist,
        umap_metric=umap_metric,
        model=model,
    )


def _umap_generator(df: pl.DataFrame, params: dict):
    """Yield all requested UMAP dimension reductions of df."""
    rank_generators = [common._user_specified_rank_generator]
    if "umap" not in params["algo"]["dimred"]:
        return
    for num_neighbors in params["algo"]["umap_num_neighbors"]:
        for min_dist in params["algo"]["umap_min_dist"]:
            for metric in params["algo"]["umap_metric"]:
                for handler in rank_generators:
                    for d in handler(params):
                        d["umap_num_neighbors"] = num_neighbors
                        d["umap_min_dist"] = min_dist
                        d["umap_metric"] = metric
                        d["dimred"] = "umap"
                        yield _umap_dimred(df, **d)
