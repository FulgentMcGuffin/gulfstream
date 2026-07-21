"""t-distributed stochastic neighbor embedding."""
from __future__ import annotations

import inspect

import polars as pl
from sklearn.manifold import TSNE

from gulfstream.common import frames
from gulfstream.common.results import DimredResults
from gulfstream.dimred.classical import _common as common


def _tsne_dimred(
    df: pl.DataFrame,
    rank_selection_method: str,
    rank: int,
    tsne_perplexity: float = 30.0,
    tsne_n_iter: int = 300,
    random_state: int | None = None,
    **kwargs,
) -> DimredResults:
    """t-distributed stochastic neighbor embedding."""
    n_feat = frames.n_features(df)
    if rank_selection_method != "user_specified":
        raise ValueError(f"Unknown rank_selection_method {rank_selection_method}.")
    if rank is None:
        raise ValueError("Must specify 'rank' if using 'user_specified' rank selection.")
    if rank > min(df.height, n_feat) or rank < 1:
        raise ValueError(
            "'rank' must be positive and not exceed number of rows and columns of raw data."
        )
    tsne_kwargs: dict = {
        "n_components": rank,
        "perplexity": tsne_perplexity,
        "random_state": random_state,
    }
    # sklearn renamed n_iter → max_iter.
    sig = inspect.signature(TSNE.__init__)
    if "max_iter" in sig.parameters:
        tsne_kwargs["max_iter"] = tsne_n_iter
    else:
        tsne_kwargs["n_iter"] = tsne_n_iter
    model = TSNE(**tsne_kwargs)
    x_dimred = model.fit_transform(frames.to_numpy(df))
    df_dimred = frames.with_same_dates(x_dimred, df)
    return DimredResults(
        df=df_dimred,
        dimred="tsne",
        rank=rank,
        rank_selection_method=rank_selection_method,
        tsne_perplexity=tsne_perplexity,
        tsne_n_iter=tsne_n_iter,
        model=model,
    )


def _tsne_generator(df: pl.DataFrame, params: dict):
    """Yield all requested t-SNE dimension reductions of df."""
    rank_generators = [common._user_specified_rank_generator]
    if "tsne" not in params["algo"]["dimred"]:
        return
    for perplexity in params["algo"]["tsne_perplexity"]:
        for n_iter in params["algo"]["tsne_n_iter"]:
            for handler in rank_generators:
                for d in handler(params):
                    d["tsne_perplexity"] = perplexity
                    d["tsne_n_iter"] = n_iter
                    d["dimred"] = "tsne"
                    yield _tsne_dimred(df, **d)
