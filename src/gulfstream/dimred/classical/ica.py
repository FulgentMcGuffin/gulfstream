"""Independent Component Analysis (FastICA) dimred."""
from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.decomposition import FastICA

from gulfstream.common import frames
from gulfstream.common.options import DimredMethod
from gulfstream.common.results import DimredResults
from gulfstream.dimred.classical import _common as common


def _resolve_rank(df: pl.DataFrame, params: dict, *, default: int = 3) -> int:
    ranks = params.get("algo", {}).get("rank") or []
    n_feat = frames.n_features(df)
    if ranks:
        return max(1, min(int(max(ranks)), n_feat, df.height - 1))
    return max(1, min(default, n_feat, df.height - 1))


def _ica_dimred(
    df: pl.DataFrame,
    *,
    rank: int,
    random_state: int | None = 42,
    ica_max_iter: int = 200,
    ica_tol: float = 1e-4,
    **kwargs,
) -> DimredResults:
    X = frames.to_numpy(df)
    n_comp = max(1, min(int(rank), X.shape[1], X.shape[0] - 1))
    model = FastICA(
        n_components=n_comp,
        random_state=random_state,
        max_iter=int(ica_max_iter),
        tol=float(ica_tol),
        whiten="unit-variance",
    )
    S = model.fit_transform(X)
    return DimredResults(
        df=frames.with_same_dates(S, df),
        dimred=DimredMethod.ICA,
        rank=n_comp,
        rank_selection_method="user_specified",
        ica_max_iter=int(ica_max_iter),
        model=model,
    )


def _ica_generator(df: pl.DataFrame, params: dict):
    if not common._need_dimred(DimredMethod.ICA, params):
        return
    ranks = params["algo"].get("rank") or [_resolve_rank(df, params)]
    for rank in ranks:
        for rs in params["algo"].get("random_state", [42]):
            for max_iter in params["algo"].get("ica_max_iter", [200]):
                yield _ica_dimred(
                    df,
                    rank=int(rank),
                    random_state=rs,
                    ica_max_iter=int(max_iter),
                )
