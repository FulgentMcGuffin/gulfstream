"""Dynamic-factor model dimred (statsmodels DynamicFactor, PCA fallback)."""
from __future__ import annotations

import logging

import numpy as np
import polars as pl
from sklearn.decomposition import PCA

from gulfstream.common import frames
from gulfstream.common.options import DimredMethod
from gulfstream.common.results import DimredResults
from gulfstream.dimred.classical import _common as common

logger = logging.getLogger(__name__)


def _resolve_rank(df: pl.DataFrame, params: dict, *, default: int = 2) -> int:
    ranks = params.get("algo", {}).get("rank") or []
    n_feat = frames.n_features(df)
    if ranks:
        return max(1, min(int(max(ranks)), n_feat, max(df.height // 5, 1)))
    return max(1, min(default, n_feat, max(df.height // 5, 1)))


def _pca_factor_fallback(X: np.ndarray, k: int) -> tuple[np.ndarray, object]:
    model = PCA(n_components=k)
    F = model.fit_transform(X)
    return F, model


def _dynamic_factor_dimred(
    df: pl.DataFrame,
    *,
    rank: int,
    factor_order: int = 1,
    df_maxiter: int = 50,
    **kwargs,
) -> DimredResults:
    X = frames.to_numpy(df)
    k = max(1, min(int(rank), X.shape[1], max(X.shape[0] // 5, 1)))
    model: object
    try:
        from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor

        # Standardize for numerical stability
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd = np.where(sd < 1e-8, 1.0, sd)
        Z = (X - mu) / sd
        mod = DynamicFactor(
            Z,
            k_factors=k,
            factor_order=int(factor_order),
            error_cov_type="diagonal",
        )
        res = mod.fit(disp=False, maxiter=int(df_maxiter), method="lbfgs")
        # factors.smoothed: (k_factors, nobs)
        F = np.asarray(res.factors.smoothed).T
        if F.ndim == 1:
            F = F.reshape(-1, 1)
        if F.shape[0] != df.height:
            # Rare length mismatch — fall back
            raise RuntimeError(
                f"DynamicFactor length mismatch: {F.shape[0]} vs {df.height}"
            )
        model = res
    except Exception as exc:
        logger.warning("DynamicFactor failed (%s); using PCA factors.", exc)
        F, model = _pca_factor_fallback(X, k)

    return DimredResults(
        df=frames.with_same_dates(F, df),
        dimred=DimredMethod.DYNAMIC_FACTOR,
        rank=F.shape[1],
        rank_selection_method="user_specified",
        factor_order=int(factor_order),
        model=model,
    )


def _dynamic_factor_generator(df: pl.DataFrame, params: dict):
    if not common._need_dimred(DimredMethod.DYNAMIC_FACTOR, params):
        return
    ranks = params["algo"].get("rank") or [_resolve_rank(df, params)]
    for rank in ranks:
        for order in params["algo"].get("factor_order", [1]):
            for maxiter in params["algo"].get("df_maxiter", [50]):
                yield _dynamic_factor_dimred(
                    df,
                    rank=int(rank),
                    factor_order=int(order),
                    df_maxiter=int(maxiter),
                )
