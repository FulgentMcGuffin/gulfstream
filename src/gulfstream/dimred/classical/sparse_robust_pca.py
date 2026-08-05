"""Sparse PCA and robust PCA dimred."""
from __future__ import annotations

import logging

import numpy as np
import polars as pl
from sklearn.decomposition import PCA, SparsePCA

from gulfstream.common import frames, utils
from gulfstream.common.options import DimredMethod
from gulfstream.common.results import DimredResults
from gulfstream.dimred.classical import _common as common

logger = logging.getLogger(__name__)


def _resolve_rank(df: pl.DataFrame, params: dict, *, default: int = 3) -> int:
    ranks = params.get("algo", {}).get("rank") or []
    n_feat = frames.n_features(df)
    if ranks:
        return max(1, min(int(max(ranks)), n_feat, df.height - 1))
    return max(1, min(default, n_feat, df.height - 1))


def _sparse_pca_dimred(
    df: pl.DataFrame,
    *,
    rank: int,
    sparse_pca_alpha: float = 1.0,
    random_state: int | None = 42,
    **kwargs,
) -> DimredResults:
    X = frames.to_numpy(df)
    n_comp = max(1, min(int(rank), X.shape[1], X.shape[0] - 1))
    model = SparsePCA(
        n_components=n_comp,
        alpha=float(sparse_pca_alpha),
        random_state=random_state,
        max_iter=200,
    )
    S = model.fit_transform(X)
    return DimredResults(
        df=frames.with_same_dates(S, df),
        dimred=DimredMethod.SPARSE_PCA,
        rank=n_comp,
        rank_selection_method="user_specified",
        model=model,
    )


def _soft_threshold(X: np.ndarray, tau: float) -> np.ndarray:
    return np.sign(X) * np.maximum(np.abs(X) - tau, 0.0)


def _robust_pca_godec(
    X: np.ndarray,
    *,
    rank: int,
    max_iter: int = 50,
    tol: float = 1e-5,
) -> tuple[np.ndarray, object]:
    """Light RPCA / GoDec-style low-rank + sparse decomposition.

    Alternates truncated SVD (low-rank) and soft-thresholding (sparse noise).
    Returns the low-rank factors projected to ``rank`` PCA scores of L.
    """
    L = X.copy()
    S = np.zeros_like(X)
    mu = np.median(np.abs(X - np.median(X))) + 1e-8
    tau = 1.5 * mu
    prev = np.inf
    for _ in range(max_iter):
        # Low-rank via truncated SVD of X - S
        M = X - S
        try:
            U, s, Vt = np.linalg.svd(M, full_matrices=False)
        except np.linalg.LinAlgError:
            break
        k = max(1, min(rank, len(s)))
        L = (U[:, :k] * s[:k]) @ Vt[:k, :]
        S = _soft_threshold(X - L, tau)
        err = float(np.linalg.norm(X - L - S) / (np.linalg.norm(X) + 1e-12))
        if abs(prev - err) < tol:
            break
        prev = err
    # Scores = leading PCs of the recovered low-rank matrix
    p = PCA(n_components=rank).fit(L)
    scores = p.transform(L)
    return scores, {"pca": p, "L": L, "S": S, "err": prev}


def _robust_pca_dimred(
    df: pl.DataFrame,
    *,
    rank: int,
    rpca_max_iter: int = 50,
    **kwargs,
) -> DimredResults:
    X = frames.to_numpy(df)
    n_comp = max(1, min(int(rank), X.shape[1], X.shape[0] - 1))
    # Standardize for scale stability
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    Z = (X - mu) / sd
    try:
        scores, model = _robust_pca_godec(Z, rank=n_comp, max_iter=int(rpca_max_iter))
    except Exception as exc:
        logger.warning("Robust PCA failed (%s); falling back to PCA.", exc)
        p = PCA(n_components=n_comp).fit(Z)
        scores, model = p.transform(Z), p
    return DimredResults(
        df=frames.with_same_dates(scores, df),
        dimred=DimredMethod.ROBUST_PCA,
        rank=scores.shape[1],
        rank_selection_method="user_specified",
        model=model,
    )


def _sparse_pca_generator(df: pl.DataFrame, params: dict):
    if not common._need_dimred(DimredMethod.SPARSE_PCA, params):
        return
    ranks = params["algo"].get("rank") or [_resolve_rank(df, params)]
    alphas = params["algo"].get("sparse_pca_alpha", [1.0])
    for rank in ranks:
        for alpha in alphas:
            for rs in utils.algo_grid(params, "random_state", [42]):
                yield _sparse_pca_dimred(
                    df,
                    rank=int(rank),
                    sparse_pca_alpha=float(alpha),
                    random_state=rs,
                )


def _robust_pca_generator(df: pl.DataFrame, params: dict):
    if not common._need_dimred(DimredMethod.ROBUST_PCA, params):
        return
    ranks = params["algo"].get("rank") or [_resolve_rank(df, params)]
    maxiters = params["algo"].get("rpca_max_iter", [50])
    for rank in ranks:
        for mi in maxiters:
            yield _robust_pca_dimred(df, rank=int(rank), rpca_max_iter=int(mi))
