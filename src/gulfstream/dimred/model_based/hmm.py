"""HMM and MSAR model-based dimred backends."""
from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.decomposition import PCA

from gulfstream.common import frames
from gulfstream.common.options import DimredMethod
from gulfstream.common.results import DimredResults
from gulfstream.dimred.model_based._common import _as_frame, _require_regimes

try:
    import hmmlearn.hmm as hmm
except ImportError:  # pragma: no cover
    hmm = None


def _hmm_dimred(
    df: pl.DataFrame,
    regimes: int,
    hmm_emissions: str = "gaussian",
    hmm_n_iter: int = 100,
    **kwargs,
) -> DimredResults:
    """Posterior state probabilities as embedding columns."""
    if hmm is None:
        raise ImportError("hmmlearn is required for HMM dimred.")
    if hmm_emissions != "gaussian":
        raise ValueError(f"Unknown HMM emissions for dimred: {hmm_emissions}")
    n = _require_regimes(regimes)
    X = frames.to_numpy(df)
    model = hmm.GaussianHMM(
        n_components=n,
        n_iter=hmm_n_iter,
        covariance_type="full",
        min_covar=1e-2,
    )
    model.fit(X)
    probs = model.predict_proba(X)
    return DimredResults(
        df=_as_frame(probs, df, "hmm_"),
        dimred="hmm",
        rank=probs.shape[1],
        rank_selection_method="regimes",
        model=model,
    )


def _hmm_generator(df: pl.DataFrame, params: dict):
    if DimredMethod.HMM not in params.get("algo", {}).get("dimred", []):
        return
    for regimes in params["algo"]["regimes"]:
        for emissions in params["algo"].get("hmm_emissions", ["gaussian"]):
            for n_iter in params["algo"].get("hmm_n_iter", [100]):
                yield _hmm_dimred(
                    df,
                    regimes=regimes,
                    hmm_emissions=emissions,
                    hmm_n_iter=n_iter,
                )


def _msar_dimred(df: pl.DataFrame, regimes: int, **kwargs) -> DimredResults:
    """Smoothed MSAR regime probabilities (after PCA→1D) as embedding columns."""
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

    n = _require_regimes(regimes)
    X = frames.to_numpy(df)
    pca = PCA(n_components=1)
    y = np.asarray(pca.fit_transform(X).ravel(), dtype=float)
    model = MarkovRegression(y, k_regimes=n, trend="c", switching_variance=True)
    result = model.fit()
    probs = np.asarray(result.smoothed_marginal_probabilities, dtype=float)
    if probs.ndim == 1:
        probs = probs.reshape(-1, 1)
    return DimredResults(
        df=_as_frame(probs, df, "msar_"),
        dimred="msar",
        rank=probs.shape[1],
        rank_selection_method="regimes",
        model=result,
    )


def _msar_generator(df: pl.DataFrame, params: dict):
    if DimredMethod.MSAR not in params.get("algo", {}).get("dimred", []):
        return
    for regimes in params["algo"]["regimes"]:
        yield _msar_dimred(df, regimes=regimes)
