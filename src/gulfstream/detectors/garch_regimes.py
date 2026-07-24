"""GARCH volatility-regime detection.

Fits a univariate GARCH(1,1) on the leading PCA score of returns, then labels
regimes by clustering (or thresholding) conditional volatility.
"""
from __future__ import annotations

import logging

import numpy as np
import polars as pl
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from gulfstream.common import frames, utils
from gulfstream.common.results import AlgoResults
from gulfstream.detectors import common_validation as common

logger = logging.getLogger(__name__)


def _first_pc_returns(df: pl.DataFrame | np.ndarray) -> np.ndarray:
    X = frames.to_numpy(df) if isinstance(df, pl.DataFrame) else np.asarray(df, dtype=float)
    if X.ndim == 1:
        levels = X.astype(float)
    else:
        levels = PCA(n_components=1).fit_transform(X).ravel()
    rets = np.diff(levels, prepend=levels[0])
    # arch prefers percent-ish scale; rescale if tiny
    scale = np.std(rets)
    if scale < 1e-8:
        return rets
    if scale < 0.01:
        rets = rets * 100.0
    return rets


def garch_predict_regimes(
    df: pl.DataFrame | np.ndarray,
    regimes: int = 2,
    garch_p: int = 1,
    garch_q: int = 1,
    garch_mean: str = "Zero",
    random_state: int | None = 42,
    **kwargs,
) -> AlgoResults:
    """Label regimes from GARCH conditional volatility.

    Parameters
    ----------
    regimes :
        Number of volatility regimes (k-means on σ̂_t).
    garch_p / garch_q :
        GARCH(p, q) orders.
    """
    from arch import arch_model

    rets = _first_pc_returns(df)
    # Drop leading zero from prepended diff for fitting stability but keep length
    am = arch_model(
        rets,
        mean=garch_mean,
        vol="GARCH",
        p=int(garch_p),
        q=int(garch_q),
        rescale=True,
    )
    try:
        fitted = am.fit(disp="off", show_warning=False)
        sigma = np.asarray(fitted.conditional_volatility, dtype=float)
    except Exception:
        logger.exception("GARCH fit failed; falling back to rolling std regimes.")
        # EWMA-ish fallback
        sigma = (
            pl.Series(rets)
            .rolling_std(window_size=20, min_samples=5)
            .fill_null(strategy="forward")
            .fill_null(strategy="backward")
            .to_numpy()
        )

    sigma = np.nan_to_num(sigma, nan=float(np.nanmean(sigma) if np.isfinite(sigma).any() else 1.0))
    sigma = np.clip(sigma, 1e-12, None)
    log_sigma = np.log(sigma)

    if regimes <= 1:
        labels = np.zeros(len(sigma), dtype=int)
    else:
        # Quantile thresholds on log-volatility (stable when σ̂ is weakly separated)
        qs = np.linspace(0.0, 1.0, int(regimes) + 1)[1:-1]
        thresholds = np.unique(np.quantile(log_sigma, qs))
        if len(thresholds) == 0 or np.allclose(log_sigma, log_sigma[0]):
            # Fall back to k-means on (σ, |r|) so mean-shift series still segment
            feats = np.column_stack([log_sigma, np.abs(rets)])
            km = KMeans(n_clusters=int(regimes), random_state=random_state, n_init=10)
            labels = km.fit_predict(feats)
            order = np.argsort(km.cluster_centers_[:, 0])
            remap = {int(old): int(new) for new, old in enumerate(order)}
            labels = np.array([remap[int(x)] for x in labels], dtype=int)
        else:
            labels = np.digitize(log_sigma, thresholds)

    # Temporal smoothing: require a minimum dwell so vol regimes do not flicker
    labels = _smooth_labels(labels, min_dwell=int(kwargs.get("garch_min_dwell", 10)))

    labels = utils._map_labels_to_ordered_integers(labels)
    bkpts = utils._convert_labels_to_bkpts(labels)
    return AlgoResults(bkpts=bkpts, labels=labels)


def _smooth_labels(labels: np.ndarray, min_dwell: int = 10) -> np.ndarray:
    """Remove short flickers sandwiched between the same regime.

    A run shorter than ``min_dwell`` is rewritten only when both neighbors share
    a label (or one side is a boundary). This avoids cascading into a single
    global regime on rapidly switching series.
    """
    if min_dwell <= 1 or len(labels) == 0:
        return labels
    out = labels.copy()
    n = len(out)
    i = 0
    while i < n:
        j = i + 1
        while j < n and out[j] == out[i]:
            j += 1
        if j - i < min_dwell:
            left = out[i - 1] if i > 0 else None
            right = out[j] if j < n else None
            if left is not None and right is not None and left == right:
                out[i:j] = left
            elif left is None and right is not None:
                out[i:j] = right
            elif right is None and left is not None:
                out[i:j] = left
        i = j
    return out


def garch_param_generator(params: dict):
    if "garch" not in params["algo"].get("regime_detection_algorithm", []):
        return
    for regimes in params["algo"].get("regimes", [2]):
        for p in params["algo"].get("garch_p", [1]):
            for q in params["algo"].get("garch_q", [1]):
                for rs in params["algo"].get("random_state", [42]):
                    for dwell in params["algo"].get("garch_min_dwell", [10]):
                        yield {
                            "regime_detection_algorithm": "garch",
                            "regimes": regimes,
                            "garch_p": p,
                            "garch_q": q,
                            "random_state": rs,
                            "garch_min_dwell": dwell,
                        }


def garch_params_printout() -> dict:
    return {
        "garch_regimes": ["number of vol regimes"],
        "garch_garch_p": ["GARCH p"],
        "garch_garch_q": ["GARCH q"],
    }


def garch_input_validator(algo_params: dict) -> bool:
    return common._valid_regimes(algo_params)
