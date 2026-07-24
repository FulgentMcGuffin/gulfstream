"""Classical detectors: MS-VAR, stochastic volatility, change-in-covariance."""
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


def _smooth_labels(labels: np.ndarray, min_dwell: int = 5) -> np.ndarray:
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
            elif left is not None and right is None:
                out[i:j] = left
            elif right is not None and left is None:
                out[i:j] = right
        i = j
    return out


# ---------------------------------------------------------------------------
# MS-VAR (regime-switching VAR via HMM on lagged multivariate states)
# ---------------------------------------------------------------------------


def ms_var_predict_regimes(
    df: pl.DataFrame | np.ndarray,
    regimes: int = 2,
    ms_var_lags: int = 1,
    ms_var_n_pc: int = 3,
    random_state: int | None = 42,
    **kwargs,
) -> AlgoResults:
    """Regime-switching VAR approximation.

    Projects to a few PCs, stacks lags into a state vector, and fits a Gaussian
    HMM — a reduced-form Markov-switching VAR (mean + covariance switch).
    """
    from hmmlearn.hmm import GaussianHMM

    X = frames.to_numpy(df) if isinstance(df, pl.DataFrame) else np.asarray(df, dtype=float)
    n, d = X.shape
    n_pc = max(1, min(int(ms_var_n_pc), d, max(n // 10, 1)))
    pcs = PCA(n_components=n_pc).fit_transform(X)
    p = max(1, int(ms_var_lags))
    # Build lagged design: [y_t, y_{t-1}, ..., y_{t-p}]
    rows = []
    for t in range(p, n):
        block = [pcs[t - i] for i in range(p + 1)]
        rows.append(np.concatenate(block))
    Z = np.asarray(rows, dtype=float)
    n_states = max(2, int(regimes))
    try:
        hmm = GaussianHMM(
            n_components=n_states,
            covariance_type="diag",
            n_iter=100,
            random_state=random_state,
        )
        hmm.fit(Z)
        raw = hmm.predict(Z)
    except Exception:
        logger.exception("MS-VAR HMM failed; falling back to k-means on lagged PCs.")
        raw = KMeans(n_clusters=n_states, random_state=random_state, n_init=10).fit_predict(Z)

    # Pad leading p observations with first predicted label
    labels = np.empty(n, dtype=int)
    labels[:p] = int(raw[0]) if len(raw) else 0
    labels[p:] = raw
    labels = _smooth_labels(labels, min_dwell=int(kwargs.get("ms_var_min_dwell", 5)))
    labels = utils._map_labels_to_ordered_integers(labels.tolist())
    bkpts = utils._convert_labels_to_bkpts(labels)
    return AlgoResults(bkpts=bkpts, labels=labels)


def ms_var_param_generator(params: dict):
    if "ms_var" not in params["algo"].get("regime_detection_algorithm", []):
        return
    for regimes in params["algo"].get("regimes", [2]):
        for lags in params["algo"].get("ms_var_lags", [1]):
            for n_pc in params["algo"].get("ms_var_n_pc", [3]):
                for rs in params["algo"].get("random_state", [42]):
                    yield {
                        "regime_detection_algorithm": "ms_var",
                        "regimes": regimes,
                        "ms_var_lags": lags,
                        "ms_var_n_pc": n_pc,
                        "random_state": rs,
                    }


# ---------------------------------------------------------------------------
# Stochastic volatility regimes
# ---------------------------------------------------------------------------


def stochastic_vol_predict_regimes(
    df: pl.DataFrame | np.ndarray,
    regimes: int = 2,
    sv_window: int = 20,
    random_state: int | None = 42,
    **kwargs,
) -> AlgoResults:
    """Label regimes from a simple stochastic-volatility proxy.

    Uses log realised / EWMA variance of the leading PC of returns, then
    quantile or k-means clustering into ``regimes`` vol states.
    """
    X = frames.to_numpy(df) if isinstance(df, pl.DataFrame) else np.asarray(df, dtype=float)
    if X.ndim == 1:
        levels = X.astype(float)
    else:
        levels = PCA(n_components=1).fit_transform(X).ravel()
    rets = np.diff(levels, prepend=levels[0])
    w = max(5, int(sv_window))
    # Realised variance proxy + EWMA blend
    rv = (
        pl.Series(rets**2)
        .rolling_mean(window_size=w, min_samples=3)
        .fill_null(strategy="forward")
        .fill_null(strategy="backward")
        .to_numpy()
    )
    alpha = 2.0 / (w + 1.0)
    ewma = np.empty_like(rv)
    ewma[0] = rv[0] if np.isfinite(rv[0]) else 1e-6
    for t in range(1, len(rv)):
        ewma[t] = alpha * (rets[t] ** 2) + (1 - alpha) * ewma[t - 1]
    proxy = 0.5 * (np.maximum(rv, 1e-12) + np.maximum(ewma, 1e-12))
    log_v = np.log(proxy)

    n_states = max(2, int(regimes))
    qs = np.linspace(0.0, 1.0, n_states + 1)[1:-1]
    thresholds = np.unique(np.quantile(log_v, qs))
    if len(thresholds) == 0 or np.allclose(log_v, log_v[0]):
        feats = np.column_stack([log_v, np.abs(rets)])
        labels = KMeans(n_clusters=n_states, random_state=random_state, n_init=10).fit_predict(
            feats
        )
    else:
        labels = np.digitize(log_v, thresholds)

    labels = _smooth_labels(labels, min_dwell=int(kwargs.get("sv_min_dwell", 10)))
    labels = utils._map_labels_to_ordered_integers(
        labels.tolist() if hasattr(labels, "tolist") else list(labels)
    )
    bkpts = utils._convert_labels_to_bkpts(labels)
    return AlgoResults(bkpts=bkpts, labels=labels)


def stochastic_vol_param_generator(params: dict):
    if "stochastic_vol" not in params["algo"].get("regime_detection_algorithm", []):
        return
    for regimes in params["algo"].get("regimes", [2]):
        for window in params["algo"].get("sv_window", [20]):
            for rs in params["algo"].get("random_state", [42]):
                yield {
                    "regime_detection_algorithm": "stochastic_vol",
                    "regimes": regimes,
                    "sv_window": window,
                    "random_state": rs,
                }


# ---------------------------------------------------------------------------
# Change-in-covariance / factor-loading breaks
# ---------------------------------------------------------------------------


def change_in_covariance_predict_regimes(
    df: pl.DataFrame | np.ndarray,
    regimes: int = 2,
    cic_window: int = 40,
    cic_n_pc: int = 3,
    random_state: int | None = 42,
    **kwargs,
) -> AlgoResults:
    """Detect regimes driven by covariance / factor-loading shifts.

    Rolling correlation/covariance is vectorised (vech); leading PCs of that
    path are clustered into ``regimes``. Captures loading breaks even when
    means are stable.
    """
    X = frames.to_numpy(df) if isinstance(df, pl.DataFrame) else np.asarray(df, dtype=float)
    n, d = X.shape
    w = max(10, int(cic_window))
    if n < w + 2:
        labels = np.zeros(n, dtype=int)
        return AlgoResults(bkpts=[], labels=labels.tolist())

    # Standardize columns
    sd = X.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    Z = (X - X.mean(axis=0)) / sd

    # Rolling correlation vech (upper triangle)
    idx = np.triu_indices(d, k=1) if d > 1 else (np.array([0]), np.array([0]))
    vech_dim = len(idx[0]) if d > 1 else 1
    path = np.zeros((n, vech_dim), dtype=float)
    for t in range(w - 1, n):
        W = Z[t - w + 1 : t + 1]
        if d == 1:
            path[t, 0] = float(np.var(W))
        else:
            C = np.corrcoef(W, rowvar=False)
            C = np.nan_to_num(C, nan=0.0)
            path[t] = C[idx]
    # Forward-fill the burn-in
    if w - 1 < n:
        path[: w - 1] = path[w - 1]

    n_pc = max(1, min(int(cic_n_pc), path.shape[1], n - 1))
    scores = PCA(n_components=n_pc).fit_transform(path)
    n_states = max(2, int(regimes))
    labels = KMeans(n_clusters=n_states, random_state=random_state, n_init=10).fit_predict(
        scores
    )
    labels = _smooth_labels(labels, min_dwell=int(kwargs.get("cic_min_dwell", 10)))
    labels = utils._map_labels_to_ordered_integers(labels.tolist())
    bkpts = utils._convert_labels_to_bkpts(labels)
    return AlgoResults(bkpts=bkpts, labels=labels)


def change_in_covariance_param_generator(params: dict):
    if "change_in_covariance" not in params["algo"].get(
        "regime_detection_algorithm", []
    ):
        return
    for regimes in params["algo"].get("regimes", [2]):
        for window in params["algo"].get("cic_window", [40]):
            for n_pc in params["algo"].get("cic_n_pc", [3]):
                for rs in params["algo"].get("random_state", [42]):
                    yield {
                        "regime_detection_algorithm": "change_in_covariance",
                        "regimes": regimes,
                        "cic_window": window,
                        "cic_n_pc": n_pc,
                        "random_state": rs,
                    }
