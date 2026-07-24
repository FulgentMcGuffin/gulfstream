"""Statistical tests for accepting/rejecting candidate breakpoints.

Registered tests (``test.choice``):

- ``mmd_no_ts`` / ``mmd_ts`` / ``mmd_perm`` — biased squared MMD with an RBF
  kernel and a permutation p-value (``mmd_perm`` uses more permutations;
  ``mmd_ts`` additionally prepares first-difference maps upstream).
- ``mmd_unbiased`` — unbiased MMD² estimator (U-statistic), permutation p-value.
- ``mmd_linear`` — Gretton linear-time MMD² (O(n) pairs), permutation p-value.
- ``energy_distance`` — Székely–Rizzo energy distance, permutation p-value.
- ``hotelling_t2`` — two-sample Hotelling's T² on window means (F / chi² p-value).
- ``multivariate_cusum`` — Crosier-style MCUSUM of Mahalanobis residuals vs the
  left-window mean; permutation p-value.
- ``ks_pca`` — Kolmogorov–Smirnov on leading PCA scores of the pooled windows.
"""
from __future__ import annotations

import itertools
import logging
from typing import Callable, Iterator

import numpy as np
import polars as pl
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances, rbf_kernel

from gulfstream.common import frames
from gulfstream.common.options import StatTest

logger = logging.getLogger(__name__)

MMD_TESTS = (
    StatTest.MMD_NO_TS,
    StatTest.MMD_TS,
    StatTest.MMD_PERM,
    StatTest.MMD_UNBIASED,
    StatTest.MMD_LINEAR,
)

DEFAULT_STATS = {choice: (0.0, 1.0) for choice in StatTest}

STAT_COLUMN_NAMES = {
    StatTest.MMD_NO_TS: ["mmd_stat", "mmd_pvalue"],
    StatTest.MMD_TS: ["mmd_stat", "mmd_pvalue"],
    StatTest.MMD_PERM: ["mmd_stat", "mmd_pvalue"],
    StatTest.MMD_UNBIASED: ["mmd_stat", "mmd_pvalue"],
    StatTest.MMD_LINEAR: ["mmd_stat", "mmd_pvalue"],
    StatTest.ENERGY_DISTANCE: ["energy_stat", "energy_pvalue"],
    StatTest.HOTELLING_T2: ["hotelling_stat", "hotelling_pvalue"],
    StatTest.MULTIVARIATE_CUSUM: ["cusum_stat", "cusum_pvalue"],
    StatTest.KS_PCA: ["ks_stat", "ks_pvalue"],
}

STAT_DTYPES = {choice: [float, float] for choice in StatTest}

PARAM_FORMATTERS = {
    choice: (lambda: {"lag": "lag", "window": "window", "sample_size": "sample_size"})
    for choice in StatTest
}


def test_param_combos(params: dict) -> Iterator[dict]:
    """Yield single-combo test parameter dicts."""
    test = params["test"]
    choices = test["choice"]
    lag_opts = test.get("lag", [{}])
    window_opts = test.get("window", [{}])
    sample_opts = test.get("sample_size", [{}])
    sig_opts = test.get("significance_level", [0.05])

    for choice, lag, window, sample_size, sig in itertools.product(
        choices, lag_opts, window_opts, sample_opts, sig_opts
    ):
        yield {
            "choice": choice,
            "lag": lag,
            "window": window,
            "sample_size": sample_size,
            "significance_level": sig,
        }


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def _biased_mmd2(x: np.ndarray, y: np.ndarray, gamma: float) -> float:
    """Biased squared MMD with RBF kernel."""
    kxx = rbf_kernel(x, x, gamma=gamma)
    kyy = rbf_kernel(y, y, gamma=gamma)
    kxy = rbf_kernel(x, y, gamma=gamma)
    return float(kxx.mean() + kyy.mean() - 2.0 * kxy.mean())


def _unbiased_mmd2(x: np.ndarray, y: np.ndarray, gamma: float) -> float:
    """Unbiased squared MMD (U-statistic): diagonal terms excluded."""
    n, m = len(x), len(y)
    if n < 2 or m < 2:
        return 0.0
    kxx = rbf_kernel(x, x, gamma=gamma)
    kyy = rbf_kernel(y, y, gamma=gamma)
    kxy = rbf_kernel(x, y, gamma=gamma)
    sum_xx = (kxx.sum() - np.trace(kxx)) / (n * (n - 1))
    sum_yy = (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
    return float(sum_xx + sum_yy - 2.0 * kxy.mean())


def _linear_mmd2(x: np.ndarray, y: np.ndarray, gamma: float) -> float:
    """Gretton linear-time MMD² (O(n) kernel evaluations via paired blocks).

    Uses equal-length paired samples; truncates to ``2 * floor(min(n, m) / 2)``.
    """
    n = min(len(x), len(y))
    n_pairs = n // 2
    if n_pairs < 1:
        return 0.0
    x = x[: 2 * n_pairs]
    y = y[: 2 * n_pairs]
    # h((x,y),(x',y')) = k(x,x') + k(y,y') - k(x,y') - k(x',y)
    total = 0.0
    for i in range(n_pairs):
        xa, xb = x[2 * i : 2 * i + 1], x[2 * i + 1 : 2 * i + 2]
        ya, yb = y[2 * i : 2 * i + 1], y[2 * i + 1 : 2 * i + 2]
        total += float(
            rbf_kernel(xa, xb, gamma=gamma)[0, 0]
            + rbf_kernel(ya, yb, gamma=gamma)[0, 0]
            - rbf_kernel(xa, yb, gamma=gamma)[0, 0]
            - rbf_kernel(xb, ya, gamma=gamma)[0, 0]
        )
    return float(total / n_pairs)


def _energy_distance(x: np.ndarray, y: np.ndarray, gamma: float | None = None) -> float:
    """Székely–Rizzo energy distance between samples (gamma unused)."""
    dxy = pairwise_distances(x, y).mean()
    dxx = pairwise_distances(x, x).mean()
    dyy = pairwise_distances(y, y).mean()
    return float(2.0 * dxy - dxx - dyy)


def _median_gamma(x: np.ndarray, y: np.ndarray) -> float:
    combo = np.vstack([x, y])
    med = float(np.median(pairwise_distances(combo)))
    return 1.0 / (2 * max(med, 1e-8) ** 2)


def _as_2d(a: pl.DataFrame | np.ndarray) -> np.ndarray:
    if isinstance(a, pl.DataFrame):
        arr = frames.to_numpy(a)
    else:
        arr = np.asarray(a, dtype=float)
    return arr.reshape(-1, 1) if arr.ndim == 1 else arr


def _pooled_cov(x: np.ndarray, y: np.ndarray, *, ridge: float = 1e-6) -> np.ndarray:
    """Pooled covariance with a small ridge for invertibility."""
    n, m = len(x), len(y)
    p = x.shape[1]
    if n + m <= p + 1:
        return np.eye(p) * max(ridge, 1.0)
    sx = np.cov(x, rowvar=False) if n > 1 else np.zeros((p, p))
    sy = np.cov(y, rowvar=False) if m > 1 else np.zeros((p, p))
    if np.ndim(sx) == 0:
        sx = np.array([[float(sx)]])
        sy = np.array([[float(sy)]])
    pooled = ((n - 1) * sx + (m - 1) * sy) / max(n + m - 2, 1)
    return pooled + ridge * np.eye(p)


def _maybe_pca_reduce(x: np.ndarray, y: np.ndarray, *, max_dim: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Project to a few PCs when dimension is large relative to sample size."""
    n, m = len(x), len(y)
    p = x.shape[1]
    target = min(max_dim, p, n + m - 2, max(n - 1, 1), max(m - 1, 1))
    if p <= target or target < 1:
        return x, y
    pca = PCA(n_components=target)
    pooled = np.vstack([x, y])
    scores = pca.fit_transform(pooled)
    return scores[:n], scores[n:]


def _hotelling_t2_stat(x: np.ndarray, y: np.ndarray, gamma: float | None = None) -> float:
    """Two-sample Hotelling T² statistic (gamma unused)."""
    x, y = _maybe_pca_reduce(x, y)
    n, m = len(x), len(y)
    if n < 2 or m < 2:
        return 0.0
    diff = x.mean(axis=0) - y.mean(axis=0)
    cov = _pooled_cov(x, y)
    try:
        inv = np.linalg.pinv(cov)
    except np.linalg.LinAlgError:
        return 0.0
    t2 = float((n * m) / (n + m) * diff @ inv @ diff)
    return max(t2, 0.0)


def _hotelling_t2_pvalue(t2: float, n: int, m: int, p: int) -> float:
    """Convert Hotelling T² to an F-based upper-tail p-value."""
    df1 = p
    df2 = n + m - p - 1
    if df2 <= 0 or t2 <= 0:
        return 1.0
    f_stat = (df2 / (df1 * max(n + m - 2, 1))) * t2
    return float(stats.f.sf(f_stat, df1, df2))


def _mcusum_stat(x: np.ndarray, y: np.ndarray, gamma: float | None = None) -> float:
    """Crosier-style multivariate CUSUM max magnitude.

    Treats ``x`` as in-control (reference mean/cov) and accumulates whitened
    residuals over ``[x; y]``. Larger max ||C|| suggests a mean shift.
    """
    if len(x) < 2 or len(y) < 1:
        return 0.0
    mu = x.mean(axis=0)
    cov = _pooled_cov(x, x)
    try:
        # Whitening: Σ^{-1/2} via eigh of covariance.
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.clip(eigvals, 1e-8, None)
        whitener = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    except np.linalg.LinAlgError:
        return 0.0
    stream = np.vstack([x, y])
    c = np.zeros(x.shape[1], dtype=float)
    max_mag = 0.0
    k = 0.5
    for row in stream:
        z = whitener @ (row - mu)
        cand = c + z
        mag = float(np.linalg.norm(cand))
        if mag <= k:
            c = np.zeros_like(c)
        else:
            c = (1.0 - k / mag) * cand
            max_mag = max(max_mag, float(np.linalg.norm(c)))
    return max_mag


def _ks_pca_stat(x: np.ndarray, y: np.ndarray, gamma: float | None = None) -> float:
    """Max two-sample KS statistic across leading PCA scores of the pool."""
    return _ks_pca_test_arrays(x, y)[0]


def _ks_pca_test_arrays(
    x: np.ndarray, y: np.ndarray, *, n_components: int | None = None
) -> tuple[float, float]:
    """Return (max KS statistic, min p-value) over PCA score coordinates."""
    n, m = len(x), len(y)
    p = x.shape[1]
    if n < 2 or m < 2:
        return 0.0, 1.0
    pooled = np.vstack([x, y])
    n_comp = n_components or min(3, p, pooled.shape[0] - 1)
    n_comp = max(1, n_comp)
    pca = PCA(n_components=n_comp)
    scores = pca.fit_transform(pooled)
    sx, sy = scores[:n], scores[n:]
    best_stat = 0.0
    best_p = 1.0
    for j in range(scores.shape[1]):
        res = stats.ks_2samp(sx[:, j], sy[:, j], alternative="two-sided", method="auto")
        if res.statistic > best_stat:
            best_stat = float(res.statistic)
        best_p = min(best_p, float(res.pvalue))
    return best_stat, best_p


def _permutation_test(
    x: np.ndarray,
    y: np.ndarray,
    statistic: Callable[[np.ndarray, np.ndarray, float | None], float],
    *,
    gamma: float | None,
    n_permutations: int,
    significance_level: float,
) -> tuple[float, float, bool]:
    """Shared permutation harness: (statistic, p_value, accept)."""
    stat = statistic(x, y, gamma)
    pooled = np.vstack([x, y])
    n_x = len(x)
    count = 0
    rng = np.random.default_rng(0)
    for _ in range(n_permutations):
        rng.shuffle(pooled)
        stat_p = statistic(pooled[:n_x], pooled[n_x:], gamma)
        if stat_p >= stat:
            count += 1
    p_value = (count + 1) / (n_permutations + 1)
    return stat, p_value, p_value < significance_level


def run_mmd_test(
    left: pl.DataFrame | np.ndarray,
    right: pl.DataFrame | np.ndarray,
    *,
    gamma: float | None = None,
    n_permutations: int = 50,
    significance_level: float = 0.05,
    unbiased: bool = False,
    linear: bool = False,
) -> tuple[float, float, bool]:
    """Return (statistic, p_value, accept_breakpoint).

    Accept means distributions differ enough to treat the candidate as a breakpoint.
    """
    x, y = _as_2d(left), _as_2d(right)
    if gamma is None:
        gamma = _median_gamma(x, y)
    if linear:
        statistic = _linear_mmd2
    elif unbiased:
        statistic = _unbiased_mmd2
    else:
        statistic = _biased_mmd2
    return _permutation_test(
        x,
        y,
        statistic,
        gamma=gamma,
        n_permutations=n_permutations,
        significance_level=significance_level,
    )


def run_energy_distance_test(
    left: pl.DataFrame | np.ndarray,
    right: pl.DataFrame | np.ndarray,
    *,
    n_permutations: int = 50,
    significance_level: float = 0.05,
) -> tuple[float, float, bool]:
    """Energy-distance two-sample test with a permutation p-value."""
    x, y = _as_2d(left), _as_2d(right)
    return _permutation_test(
        x,
        y,
        _energy_distance,
        gamma=None,
        n_permutations=n_permutations,
        significance_level=significance_level,
    )


def run_hotelling_t2_test(
    left: pl.DataFrame | np.ndarray,
    right: pl.DataFrame | np.ndarray,
    *,
    significance_level: float = 0.05,
) -> tuple[float, float, bool]:
    """Two-sample Hotelling T² with an F-distribution p-value."""
    x, y = _maybe_pca_reduce(_as_2d(left), _as_2d(right))
    t2 = _hotelling_t2_stat(x, y)
    # Recompute on already-reduced arrays without a second PCA pass.
    n, m, p = len(x), len(y), x.shape[1]
    if n < 2 or m < 2:
        return 0.0, 1.0, False
    diff = x.mean(axis=0) - y.mean(axis=0)
    cov = _pooled_cov(x, y)
    try:
        inv = np.linalg.pinv(cov)
    except np.linalg.LinAlgError:
        return 0.0, 1.0, False
    t2 = float((n * m) / (n + m) * diff @ inv @ diff)
    p_value = _hotelling_t2_pvalue(max(t2, 0.0), n, m, p)
    return max(t2, 0.0), p_value, p_value < significance_level


def run_multivariate_cusum_test(
    left: pl.DataFrame | np.ndarray,
    right: pl.DataFrame | np.ndarray,
    *,
    n_permutations: int = 50,
    significance_level: float = 0.05,
) -> tuple[float, float, bool]:
    """Multivariate CUSUM max-magnitude test with a permutation p-value."""
    x, y = _maybe_pca_reduce(_as_2d(left), _as_2d(right))
    return _permutation_test(
        x,
        y,
        _mcusum_stat,
        gamma=None,
        n_permutations=n_permutations,
        significance_level=significance_level,
    )


def run_ks_pca_test(
    left: pl.DataFrame | np.ndarray,
    right: pl.DataFrame | np.ndarray,
    *,
    significance_level: float = 0.05,
    n_components: int | None = None,
) -> tuple[float, float, bool]:
    """KS test on leading PCA scores of the pooled left/right windows."""
    x, y = _as_2d(left), _as_2d(right)
    stat, p_value = _ks_pca_test_arrays(x, y, n_components=n_components)
    return stat, p_value, p_value < significance_level


def test_breakpoint(
    df: pl.DataFrame,
    bkpt: int,
    test_params: dict,
    *,
    mapped_df: pl.DataFrame | None = None,
) -> tuple[float, float, bool]:
    """Test a candidate breakpoint using the configured statistical test."""
    choice = test_params["choice"]
    lag = int(test_params.get("lag", {}).get("lag", 0) or 0)
    window = int(test_params.get("window", {}).get("window", 40) or 40)
    sample_size = int(
        test_params.get("sample_size", {}).get("num_samples", window) or window
    )
    sig = float(test_params.get("significance_level", 0.05) or 0.05)

    series = mapped_df if mapped_df is not None else df
    left_end = bkpt - lag
    right_start = bkpt + lag
    left_start = max(0, left_end - window)
    right_end = min(series.height, right_start + window)
    if left_end - left_start < 5 or right_end - right_start < 5:
        return 0.0, 1.0, False

    left = frames.slice_rows(series, left_start, left_end)
    right = frames.slice_rows(series, right_start, right_end)
    # Subsample if requested.
    if sample_size < left.height:
        left = frames.slice_rows(left, left.height - sample_size, left.height)
    if sample_size < right.height:
        right = frames.slice_rows(right, 0, sample_size)

    if choice in (StatTest.MMD_NO_TS, StatTest.MMD_TS, StatTest.MMD_PERM):
        n_perm = 100 if choice == StatTest.MMD_PERM else 50
        return run_mmd_test(left, right, n_permutations=n_perm, significance_level=sig)
    if choice == StatTest.MMD_UNBIASED:
        return run_mmd_test(
            left, right, n_permutations=50, significance_level=sig, unbiased=True
        )
    if choice == StatTest.MMD_LINEAR:
        return run_mmd_test(
            left, right, n_permutations=50, significance_level=sig, linear=True
        )
    if choice == StatTest.ENERGY_DISTANCE:
        return run_energy_distance_test(
            left, right, n_permutations=50, significance_level=sig
        )
    if choice == StatTest.HOTELLING_T2:
        return run_hotelling_t2_test(left, right, significance_level=sig)
    if choice == StatTest.MULTIVARIATE_CUSUM:
        return run_multivariate_cusum_test(
            left, right, n_permutations=50, significance_level=sig
        )
    if choice == StatTest.KS_PCA:
        return run_ks_pca_test(left, right, significance_level=sig)
    raise ValueError(f"Unknown test choice {choice}")
