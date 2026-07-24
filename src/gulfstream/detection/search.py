"""Candidate breakpoint search: Wild Binary Segmentation and BOCPD.

These supplement ruptures PELT / Binseg / BottomUp inside
:func:`gulfstream.detection.algorithm.candidate_breakpoints`.
"""
from __future__ import annotations

import logging

import numpy as np
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


def _as_2d(signal: np.ndarray) -> np.ndarray:
    X = np.asarray(signal, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return X


def _project(X: np.ndarray, max_dim: int = 8) -> np.ndarray:
    """Light PCA projection for numerical stability in high-d feature maps."""
    n, d = X.shape
    if d <= max_dim or n <= max_dim + 1:
        return X
    k = min(max_dim, n - 1, d)
    return PCA(n_components=k).fit_transform(X)


def _cusum_best(
    X: np.ndarray,
    start: int,
    end: int,
    *,
    min_size: int,
) -> tuple[int | None, float]:
    """Best L2 mean-change CUSUM location on ``X[start:end]``.

    Returns ``(index_in_full_series, strength)`` or ``(None, -inf)``.
    """
    length = end - start
    if length < 2 * min_size:
        return None, float("-inf")
    seg = X[start:end]
    # Prefix sums for O(1) segment means
    csum = np.vstack([np.zeros((1, seg.shape[1])), np.cumsum(seg, axis=0)])
    best_b: int | None = None
    best_s = float("-inf")
    for b in range(min_size, length - min_size + 1):
        n_l, n_r = b, length - b
        mean_l = csum[b] / n_l
        mean_r = (csum[length] - csum[b]) / n_r
        # Scaled Euclidean jump (multivariate CUSUM)
        strength = float(
            np.sqrt(n_l * n_r / length) * np.linalg.norm(mean_l - mean_r)
        )
        if strength > best_s:
            best_s = strength
            best_b = start + b
    return best_b, best_s


def _draw_intervals(
    n: int,
    n_intervals: int,
    *,
    min_size: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    """Uniform random intervals of length at least ``2 * min_size``."""
    min_len = 2 * min_size
    if n < min_len:
        return [(0, n)] if n >= 2 else []
    intervals: list[tuple[int, int]] = [(0, n)]  # always include full span
    for _ in range(max(0, n_intervals - 1)):
        # Rejection sample until valid length
        for _try in range(32):
            a, b = rng.integers(0, n, size=2)
            s, e = (int(a), int(b)) if a < b else (int(b), int(a))
            if e - s >= min_len:
                intervals.append((s, e))
                break
    return intervals


def wild_binary_segmentation(
    signal: np.ndarray,
    *,
    n_intervals: int = 200,
    n_bkps: int | None = None,
    min_size: int = 10,
    threshold: float | None = None,
    random_state: int | None = 42,
) -> list[int]:
    """Wild Binary Segmentation (Fryzlewicz) candidate breakpoints.

    Draws random intervals, maximises a multivariate L2-CUSUM on each, then
    recursively accepts the strongest location above ``threshold`` (default:
    ``sqrt(2 log n) * mad_scale``). Caps at ``n_bkps`` candidates when set.
    """
    X = _project(_as_2d(signal))
    n = X.shape[0]
    if n < 2 * min_size:
        return []
    if n_bkps is None:
        n_bkps = max(1, n // 50)
    rng = np.random.default_rng(random_state)
    intervals = _draw_intervals(n, int(n_intervals), min_size=min_size, rng=rng)

    # Precompute best (location, strength) per interval
    pool: list[tuple[int, float, int, int]] = []
    for s, e in intervals:
        b, strength = _cusum_best(X, s, e, min_size=min_size)
        if b is not None:
            pool.append((b, strength, s, e))

    if not pool:
        return []

    strengths = np.asarray([p[1] for p in pool], dtype=float)
    med = float(np.median(strengths))
    mad = float(np.median(np.abs(strengths - med))) + 1e-12
    if threshold is None:
        # Scale-aware default akin to Fryzlewicz-style zeta_T
        threshold = float(np.sqrt(2.0 * np.log(max(n, 3))) * (mad * 1.4826 + 1e-8))
        # Keep at least some candidates even on flat series
        threshold = min(threshold, float(np.quantile(strengths, 0.75)))

    found: list[int] = []

    def recurse(start: int, end: int) -> None:
        if len(found) >= n_bkps or end - start < 2 * min_size:
            return
        best: tuple[int, float] | None = None
        for b, strength, s, e in pool:
            if s >= start and e <= end and start + min_size <= b <= end - min_size:
                if best is None or strength > best[1]:
                    best = (b, strength)
        # Also evaluate the full current interval (integrated WBS)
        b_full, s_full = _cusum_best(X, start, end, min_size=min_size)
        if b_full is not None and (best is None or s_full > best[1]):
            best = (b_full, s_full)
        if best is None or best[1] < threshold:
            return
        b = int(best[0])
        found.append(b)
        recurse(start, b)
        recurse(b, end)

    recurse(0, n)
    return sorted(set(b for b in found if 0 < b < n - 1))


def _log_student_predictive(
    x: np.ndarray,
    mean: np.ndarray,
    *,
    kappa: float,
    alpha: float,
    beta: np.ndarray,
) -> float:
    """Log predictive density under independent Normal-Inverse-Gamma margins."""
    # Predictive is Student-t; use Gaussian approx with predictive variance
    # Var = beta*(kappa+1)/(alpha*kappa) per dim
    var = beta * (kappa + 1.0) / (max(alpha, 1e-8) * max(kappa, 1e-8))
    var = np.maximum(var, 1e-8)
    resid = x - mean
    return float(-0.5 * np.sum(np.log(2 * np.pi * var) + resid**2 / var))


def bayesian_online_changepoint_detection(
    signal: np.ndarray,
    *,
    hazard: float = 1.0 / 100.0,
    threshold: float = 0.5,
    max_run: int | None = None,
    min_size: int = 10,
) -> list[int]:
    """Bayesian Online Changepoint Detection (Adams & MacKay) candidates.

    Constant hazard rate; independent Normal-Inverse-Gamma observations after
    a light PCA projection. Returns indices where the posterior mass on run
    length 0 exceeds ``threshold``, with a ``min_size`` refractory period.
    """
    X = _project(_as_2d(signal))
    n, d = X.shape
    if n < 2 * min_size:
        return []

    H = float(np.clip(hazard, 1e-6, 1.0 - 1e-6))
    # Truncate run-length support — full O(n²) is unnecessary for candidate search.
    R = min(n, int(max_run) if max_run is not None else min(n, 250))

    # Priors
    mu0 = np.zeros(d)
    kappa0 = 1.0
    alpha0 = 1.0
    beta0 = np.ones(d)

    # Run-length posterior (log space for growth probs we store unnormalized then norm)
    # Keep sufficient stats for each possible run length ending at t-1.
    # For simplicity use O(n * R) with truncated R.
    log_P = np.full(R + 1, -np.inf)
    log_P[0] = 0.0  # r=0 at t=0

    # Stats indexed by run length: mean, kappa, alpha, beta
    means = np.tile(mu0, (R + 1, 1))
    kappas = np.full(R + 1, kappa0)
    alphas = np.full(R + 1, alpha0)
    betas = np.tile(beta0, (R + 1, 1))

    candidates: list[int] = []
    last_cp = -min_size

    for t in range(n):
        x = X[t]
        # Predictive under each previous run length
        log_pred = np.full(R + 1, -np.inf)
        active = np.isfinite(log_P) & (log_P > -1e100)
        for r in np.where(active)[0]:
            log_pred[r] = _log_student_predictive(
                x, means[r], kappa=kappas[r], alpha=alphas[r], beta=betas[r]
            )

        # Growth: r -> r+1
        log_growth = np.full(R + 1, -np.inf)
        for r in range(R):
            if not np.isfinite(log_P[r]):
                continue
            log_growth[r + 1] = log_P[r] + np.log(1.0 - H) + log_pred[r]

        # Changepoint: any r -> 0, predictive under prior
        log_cp_pred = _log_student_predictive(
            x, mu0, kappa=kappa0, alpha=alpha0, beta=beta0
        )
        log_cp = -np.inf
        for r in range(R + 1):
            if not np.isfinite(log_P[r]):
                continue
            val = log_P[r] + np.log(H) + log_cp_pred
            log_cp = np.logaddexp(log_cp, val)

        new_log_P = np.full(R + 1, -np.inf)
        new_log_P[0] = log_cp
        new_log_P[1:] = log_growth[1:]
        # Normalize
        m = np.max(new_log_P[np.isfinite(new_log_P)]) if np.any(np.isfinite(new_log_P)) else 0.0
        probs = np.exp(new_log_P - m)
        probs[~np.isfinite(new_log_P)] = 0.0
        z = probs.sum()
        if z <= 0:
            probs = np.zeros_like(probs)
            probs[0] = 1.0
            z = 1.0
        probs /= z
        new_log_P = np.log(np.maximum(probs, 1e-300))

        cp_prob = float(probs[0])
        if (
            t >= min_size
            and t < n - min_size
            and cp_prob >= threshold
            and t - last_cp >= min_size
        ):
            candidates.append(t)
            last_cp = t

        # Update sufficient statistics for surviving run lengths
        new_means = np.tile(mu0, (R + 1, 1))
        new_kappas = np.full(R + 1, kappa0)
        new_alphas = np.full(R + 1, alpha0)
        new_betas = np.tile(beta0, (R + 1, 1))
        # r=0: restart from prior + x
        new_kappas[0] = kappa0 + 1.0
        new_alphas[0] = alpha0 + 0.5
        new_means[0] = (kappa0 * mu0 + x) / new_kappas[0]
        new_betas[0] = beta0 + 0.5 * (
            kappa0 * (x - mu0) ** 2 / new_kappas[0]
        )
        # r>=1 grew from r-1
        for r in range(1, R + 1):
            prev = r - 1
            if probs[r] <= 0 and not np.isfinite(log_growth[r]):
                continue
            k_prev = kappas[prev]
            m_prev = means[prev]
            new_kappas[r] = k_prev + 1.0
            new_alphas[r] = alphas[prev] + 0.5
            new_means[r] = (k_prev * m_prev + x) / new_kappas[r]
            new_betas[r] = betas[prev] + 0.5 * (
                k_prev * (x - m_prev) ** 2 / new_kappas[r]
            )

        log_P = new_log_P
        means, kappas, alphas, betas = new_means, new_kappas, new_alphas, new_betas

    return [int(b) for b in candidates if 0 < b < n - 1]
