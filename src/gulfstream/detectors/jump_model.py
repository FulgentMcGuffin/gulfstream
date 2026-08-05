"""Jump-model (temporal clustering) regime detection.

Alternating minimization of within-cluster squared error plus a jump penalty
λ · #{t : s_t ≠ s_{t-1}}, with exact DP assignment given cluster centres
(Bemporad / Nystrup-style discrete jump model).
"""
from __future__ import annotations

import logging

import numpy as np
import polars as pl

from gulfstream.common import frames, utils
from gulfstream.common.results import AlgoResults
from gulfstream.detectors import common_validation as common

logger = logging.getLogger(__name__)


def _dp_assign(X: np.ndarray, centres: np.ndarray, jump_penalty: float) -> np.ndarray:
    """Exact DP state path for fixed centres and jump penalty."""
    n, _ = X.shape
    k = centres.shape[0]
    # costs[t, j] = ||x_t - μ_j||²
    costs = np.sum((X[:, None, :] - centres[None, :, :]) ** 2, axis=2)
    dp = np.full((n, k), np.inf)
    prev = np.full((n, k), -1, dtype=int)
    dp[0] = costs[0]
    for t in range(1, n):
        for j in range(k):
            # stay vs jump
            stay = dp[t - 1, j]
            jump = dp[t - 1].min() + jump_penalty
            if stay <= jump:
                dp[t, j] = costs[t, j] + stay
                prev[t, j] = j
            else:
                dp[t, j] = costs[t, j] + jump
                prev[t, j] = int(np.argmin(dp[t - 1]))
    # backtrack
    labels = np.empty(n, dtype=int)
    labels[-1] = int(np.argmin(dp[-1]))
    for t in range(n - 2, -1, -1):
        labels[t] = prev[t + 1, labels[t + 1]]
    return labels


def _update_centres(X: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    p = X.shape[1]
    centres = np.zeros((k, p), dtype=float)
    for j in range(k):
        mask = labels == j
        if mask.any():
            centres[j] = X[mask].mean(axis=0)
        else:
            # re-seed empty cluster at a random point
            centres[j] = X[np.random.randint(0, len(X))]
    return centres


def jump_model_predict_regimes(
    df: pl.DataFrame | np.ndarray,
    regimes: int,
    jump_penalty: float = 1.0,
    max_iter: int = 20,
    random_state: int | None = 42,
    tol: float = 1e-6,
    **kwargs,
) -> AlgoResults:
    """Fit a discrete jump model with ``regimes`` modes and penalty ``jump_penalty``."""
    X = frames.to_numpy(df) if isinstance(df, pl.DataFrame) else np.asarray(df, dtype=float)
    n, p = X.shape
    if regimes < 1:
        raise ValueError("regimes must be positive")
    rng = np.random.default_rng(random_state)
    # k-means++-ish init: random subset of rows as centres
    init_idx = rng.choice(n, size=min(regimes, n), replace=False)
    centres = X[init_idx].copy()
    if centres.shape[0] < regimes:
        pad = np.repeat(centres[-1:], regimes - centres.shape[0], axis=0)
        centres = np.vstack([centres, pad])

    labels = np.zeros(n, dtype=int)
    prev_obj = np.inf
    for _ in range(max_iter):
        labels = _dp_assign(X, centres, float(jump_penalty))
        centres = _update_centres(X, labels, regimes)
        # objective
        recon = np.sum((X - centres[labels]) ** 2)
        jumps = float(np.sum(labels[1:] != labels[:-1]))
        obj = recon + float(jump_penalty) * jumps
        if abs(prev_obj - obj) < tol * max(1.0, prev_obj):
            break
        prev_obj = obj

    labels = utils._map_labels_to_ordered_integers(labels)
    bkpts = utils._convert_labels_to_bkpts(labels)
    return AlgoResults(bkpts=bkpts, labels=labels)


def jump_model_param_generator(params: dict):
    if "jump_model" not in params["algo"].get("regime_detection_algorithm", []):
        return
    for regimes in params["algo"]["regimes"]:
        for jump_penalty in common.algo_grid(params, "jump_penalty", [1.0]):
            for random_state in common.algo_grid(params, "random_state", [42]):
                for max_iter in common.algo_grid(params, "jump_max_iter", [20]):
                    yield {
                        "regime_detection_algorithm": "jump_model",
                        "regimes": regimes,
                        "jump_penalty": jump_penalty,
                        "random_state": random_state,
                        "max_iter": max_iter,
                    }


def jump_model_params_printout() -> dict:
    return {
        "jump_model_regimes": ["number of regimes"],
        "jump_model_jump_penalty": ["jump penalty"],
        "jump_model_random_state": ["random_state"],
    }


def jump_model_input_validator(algo_params: dict) -> bool:
    return common._valid_regimes(algo_params)
