"""Statistical tests for accepting/rejecting candidate breakpoints."""
from __future__ import annotations

import itertools
import logging
from typing import Iterator

import numpy as np
import polars as pl
from sklearn.metrics.pairwise import rbf_kernel

from gulfstream.common import frames

logger = logging.getLogger(__name__)


def _valid_mmd_common(test_params: dict) -> bool:
    valid = True
    for key in ("lag", "window", "sample_size"):
        if key not in test_params:
            logger.error("'%s' must be provided in 'test' for MMD tests.", key)
            valid = False
        elif not isinstance(test_params[key], list) or len(test_params[key]) == 0:
            logger.error("'%s' must be a nonempty list of dicts.", key)
            valid = False
    if "significance_level" in test_params:
        levels = test_params["significance_level"]
        if not isinstance(levels, list) or not all(isinstance(x, (int, float)) for x in levels):
            logger.error("'significance_level' must be a list of floats.")
            valid = False
    return valid


def _valid_mmd_no_ts(test_params: dict) -> bool:
    return _valid_mmd_common(test_params)


def _valid_mmd_ts(test_params: dict) -> bool:
    return _valid_mmd_common(test_params)


def _valid_mmd_perm(test_params: dict) -> bool:
    return _valid_mmd_common(test_params)


STAT_TEST_INPUT_VALIDATORS = {
    "mmd_no_ts": _valid_mmd_no_ts,
    "mmd_ts": _valid_mmd_ts,
    "mmd_perm": _valid_mmd_perm,
}

DEFAULT_STATS = {
    "mmd_no_ts": (0.0, 1.0),
    "mmd_ts": (0.0, 1.0),
    "mmd_perm": (0.0, 1.0),
}

STAT_COLUMN_NAMES = {
    "mmd_no_ts": ["mmd_stat", "mmd_pvalue"],
    "mmd_ts": ["mmd_stat", "mmd_pvalue"],
    "mmd_perm": ["mmd_stat", "mmd_pvalue"],
}

STAT_DTYPES = {
    "mmd_no_ts": [float, float],
    "mmd_ts": [float, float],
    "mmd_perm": [float, float],
}

PARAM_FORMATTERS = {
    "mmd_no_ts": lambda: {"lag": "lag", "window": "window", "sample_size": "sample_size"},
    "mmd_ts": lambda: {"lag": "lag", "window": "window", "sample_size": "sample_size"},
    "mmd_perm": lambda: {"lag": "lag", "window": "window", "sample_size": "sample_size"},
}


def _test_param_combos(params: dict) -> Iterator[dict]:
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


def _biased_mmd2(x: np.ndarray, y: np.ndarray, gamma: float) -> float:
    """Biased squared MMD with RBF kernel."""
    kxx = rbf_kernel(x, x, gamma=gamma)
    kyy = rbf_kernel(y, y, gamma=gamma)
    kxy = rbf_kernel(x, y, gamma=gamma)
    return float(kxx.mean() + kyy.mean() - 2.0 * kxy.mean())


def run_mmd_test(
    left: pl.DataFrame | np.ndarray,
    right: pl.DataFrame | np.ndarray,
    *,
    gamma: float | None = None,
    n_permutations: int = 50,
    significance_level: float = 0.05,
) -> tuple[float, float, bool]:
    """Return (statistic, p_value, accept_breakpoint).

    Accept means distributions differ enough to treat the candidate as a breakpoint.
    """
    if isinstance(left, pl.DataFrame):
        x = frames.to_numpy(left)
    else:
        x = np.asarray(left, dtype=float)
    if isinstance(right, pl.DataFrame):
        y = frames.to_numpy(right)
    else:
        y = np.asarray(right, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if gamma is None:
        combo = np.vstack([x, y])
        from sklearn.metrics.pairwise import pairwise_distances

        med = float(np.median(pairwise_distances(combo)))
        gamma = 1.0 / (2 * max(med, 1e-8) ** 2)

    stat = _biased_mmd2(x, y, gamma)
    # Permutation p-value.
    pooled = np.vstack([x, y])
    n_x = len(x)
    count = 0
    rng = np.random.default_rng(0)
    for _ in range(n_permutations):
        rng.shuffle(pooled)
        stat_p = _biased_mmd2(pooled[:n_x], pooled[n_x:], gamma)
        if stat_p >= stat:
            count += 1
    p_value = (count + 1) / (n_permutations + 1)
    accept = p_value < significance_level
    return stat, p_value, accept


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

    if choice in ("mmd_no_ts", "mmd_ts", "mmd_perm"):
        n_perm = 100 if choice == "mmd_perm" else 50
        return run_mmd_test(left, right, n_permutations=n_perm, significance_level=sig)
    raise ValueError(f"Unknown test choice {choice}")
