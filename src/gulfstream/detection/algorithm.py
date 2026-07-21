"""Custom recursive ruptures + statistical testing."""
from __future__ import annotations

import logging
from typing import Iterator

import numpy as np
import polars as pl
import ruptures as rpt

from gulfstream.detection import stat_tests as bkpt_stat_tests
from gulfstream.detection import hyperparams as custom_hyperparameter_selection
from gulfstream.common import frames
from gulfstream.common import utils
from gulfstream.common.results import SegmentResults

logger = logging.getLogger(__name__)


def _resolve_ruptures_gamma(df: pl.DataFrame, kernel_params: dict) -> dict:
    params = dict(kernel_params)
    if params.get("kernel") != "rbf":
        return params
    method = params.get("gamma_method")
    if method == "user_specified":
        return params
    if method in ("median", "sk_scale") or isinstance(params.get("gamma"), str):
        gamma_src = method if method in ("median", "sk_scale") else params["gamma"]
        params["gamma"] = utils._calculate_bandwidth(df, gamma_src)
    return params


def _kernel_ruptures_generator(
    dfs: list[pl.DataFrame],
    params: dict,
) -> Iterator[list[dict]]:
    """Yield lists of ruptures kernel params (one entry per DF / PC)."""
    options = params["algo"].get("ruptures_kernel_params", [])
    for opt in options:
        resolved = []
        for df in dfs:
            resolved.append(_resolve_ruptures_gamma(df, opt))
        yield resolved


def _late_algo_param_combos(params: dict) -> Iterator[dict]:
    depths = params["algo"].get("depth", [1])
    for depth in depths:
        yield {"depth": depth}


def _pelt_candidate(
    signal: np.ndarray,
    kernel_params: dict,
    min_size: int = 10,
) -> list[int]:
    """Find candidate breakpoints via PELT on the mapped signal."""
    model = kernel_params.get("kernel", "rbf")
    # ruptures cost models: rbf / l1 / l2. Map linear/cosine → l2.
    cost = "rbf" if model == "rbf" else "l2"
    cost_params = {}
    if cost == "rbf":
        cost_params["gamma"] = float(kernel_params.get("gamma", 1.0))
    try:
        algo = rpt.Pelt(model=cost, params=cost_params or None, min_size=min_size)
        pen = float(np.log(len(signal)) * np.std(signal) ** 2)
        bkpts = algo.fit_predict(signal, pen=pen)
    except Exception:
        logger.exception("PELT failed; falling back to Dynp single break")
        try:
            algo = rpt.Dynp(model=cost, params=cost_params or None, min_size=min_size, jump=5)
            bkpts = algo.fit_predict(signal, n_bkps=1)
        except Exception:
            return []
    n = len(signal)
    return [int(b) for b in bkpts if 0 < int(b) < n - 1]


def _find_and_test_bkpts(
    df: pl.DataFrame,
    mapped_dfs: list[pl.DataFrame],
    case_params: dict,
    dates: list[str],
    *,
    mapped_df_diffs: list[pl.DataFrame] | None = None,
    df_pca: pl.DataFrame | None = None,
) -> SegmentResults:
    """Recursive change-point search with MMD accept/reject."""
    depth = int(case_params["algo"].get("depth", 1))
    kernel_params = case_params["algo"].get("ruptures_kernel_params", {"kernel": "rbf"})
    if isinstance(kernel_params, list):
        kernel_params = kernel_params[0] if kernel_params else {"kernel": "rbf"}

    mapped = mapped_dfs[0] if mapped_dfs else df
    # Align lengths (diffs may be shorter).
    n = min(df.height, mapped.height)
    df = frames.slice_rows(df, 0, n)
    mapped = frames.slice_rows(mapped, 0, n)

    valid: list[int] = []
    invalid: list[int] = []
    stats: dict[int, tuple] = {}
    hierarchy: dict[int, int] = {}

    def recurse(start: int, end: int, level: int) -> None:
        if level > depth or end - start < 30:
            return
        segment = frames.slice_rows(mapped, start, end)
        local = _pelt_candidate(frames.to_numpy(segment), kernel_params, min_size=10)
        if not local:
            return
        # Take strongest (first) candidate in segment coordinates.
        for loc in local[:3]:
            bkpt = start + loc
            if bkpt <= start + 5 or bkpt >= end - 5:
                continue
            date = dates[bkpt] if bkpt < len(dates) else str(bkpt)
            test_params = dict(case_params["test"])
            # Data-driven hyperparams around this candidate.
            try:
                hyp = custom_hyperparameter_selection.select_custom_hyperparameters(
                    {"test": test_params, "algo": case_params.get("algo", {})},
                    date,
                    df=df_pca if df_pca is not None else df,
                    bkpt=bkpt,
                )
                if hyp.lag:
                    test_params["lag"] = hyp.lag
                if hyp.window:
                    test_params["window"] = hyp.window
                if hyp.sample_size:
                    test_params["sample_size"] = hyp.sample_size
            except Exception:
                logger.exception("Hyperparameter selection failed at %s; using defaults", date)

            try:
                stat, pval, accept = bkpt_stat_tests.test_breakpoint(
                    frames.slice_rows(df, start, end),
                    bkpt - start,
                    test_params,
                    mapped_df=segment,
                )
            except Exception:
                logger.exception("Stat test failed at %s", date)
                invalid.append(bkpt)
                stats[bkpt] = (0.0, 1.0)
                hierarchy[bkpt] = level
                continue

            stats[bkpt] = (stat, pval)
            hierarchy[bkpt] = level
            if accept:
                valid.append(bkpt)
                recurse(start, bkpt, level + 1)
                recurse(bkpt, end, level + 1)
            else:
                invalid.append(bkpt)
            break  # one accepted/rejected candidate per segment at this level

    recurse(0, n, 1)
    valid = sorted(set(valid))
    invalid = sorted(set(invalid) - set(valid))
    labels = utils._convert_bkpts_to_labels(valid, n)
    return SegmentResults(
        bkpts=valid,
        invalid_bkpts=invalid,
        stats=stats,
        hierarchy={b: hierarchy.get(b, 1) for b in valid + invalid},
        labels=labels,
        params=case_params,
    )
