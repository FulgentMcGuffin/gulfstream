"""Recursive ruptures + statistical testing for candidate breakpoints."""
from __future__ import annotations

import logging
from typing import Iterator

import numpy as np
import polars as pl
import ruptures as rpt

from gulfstream.common import frames
from gulfstream.common import utils
from gulfstream.common.options import SearchMethod
from gulfstream.common.results import SegmentResults
from gulfstream.detection import hyperparams as hyperparameter_selection
from gulfstream.detection import stat_tests as bkpt_stat_tests

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


def kernel_ruptures_generator(
    dfs: list[pl.DataFrame],
    params: dict,
) -> Iterator[list[dict]]:
    """Yield lists of ruptures kernel params (one entry per DF / PC)."""
    options = params["algo"].get("ruptures_kernel_params", [])
    for opt in options:
        resolved = [ _resolve_ruptures_gamma(df, opt) for df in dfs ]
        yield resolved


def late_algo_param_combos(params: dict) -> Iterator[dict]:
    depths = params["algo"].get("depth", [1])
    search_methods = params["algo"].get("search_method", [SearchMethod.PELT])
    for depth in depths:
        for search in search_methods:
            yield {"depth": depth, "search_method": search}


def _cost_and_params(kernel_params: dict) -> tuple[str, dict | None]:
    model = kernel_params.get("kernel", "rbf")
    cost = "rbf" if model == "rbf" else "l2"
    if cost == "rbf":
        return cost, {"gamma": float(kernel_params.get("gamma", 1.0))}
    return cost, None


def candidate_breakpoints(
    signal: np.ndarray,
    kernel_params: dict,
    *,
    search_method: str = SearchMethod.PELT,
    min_size: int = 10,
    algo_params: dict | None = None,
) -> list[int]:
    """Find candidate breakpoints via PELT, Binseg, BottomUp, WBS, or BOCPD."""
    from gulfstream.detection import search as search_mod

    cost, cost_params = _cost_and_params(kernel_params)
    method = str(search_method).lower()
    n = len(signal)
    algo_params = algo_params or {}
    n_bkps = max(1, n // 50)
    try:
        if method == SearchMethod.BINSEG:
            algo = rpt.Binseg(model=cost, params=cost_params, min_size=min_size, jump=5)
            bkpts = algo.fit_predict(signal, n_bkps=n_bkps)
        elif method == SearchMethod.BOTTOMUP:
            algo = rpt.BottomUp(model=cost, params=cost_params, min_size=min_size, jump=5)
            bkpts = algo.fit_predict(signal, n_bkps=n_bkps)
        elif method == SearchMethod.WBS or method == "wbs":
            wbs_intervals = algo_params.get("wbs_n_intervals", [200])
            wbs_thr = algo_params.get("wbs_threshold", [None])
            wbs_seed = algo_params.get("random_state", [42])
            n_int = int(wbs_intervals[0] if isinstance(wbs_intervals, list) else wbs_intervals)
            thr = wbs_thr[0] if isinstance(wbs_thr, list) else wbs_thr
            seed = wbs_seed[0] if isinstance(wbs_seed, list) else wbs_seed
            bkpts = search_mod.wild_binary_segmentation(
                signal,
                n_intervals=n_int,
                n_bkps=n_bkps,
                min_size=min_size,
                threshold=None if thr is None else float(thr),
                random_state=None if seed is None else int(seed),
            )
        elif method == SearchMethod.BOCPD or method == "bocpd":
            hazards = algo_params.get("bocpd_hazard", [1.0 / 100.0])
            thresholds = algo_params.get("bocpd_threshold", [0.5])
            max_runs = algo_params.get("bocpd_max_run", [None])
            hazard = float(hazards[0] if isinstance(hazards, list) else hazards)
            thr = float(thresholds[0] if isinstance(thresholds, list) else thresholds)
            max_run = max_runs[0] if isinstance(max_runs, list) else max_runs
            bkpts = search_mod.bayesian_online_changepoint_detection(
                signal,
                hazard=hazard,
                threshold=thr,
                max_run=None if max_run is None else int(max_run),
                min_size=min_size,
            )
        else:
            algo = rpt.Pelt(model=cost, params=cost_params, min_size=min_size)
            pen = float(np.log(max(n, 2)) * np.std(signal) ** 2)
            bkpts = algo.fit_predict(signal, pen=pen)
    except Exception:
        logger.exception("%s failed; falling back to Dynp single break", method)
        try:
            algo = rpt.Dynp(model=cost, params=cost_params, min_size=min_size, jump=5)
            bkpts = algo.fit_predict(signal, n_bkps=1)
        except Exception:
            return []
    return [int(b) for b in bkpts if 0 < int(b) < n - 1]


def find_and_test_bkpts(
    df: pl.DataFrame,
    mapped_dfs: list[pl.DataFrame],
    case_params: dict,
    dates: list[str],
    *,
    mapped_df_diffs: list[pl.DataFrame] | None = None,
    df_pca: pl.DataFrame | None = None,
) -> SegmentResults:
    """Recursive change-point search with statistical accept/reject."""
    depth = int(case_params["algo"].get("depth", 1))
    search_method = case_params["algo"].get("search_method", SearchMethod.PELT)
    if isinstance(search_method, list):
        search_method = search_method[0] if search_method else SearchMethod.PELT
    kernel_params = case_params["algo"].get("ruptures_kernel_params", {"kernel": "rbf"})
    if isinstance(kernel_params, list):
        kernel_params = kernel_params[0] if kernel_params else {"kernel": "rbf"}

    mapped = mapped_dfs[0] if mapped_dfs else df
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
        local = candidate_breakpoints(
            frames.to_numpy(segment),
            kernel_params,
            search_method=search_method,
            min_size=10,
            algo_params=case_params.get("algo", {}),
        )
        if not local:
            return
        for loc in local[:3]:
            bkpt = start + loc
            if bkpt <= start + 5 or bkpt >= end - 5:
                continue
            date = dates[bkpt] if bkpt < len(dates) else str(bkpt)
            test_params = dict(case_params["test"])
            try:
                hyp = hyperparameter_selection.select_hyperparameters(
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
            break

    recurse(0, n, 1)
    valid = sorted(set(valid))
    invalid = sorted(set(invalid) - set(valid))
    labels = utils.convert_bkpts_to_labels(valid, n)
    return SegmentResults(
        bkpts=valid,
        invalid_bkpts=invalid,
        stats=stats,
        hierarchy={b: hierarchy.get(b, 1) for b in valid + invalid},
        labels=labels,
        params=case_params,
    )
