"""Kernel feature map approximation (RFF / Nystroem / raw) for regime detection."""
from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import polars as pl
from sklearn.kernel_approximation import Nystroem, RBFSampler

from gulfstream.common import frames
from gulfstream.common import utils
from gulfstream.common.results import ManyMappingResults, MappingResults

logger = logging.getLogger(__name__)

SEEDS = [41, 42, 100, 200, 500]


def map_data(
    df: pl.DataFrame,
    method: Literal["rff", "nystroem", "raw", "inv_rff"],
    **kwargs,
) -> MappingResults:
    handlers = {
        "rff": _rff_data,
        "nystroem": _nystroem_data,
        "raw": _raw_data,
        "inv_rff": _rff_data,
    }
    handler = handlers.get(method)
    if not handler:
        raise ValueError(f"Unknown kernel feature map approximation method {method}.")
    return handler(df, **kwargs)


def _process_heuristics(df: pl.DataFrame | None, kernel_params: dict) -> dict:
    if df is None:
        return dict(kernel_params) if kernel_params else {}
    params = dict(kernel_params or {})
    if params.get("kernel") == "rbf":
        gamma_method = params.get("gamma_method")
        if gamma_method and gamma_method != "user_specified":
            params["gamma"] = utils._calculate_bandwidth(df, gamma_method)
        elif "gamma" in params and isinstance(params["gamma"], str):
            params["gamma"] = utils._calculate_bandwidth(df, params["gamma"])
        elif gamma_method == "user_specified" and "gamma" not in params:
            raise KeyError("gamma required for user_specified")
    return params


def _rff_data(
    df: pl.DataFrame,
    *,
    num_features: int,
    kernel_params: dict,
    random_state: int | None = None,
    **kwargs,
) -> MappingResults:
    gamma = kernel_params.get("gamma")
    if gamma is None:
        gamma = utils._calculate_bandwidth(df, "median")
    sampler = RBFSampler(
        gamma=float(gamma),
        n_components=int(num_features),
        random_state=random_state,
    )
    mapped = sampler.fit_transform(frames.to_numpy(df))
    out = frames.with_same_dates(mapped, df)
    return MappingResults(
        df=out,
        feature_map_approx_method="rff",
        num_features=int(num_features),
        model=sampler,
        kernel_params=dict(kernel_params),
    )


def _nystroem_data(
    df: pl.DataFrame,
    *,
    num_features: int,
    kernel_params: dict,
    random_state: int | None = None,
    **kwargs,
) -> MappingResults:
    params = dict(kernel_params)
    kernel = params.pop("kernel", "rbf")
    params.pop("gamma_method", None)
    model = Nystroem(
        kernel=kernel,
        n_components=int(num_features),
        random_state=random_state,
        **{k: v for k, v in params.items() if k in {"gamma", "degree", "coef0"}},
    )
    mapped = model.fit_transform(frames.to_numpy(df))
    out = frames.with_same_dates(mapped, df)
    return MappingResults(
        df=out,
        feature_map_approx_method="nystroem",
        num_features=int(num_features),
        model=model,
        kernel_params=dict(kernel_params),
    )


def _raw_data(df: pl.DataFrame, **kwargs) -> MappingResults:
    return MappingResults(
        df=df.clone(),
        feature_map_approx_method="raw",
        num_features=frames.n_features(df),
        kernel_params={},
    )


def _get_diffs_if_needed(df: pl.DataFrame, params: dict) -> pl.DataFrame | None:
    if "test" not in params or "choice" not in params["test"]:
        return None
    if "mmd_ts" in params["test"]["choice"] or "mmd_perm" in params["test"]["choice"]:
        feat_cols = frames.feature_columns(df)
        return df.with_columns([pl.col(c).diff() for c in feat_cols]).drop_nulls()
    return None


def _map_df_many_times(df, method, num_mappings, **kwargs):
    if df is None:
        return None
    return [map_data(df, method, **kwargs).df for _ in range(num_mappings)]


def _rff_param_generator(params: dict):
    for kernel_params in params["algo"].get("feature_map_kernel_params", []):
        if kernel_params.get("kernel") == "rbf":
            yield kernel_params


def _nystroem_param_generator(params: dict):
    valid = {"rbf", "laplacian", "poly", "additive_chi2", "sigmoid", "chi2", "linear", "cosine"}
    for kernel_params in params["algo"].get("feature_map_kernel_params", []):
        if kernel_params.get("kernel") in valid:
            yield kernel_params


def _raw_param_generator(params: dict):
    yield {}


def _mapping_param_generator(params: dict):
    handlers = {
        "rff": _rff_param_generator,
        "nystroem": _nystroem_param_generator,
        "inv_rff": _rff_param_generator,
        "raw": _raw_param_generator,
    }
    for method in params["algo"].get("feature_map_approx_method", []):
        handler = handlers.get(method)
        if not handler:
            raise ValueError(f"Unknown method {method}.")
        for kernel_params in handler(params):
            yield method, kernel_params


def _num_features_generator(df: pl.DataFrame, params: dict, **kwargs):
    method = kwargs.get("feature_map_approx_method")
    if method == "raw":
        yield {"num_features": frames.n_features(df)}
        return
    for n in params["algo"].get("num_features", []):
        yield {"num_features": n}


def _feature_map_generator(df: pl.DataFrame, params: dict):
    df_diff = _get_diffs_if_needed(df, params)
    for method, kernel_params in _mapping_param_generator(params):
        num_mappings_list = [1] if method == "raw" else params["algo"].get("num_mappings", [1])
        max_maps = max(num_mappings_list)
        signal_params = _process_heuristics(df, kernel_params)
        diff_params = _process_heuristics(df_diff, kernel_params)
        for num_features_res in _num_features_generator(
            df, params, feature_map_approx_method=method, kernel_params=signal_params
        ):
            num_features = num_features_res["num_features"]
            mapped_dfs = _map_df_many_times(
                df, method, max_maps, num_features=num_features, kernel_params=signal_params
            )
            mapped_df_diffs = _map_df_many_times(
                df_diff, method, max_maps, num_features=num_features, kernel_params=diff_params
            )
            for num_maps in num_mappings_list:
                yield ManyMappingResults(
                    dfs=mapped_dfs[:num_maps],
                    kernel_params=signal_params,
                    feature_map_approx_method=method,
                    num_features=num_features,
                    df_diffs=mapped_df_diffs[:num_maps] if df_diff is not None else None,
                    diff_kernel_params=diff_params if df_diff is not None else None,
                )


def pca_feature_map_generator(df: pl.DataFrame, params: dict):
    gens = [
        _feature_map_generator(frames.select_features(df, [col]), params)
        for col in frames.feature_columns(df)
    ]
    for objects in zip(*gens):
        yield list(objects)


def _legacy_feature_map_generator(df: pl.DataFrame, params: dict):
    """Generate kernel feature mappings for legacy regime detection.

    Differs from ``_feature_map_generator`` by:
    1. Emitting only one mapping per parameter combination.
    2. Skipping first-order differences.
    """
    for method, kernel_params in _mapping_param_generator(params):
        signal_params = _process_heuristics(df, kernel_params)
        num_features_params = {
            "feature_map_approx_method": method,
            "kernel_params": signal_params,
        }
        for num_features_res in _num_features_generator(df, params, **num_features_params):
            num_features = num_features_res.get("num_features")
            if not num_features:
                raise KeyError("'num_features' must be specified.")
            mapped_dfs = _map_df_many_times(
                df,
                method,
                1,
                num_features=num_features,
                kernel_params=signal_params,
            )
            yield ManyMappingResults(
                dfs=mapped_dfs,
                kernel_params=signal_params,
                feature_map_approx_method=method,
                num_features=num_features,
                kernel_approx_error=num_features_res.get("kernel_approx_error"),
            )
