from __future__ import annotations
"""
Regime detection using HDBSCAN (hierarchical density-based clustering).
"""
import logging

import numpy as np
import polars as pl

from gulfstream.dimred import density as dens
from gulfstream.common import frames
from gulfstream.common import utils
from gulfstream.common.results import AlgoResults

logger = logging.getLogger(__name__)


def hdbscan_predict_regimes(
    df: pl.DataFrame | np.ndarray,
    min_cluster_size: int = 5,
    min_samples: int | None = None,
    metric: str = "euclidean",
    cluster_selection_epsilon: float = 0.0,
    cluster_selection_method: str = "eom",
    allow_single_cluster: bool = False,
    **kwargs,
) -> AlgoResults:
    """Predict regimes by HDBSCAN clustering on ``df`` (noise label ``-1``)."""
    X = frames.to_numpy(df) if isinstance(df, pl.DataFrame) else np.asarray(df)
    _model, raw = dens.fit_hdbscan(
        X,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_epsilon=cluster_selection_epsilon,
        cluster_selection_method=cluster_selection_method,
        allow_single_cluster=allow_single_cluster,
    )
    labels = utils._map_labels_to_ordered_integers(raw.tolist())
    bkpts = utils._convert_labels_to_bkpts(labels)
    return AlgoResults(bkpts=bkpts, labels=labels)


def hdbscan_param_generator(params: dict):
    """Yield valid HDBSCAN regime-detection parameter combinations."""
    if "hdbscan" not in params["algo"]["regime_detection_algorithm"]:
        return
    for min_cluster_size in params["algo"].get("hdbscan_min_cluster_size", [5]):
        for min_samples in params["algo"].get("hdbscan_min_samples", [None]):
            for metric in params["algo"].get("hdbscan_metric", ["euclidean"]):
                for eps in params["algo"].get("hdbscan_cluster_selection_epsilon", [0.0]):
                    for method in params["algo"].get(
                        "hdbscan_cluster_selection_method", ["eom"]
                    ):
                        for allow in params["algo"].get(
                            "hdbscan_allow_single_cluster", [False]
                        ):
                            yield {
                                "regime_detection_algorithm": "hdbscan",
                                "min_cluster_size": min_cluster_size,
                                "min_samples": min_samples,
                                "metric": metric,
                                "cluster_selection_epsilon": eps,
                                "cluster_selection_method": method,
                                "allow_single_cluster": allow,
                            }


def hdbscan_params_printout() -> dict:
    return {
        "hdbscan_min_cluster_size": ["min_cluster_size"],
        "hdbscan_min_samples": ["min_samples"],
        "hdbscan_metric": ["metric"],
        "hdbscan_cluster_selection_epsilon": ["cluster_selection_epsilon"],
        "hdbscan_cluster_selection_method": ["cluster_selection_method"],
        "hdbscan_allow_single_cluster": ["allow_single_cluster"],
    }


def _valid_positive_int_list(algo_params: dict, key: str) -> bool:
    vals = algo_params.get(key)
    if vals is None:
        return True
    if not isinstance(vals, list) or not vals:
        logger.error("'%s' must be a nonempty list.", key)
        return False
    if not all(isinstance(x, int) and x >= 2 for x in vals):
        logger.error("All entries of '%s' must be ints >= 2.", key)
        return False
    return True


def hdbscan_input_validator(algo_params: dict) -> bool:
    """Return True if ``algo_params`` is valid for HDBSCAN regime detection."""
    valid = True
    if not _valid_positive_int_list(algo_params, "hdbscan_min_cluster_size"):
        valid = False
    ms = algo_params.get("hdbscan_min_samples")
    if ms is not None:
        if not isinstance(ms, list) or not ms:
            logger.error("'hdbscan_min_samples' must be a nonempty list.")
            valid = False
        elif not all(x is None or (isinstance(x, int) and x >= 1) for x in ms):
            logger.error(
                "All entries of 'hdbscan_min_samples' must be positive ints or null."
            )
            valid = False
    metrics = algo_params.get("hdbscan_metric")
    if metrics is not None and (
        not isinstance(metrics, list) or not all(isinstance(x, str) for x in metrics)
    ):
        logger.error("'hdbscan_metric' must be list[str].")
        valid = False
    methods = algo_params.get("hdbscan_cluster_selection_method")
    if methods is not None:
        if not isinstance(methods, list) or not all(
            m in ("eom", "leaf") for m in methods
        ):
            logger.error(
                "'hdbscan_cluster_selection_method' must be list of 'eom'/'leaf'."
            )
            valid = False
    eps = algo_params.get("hdbscan_cluster_selection_epsilon")
    if eps is not None and (
        not isinstance(eps, list)
        or not all(isinstance(x, (int, float)) and x >= 0 for x in eps)
    ):
        logger.error(
            "'hdbscan_cluster_selection_epsilon' must be list of non-negative floats."
        )
        valid = False
    allow = algo_params.get("hdbscan_allow_single_cluster")
    if allow is not None and (
        not isinstance(allow, list) or not all(isinstance(x, bool) for x in allow)
    ):
        logger.error("'hdbscan_allow_single_cluster' must be list[bool].")
        valid = False
    return valid
