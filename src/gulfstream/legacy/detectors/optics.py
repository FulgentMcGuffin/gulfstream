from __future__ import annotations
"""
Regime detection using OPTICS (Ordering Points To Identify the Clustering Structure).
"""
import logging
import math

import numpy as np
import polars as pl

from gulfstream.dimred import density as dens
from gulfstream.common import frames
from gulfstream.common import utils
from gulfstream.common.results import AlgoResults

logger = logging.getLogger(__name__)


def optics_predict_regimes(
    df: pl.DataFrame | np.ndarray,
    min_samples: int = 5,
    max_eps: float = math.inf,
    metric: str = "minkowski",
    p: int = 2,
    cluster_method: str = "xi",
    xi: float = 0.05,
    min_cluster_size: int | None = None,
    eps: float | None = None,
    **kwargs,
) -> AlgoResults:
    """Predict regimes by OPTICS clustering on ``df`` (noise label ``-1``)."""
    X = frames.to_numpy(df) if isinstance(df, pl.DataFrame) else np.asarray(df)
    _model, raw = dens.fit_optics(
        X,
        min_samples=min_samples,
        max_eps=max_eps,
        metric=metric,
        p=p,
        cluster_method=cluster_method,
        xi=xi,
        min_cluster_size=min_cluster_size,
        eps=eps,
    )
    labels = utils._map_labels_to_ordered_integers(raw.tolist())
    bkpts = utils._convert_labels_to_bkpts(labels)
    return AlgoResults(bkpts=bkpts, labels=labels)


def optics_param_generator(params: dict):
    """Yield valid OPTICS regime-detection parameter combinations."""
    if "optics" not in params["algo"]["regime_detection_algorithm"]:
        return
    for min_samples in params["algo"].get("optics_min_samples", [5]):
        for max_eps in params["algo"].get("optics_max_eps", [math.inf]):
            for metric in params["algo"].get("optics_metric", ["minkowski"]):
                for p in params["algo"].get("optics_p", [2]):
                    for cluster_method in params["algo"].get(
                        "optics_cluster_method", ["xi"]
                    ):
                        for xi in params["algo"].get("optics_xi", [0.05]):
                            for min_cluster_size in params["algo"].get(
                                "optics_min_cluster_size", [None]
                            ):
                                for eps in params["algo"].get("optics_eps", [None]):
                                    yield {
                                        "regime_detection_algorithm": "optics",
                                        "min_samples": min_samples,
                                        "max_eps": max_eps,
                                        "metric": metric,
                                        "p": p,
                                        "cluster_method": cluster_method,
                                        "xi": xi,
                                        "min_cluster_size": min_cluster_size,
                                        "eps": eps,
                                    }


def optics_params_printout() -> dict:
    return {
        "optics_min_samples": ["min_samples"],
        "optics_max_eps": ["max_eps"],
        "optics_metric": ["metric"],
        "optics_p": ["p"],
        "optics_cluster_method": ["cluster_method"],
        "optics_xi": ["xi"],
        "optics_min_cluster_size": ["min_cluster_size"],
        "optics_eps": ["eps"],
    }


def optics_input_validator(algo_params: dict) -> bool:
    """Return True if ``algo_params`` is valid for OPTICS regime detection."""
    valid = True
    ms = algo_params.get("optics_min_samples")
    if ms is not None:
        if not isinstance(ms, list) or not ms:
            logger.error("'optics_min_samples' must be a nonempty list.")
            valid = False
        elif not all(isinstance(x, int) and x >= 2 for x in ms):
            logger.error("All entries of 'optics_min_samples' must be ints >= 2.")
            valid = False
    max_eps = algo_params.get("optics_max_eps")
    if max_eps is not None:
        if not isinstance(max_eps, list) or not max_eps:
            logger.error("'optics_max_eps' must be a nonempty list.")
            valid = False
        elif not all(
            isinstance(x, (int, float)) and (math.isinf(float(x)) or x > 0)
            for x in max_eps
        ):
            logger.error(
                "All entries of 'optics_max_eps' must be positive floats or inf."
            )
            valid = False
    metrics = algo_params.get("optics_metric")
    if metrics is not None and (
        not isinstance(metrics, list) or not all(isinstance(x, str) for x in metrics)
    ):
        logger.error("'optics_metric' must be list[str].")
        valid = False
    p_vals = algo_params.get("optics_p")
    if p_vals is not None and (
        not isinstance(p_vals, list)
        or not all(isinstance(x, int) and x >= 1 for x in p_vals)
    ):
        logger.error("'optics_p' must be list of positive ints.")
        valid = False
    methods = algo_params.get("optics_cluster_method")
    if methods is not None and (
        not isinstance(methods, list)
        or not all(m in ("xi", "dbscan") for m in methods)
    ):
        logger.error("'optics_cluster_method' must be list of 'xi'/'dbscan'.")
        valid = False
    xi = algo_params.get("optics_xi")
    if xi is not None and (
        not isinstance(xi, list)
        or not all(isinstance(x, (int, float)) and 0 < float(x) < 1 for x in xi)
    ):
        logger.error("'optics_xi' must be list of floats in (0, 1).")
        valid = False
    mcs = algo_params.get("optics_min_cluster_size")
    if mcs is not None:
        if not isinstance(mcs, list) or not mcs:
            logger.error("'optics_min_cluster_size' must be a nonempty list.")
            valid = False
        elif not all(x is None or (isinstance(x, int) and x >= 2) for x in mcs):
            logger.error(
                "All entries of 'optics_min_cluster_size' must be ints >= 2 or null."
            )
            valid = False
    eps = algo_params.get("optics_eps")
    if eps is not None:
        if not isinstance(eps, list) or not eps:
            logger.error("'optics_eps' must be a nonempty list.")
            valid = False
        elif not all(
            x is None or (isinstance(x, (int, float)) and x > 0) for x in eps
        ):
            logger.error("All entries of 'optics_eps' must be positive floats or null.")
            valid = False
    return valid
