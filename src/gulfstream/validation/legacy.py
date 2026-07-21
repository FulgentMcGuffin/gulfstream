"""Input validation for the legacy regime detection algorithms."""
from __future__ import annotations

import logging

from gulfstream.validation import (
    algo,
    common,
    dimred,
    kernel_feature_map,
    log,
    metrics,
)
from gulfstream.legacy.detectors.bayesian_gmm import bayesian_gmm_input_validator
from gulfstream.legacy.detectors.hdbscan import hdbscan_input_validator
from gulfstream.legacy.detectors.hmm import hmm_input_validator
from gulfstream.legacy.detectors.kmeans import kmeans_input_validator
from gulfstream.legacy.detectors.msar import msar_input_validator
from gulfstream.legacy.detectors.optics import optics_input_validator
from gulfstream.legacy.detectors.ruptures_methods import ruptures_input_validator
from gulfstream.legacy.detectors.wasserstein import wass_input_validator

logger = logging.getLogger(__name__)

LEGACY_INPUT_VALIDATORS = {
    "bayesian_gmm": bayesian_gmm_input_validator,
    "hmm": hmm_input_validator,
    "kmeans": kmeans_input_validator,
    "hdbscan": hdbscan_input_validator,
    "optics": optics_input_validator,
    "msar": msar_input_validator,
    "ruptures": ruptures_input_validator,
    "wasserstein": wass_input_validator,
}


def _valid_legacy_post_processing_params(params: dict) -> bool:
    if not common._is_nonempty_list(params["algo"], "post_processing_method", str):
        return False
    valid = True
    valid_methods = {
        "majority_voting": algo._valid_majority_voting,
        "no_post_processing": algo._valid_no_post_processing,
        "neighbor_comparison": algo._valid_neighbor_comparison,
    }
    for method in params["algo"]["post_processing_method"]:
        if method not in valid_methods:
            logger.error(
                "Unknown post_processing_method %s. Known methods are %s.",
                method,
                ", ".join(valid_methods.keys()),
            )
            valid = False
        elif not valid_methods[method](params):
            valid = False
    return valid


def _valid_legacy_algo_params(params: dict) -> bool:
    algos = params.get("algo", {}).get("regime_detection_algorithm")
    if algos is None:
        logger.error("'regime_detection_algorithm' must be specified.")
        return False
    if not isinstance(algos, list):
        logger.error(
            "'regime_detection_algorithm' must be type list[str]. Got type %s.",
            type(algos),
        )
        return False
    if not all(x in LEGACY_INPUT_VALIDATORS for x in algos):
        logger.error("At least one entry of 'regime_detection_algorithm' is invalid.")
        return False
    valid = True
    for algo in algos:
        handler = LEGACY_INPUT_VALIDATORS[algo]
        if not handler(params["algo"]):
            valid = False
    return valid


def _valid_legacy_metrics_params(params: dict) -> bool:
    if not metrics._provided_metrics(params):
        return False
    valid = True
    if not metrics._valid_mode(params):
        valid = False
    if not metrics._valid_dir(params):
        valid = False
    if not metrics._valid_plot(params):
        valid = False
    if not metrics._valid_regime_cluster_algorithms(params):
        valid = False
    elif params["metrics"]["plot"]:
        validators = [
            metrics._valid_explainability_features,
            metrics._valid_features_to_plot,
        ]
        for validator in validators:
            if not validator(params):
                valid = False
    return valid


def _valid_legacy_kernel_feature_map_params(params: dict) -> bool:
    if not kernel_feature_map._provided_feature_map_approx_method(params):
        return False
    valid = True
    if (
        kernel_feature_map._using_rff(params)
        or kernel_feature_map._using_inv_rff(params)
    ):
        kernel_params_list = params["algo"].get("feature_map_kernel_params")
        if not kernel_params_list:
            logger.error(
                "Must provide nonempty list for 'feature_map_kernel_params' if using "
                "'feature_map_approx_method' other than 'raw'."
            )
            valid = False
        else:
            provided_rbf = False
            for kernel_params in kernel_params_list:
                if kernel_params.get("kernel") == "rbf":
                    provided_rbf = True
                    if not kernel_feature_map._valid_rbf_kernel_params(
                        kernel_params
                    ):
                        valid = False
            if not provided_rbf:
                logger.error(
                    "If using 'rff' or 'inv_rff' for 'feature_map_approx_method', must provide "
                    "at least one entry of 'feature_map_kernel_params' with 'rbf' kernel."
                )
                valid = False
    if kernel_feature_map._using_nystroem(params):
        if not kernel_feature_map._valid_kernels(params):
            valid = False
    if (
        kernel_feature_map._using_rff(params)
        or kernel_feature_map._using_inv_rff(params)
        or kernel_feature_map._using_nystroem(params)
    ):
        if not kernel_feature_map._valid_num_features(params):
            valid = False
    return valid


def _valid_legacy_params_with_user_specified_df(params: dict) -> bool:
    """Validate input for legacy algorithms with user-specified time series data."""
    if not common._provided_algo(params):
        return False
    validators = [
        dimred._valid_dimred_params,
        _valid_legacy_kernel_feature_map_params,
        _valid_legacy_post_processing_params,
        _valid_legacy_algo_params,
        log._valid_log_params,
        _valid_legacy_metrics_params,
    ]
    valid = True
    for validator in validators:
        if not validator(params):
            valid = False
    return valid
