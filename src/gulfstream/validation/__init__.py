"""Main implementation for input validation."""
from __future__ import annotations

import logging

from gulfstream.detection.stat_tests import STAT_TEST_INPUT_VALIDATORS
from gulfstream.validation import (
    algo,
    dimred,
    kernel_feature_map,
    log,
    metrics,
    retrain,
    robustness,
    stability,
)

logger = logging.getLogger(__name__)


def _valid_test_params(params: dict) -> bool:
    valid = True
    test_params = params.get("test")
    if not test_params:
        logger.error("'test' (dict) must be provided.")
        return False
    if not isinstance(test_params, dict):
        logger.error("'test' must be type dict. Got type %s.", type(test_params))
        return False
    if "choice" not in test_params:
        logger.error(
            "Must include key 'choice' (list[str]) in 'test' specifying which "
            "statistical test(s) to use."
        )
        return False
    requested_tests = set(test_params["choice"])
    for test in requested_tests:
        handler = STAT_TEST_INPUT_VALIDATORS.get(test)
        if not handler:
            logger.error(
                "Unknown test %s. Valid tests are %s.",
                test,
                ", ".join(STAT_TEST_INPUT_VALIDATORS.keys()),
            )
            valid = False
        elif not handler(params["test"]):
            valid = False
    return valid


def _valid_params_for_user_specified_df(params: dict) -> bool:
    valid = True
    if not algo._valid_algo_params(params):
        valid = False
    if not dimred._valid_dimred_params(params):
        valid = False
    if not kernel_feature_map._valid_kernel_feature_map_params(params):
        valid = False
    if not log._valid_log_params(params):
        valid = False
    if not metrics._valid_metrics_params(params):
        valid = False
    if not _valid_test_params(params):
        valid = False
    if not robustness._valid_robustness_params(params):
        valid = False
    if not stability._valid_stability_params(params):
        valid = False
    if not retrain._valid_retrain_params(params):
        valid = False
    return valid
