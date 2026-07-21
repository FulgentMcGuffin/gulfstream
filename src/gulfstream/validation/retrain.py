"""Input validation for the Graph 2 retrain section."""
from __future__ import annotations

import logging

import polars as pl

logger = logging.getLogger(__name__)


def _valid_retrain_params(params: dict) -> bool:
    """Validate ``params['retrain']`` when present.

    Required:
      - interactive: bool
      - features: list | dict | '__auto__'
    When interactive is False:
      - threshold: float
      - max_iter: int > 0
    Optional:
      - num_worst_features: int > 0
      - regimes_df: null | path str | {path: ...} | list[records] | DataFrame
    """
    if "retrain" not in params:
        return True
    cfg = params["retrain"]
    if not isinstance(cfg, dict):
        logger.error("'retrain' must be a dict.")
        return False

    valid = True
    if "interactive" not in cfg:
        logger.error("'retrain.interactive' (bool) is required.")
        valid = False
    elif not isinstance(cfg["interactive"], bool):
        logger.error("'retrain.interactive' must be bool.")
        valid = False

    if "features" not in cfg:
        logger.error("'retrain.features' is required.")
        valid = False
    else:
        feats = cfg["features"]
        ok = (
            feats == "__auto__"
            or isinstance(feats, dict)
            or (
                isinstance(feats, list)
                and (
                    feats == ["__auto__"]
                    or all(isinstance(x, str) for x in feats)
                )
            )
        )
        if not ok:
            logger.error(
                "'retrain.features' must be '__auto__', list[str], or dict."
            )
            valid = False

    interactive = cfg.get("interactive")
    if interactive is False:
        if "threshold" not in cfg:
            logger.error("'retrain.threshold' (float) is required when interactive=false.")
            valid = False
        elif not isinstance(cfg["threshold"], (int, float)):
            logger.error("'retrain.threshold' must be a float.")
            valid = False
        if "max_iter" not in cfg:
            logger.error("'retrain.max_iter' (int > 0) is required when interactive=false.")
            valid = False
        elif not isinstance(cfg["max_iter"], int) or cfg["max_iter"] <= 0:
            logger.error("'retrain.max_iter' must be an int > 0.")
            valid = False

    if "num_worst_features" in cfg:
        n = cfg["num_worst_features"]
        if not isinstance(n, int) or n <= 0:
            logger.error("'retrain.num_worst_features' must be an int > 0.")
            valid = False

    if "regimes_df" in cfg and cfg["regimes_df"] is not None:
        spec = cfg["regimes_df"]
        ok_spec = (
            isinstance(spec, (str, pl.DataFrame))
            or (isinstance(spec, dict) and ("path" in spec or "End" in spec))
            or (isinstance(spec, list) and all(isinstance(r, dict) for r in spec))
        )
        if not ok_spec:
            logger.error(
                "'retrain.regimes_df' must be null, a path string, "
                "{path: ...}, list of records, or DataFrame."
            )
            valid = False

    return valid
