"""Input validation for robustness section."""
import logging

logger = logging.getLogger(__name__)


def _valid_robustness_params(params: dict) -> bool:
    cfg = params.get("robustness")
    if cfg is None:
        return True
    if not isinstance(cfg, dict):
        logger.error("'robustness' must be a dict.")
        return False
    valid = True
    if "enabled" in cfg and not isinstance(cfg["enabled"], bool):
        logger.error("'robustness.enabled' must be bool.")
        valid = False
    if "match_tolerance" in cfg and (
        not isinstance(cfg["match_tolerance"], int) or cfg["match_tolerance"] < 0
    ):
        logger.error("'robustness.match_tolerance' must be a non-negative int.")
        valid = False
    if "low_persistence_threshold" in cfg:
        t = cfg["low_persistence_threshold"]
        if not isinstance(t, (int, float)) or not (0 <= t <= 1):
            logger.error("'robustness.low_persistence_threshold' must be in [0, 1].")
            valid = False
    return valid
