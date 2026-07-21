"""Input validation for stability section."""
import logging

logger = logging.getLogger(__name__)


def _valid_stability_params(params: dict) -> bool:
    cfg = params.get("stability")
    if cfg is None:
        return True
    if not isinstance(cfg, dict):
        logger.error("'stability' must be a dict.")
        return False
    valid = True
    if "enabled" in cfg and not isinstance(cfg["enabled"], bool):
        logger.error("'stability.enabled' must be bool.")
        valid = False
    if "match_tolerance" in cfg and (
        not isinstance(cfg["match_tolerance"], int) or cfg["match_tolerance"] < 0
    ):
        logger.error("'stability.match_tolerance' must be a non-negative int.")
        valid = False
    if "stability_floor" in cfg:
        t = cfg["stability_floor"]
        if not isinstance(t, (int, float)) or not (0 <= t <= 1):
            logger.error("'stability.stability_floor' must be in [0, 1].")
            valid = False
    return valid
