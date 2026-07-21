"""Shared input validators for legacy regime detection methods."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _valid_regimes(algo_params: dict) -> bool:
    """Validate ``algo_params['regimes']`` is a nonempty list of positive ints."""
    regimes = algo_params.get("regimes")
    if regimes is None:
        logger.error("'regimes' (list[int]) must be specified.")
        return False
    if not isinstance(regimes, list):
        logger.error("'regimes' must be type list[int]. Got type %s.", type(regimes))
        return False
    if len(regimes) == 0:
        logger.error("'regimes' must be a nonempty list.")
        return False
    if not all(isinstance(x, int) and x > 0 for x in regimes):
        logger.error("All entries of 'regimes' must be positive integers.")
        return False
    return True
