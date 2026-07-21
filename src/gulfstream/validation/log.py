"""Input validation for the 'log' section of the parameters dict."""
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _provided_log(params: dict) -> bool:
    log = params.get('log')
    if not log:
        logger.error("Must provide 'log' (dict).")
        return False
    elif not isinstance(log, dict):
        logger.error("'log' must be type dict. Got type %s.", type(log))
        return False
    return True


def _valid_dir(params: dict) -> bool:
    my_dir = params['log'].get('dir')
    if not my_dir:
        logger.error("Must specify 'dir' (str) in 'log'.")
        return False
    elif not isinstance(my_dir, str):
        logger.error("'dir' in 'log' must be type str. Got type %s.", type(my_dir))
        return False
    try:
        # Basic cross-platform sanity check.
        p = Path(my_dir)
        return bool(p.name or p.root)
    except:
        return False


def _valid_level(params: dict) -> bool:
    level = params['log'].get('level')
    if not level:
        logger.error("Must specify 'level' (str) in 'log'.")
        return False
    elif not isinstance(level, str):
        logger.error("'level' in 'log' must be type str. Got type %s.", type(level))
        return False
    valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    if level.upper() not in valid_levels:
        logger.error("'level' in 'log' must be one of %s.", ', '.join(valid_levels))
        return False
    return True


def _valid_log_params(params: dict) -> bool:
    if not _provided_log(params):
        return False
    valid = True
    if not _valid_dir(params):
        valid = False
    if not _valid_level(params):
        valid = False
    return valid
