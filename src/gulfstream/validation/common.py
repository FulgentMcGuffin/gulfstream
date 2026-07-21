"""Functions used in multiple source files for input validation."""
import logging

logger = logging.getLogger(__name__)


def _is_nonempty_list(params: dict, entry: str, dtype: type) -> bool:
    my_entry = params.get(entry)
    if not my_entry:
        logger.error(f"'{entry}' must be provided.")
        return False
    elif not isinstance(my_entry, list):
        logger.error(f"'{entry}' must be a list.")
        return False
    elif not all(isinstance(x, dtype) for x in my_entry):
        logger.error(f"All entries in '{entry}' must be {dtype}.")
        return False
    return True


def _provided_algo(params: dict) -> bool:
    algo = params.get('algo')
    if not algo:
        logger.error("'algo' must be provided.")
        return False
    elif not isinstance(algo, dict):
        logger.error("'algo' must be a dict.")
        return False
    return True
