"""Input validation for algo parameters."""
import logging
from . import common

logger = logging.getLogger(__name__)


def _valid_recursive_method(params: dict) -> bool:
    if not common._is_nonempty_list(params['algo'], 'recursive_method', str):
        return False
    valid = True
    for method in params['algo']['recursive_method']:
        if method not in ['full', 'iterative_pca']:
            logger.error("Unknown 'recursive_method' %s.", method)
            valid = False
    return valid


def _valid_depth(params: dict) -> bool:
    if not common._is_nonempty_list(params['algo'], 'depth', int):
        return False
    valid = True
    for depth in params['algo']['depth']:
        if depth < 1:
            logger.error("Parameter 'depth' must be positive. Got %s.", depth)
            valid = False
    return valid


def _valid_ruptures_kernel_params(params: dict) -> bool:
    if not common._is_nonempty_list(params['algo'], 'ruptures_kernel_params', dict):
        return False
    valid = True
    valid_kernels = ['rbf', 'linear', 'cosine']
    valid_gamma_methods = ['user_specified', 'median', 'sk_scale']
    for kernel_params in params['algo']['ruptures_kernel_params']:
        kernel = kernel_params.get('kernel')
        if not kernel:
            logger.error("Every dict in 'ruptures_kernel_params' must specify parameter 'kernel'.")
            valid = False
        elif kernel not in valid_kernels:
            logger.error(
                "A dict in 'ruptures_kernel_params' has unknown 'kernel': %s. Valid kernels are %s.",
                kernel, ', '.join(valid_kernels)
            )
            valid = False
        elif kernel == 'rbf':
            gamma_method = kernel_params.get('gamma_method')
            gamma = kernel_params.get('gamma')
            if not gamma_method:
                logger.error(
                    "A dict in 'ruptures_kernel_params' using kernel 'rbf' is missing "
                    "parameter 'gamma_method' (str)."
                )
                valid = False
            elif gamma_method not in valid_gamma_methods:
                logger.error(
                    "A dict in 'ruptures_kernel_params' using kernel 'rbf' has unknown method "
                    "%s for parameter 'gamma_method'. Options are %s.",
                    gamma_method, ', '.join(valid_gamma_methods)
                )
                valid = False
            elif gamma_method == 'user_specified':
                if not gamma:
                    logger.error(
                        "A dict in 'ruptures_kernel_params' using kernel 'rbf' and "
                        "'gamma_method' == 'user_specified' failed to provide parameter 'gamma' (float)."
                    )
                    valid = False
                elif not isinstance(gamma, (int, float)):
                    logger.error(
                        "A dict in 'ruptures_kernel_params' using kernel 'rbf' and "
                        "'gamma_method' == 'user_specified' provided parameter 'gamma' of "
                        "invalid type. Must be float."
                    )
                    valid = False
            else:
                if gamma:
                    logger.warning(
                        "A dict in 'ruptures_kernel_params' using kernel 'rbf' and "
                        "'gamma_method' != 'user_specified' provided extra parameter 'gamma'. "
                        "Will be ignored."
                    )
    return valid


def _valid_majority_voting(params: dict) -> bool:
    valid = True
    min_reg_lens = params['algo'].get('min_regime_length')
    if not min_reg_lens:
        logger.error(
            "'min_regime_length' (list[int]) must be provided if 'majority_voting' "
            "is provided to 'post_processing_method'."
        )
        valid = False
    elif not isinstance(min_reg_lens, list):
        logger.error(
            "'min_regime_length' must be a nonempty list of positive integers if "
            "'majority_voting' is provided to 'post_processing_method'."
        )
        valid = False
    elif not all(x > 0 for x in min_reg_lens):
        logger.error("All entries of 'min_regime_length' must be positive integers.")
        valid = False
    include_last = params['algo'].get('include_last_regime')
    if not include_last:
        logger.error(
            "'include_last_regime' (list[bool]) must be provided if 'majority_voting' "
            "is provided to 'post_processing_method'."
        )
        valid = False
    elif not isinstance(include_last, list):
        logger.error(
            "'include_last_regime' must be a nonempty list of bool if 'majority_voting' "
            "is provided to 'post_processing_method'."
        )
        valid = False
    elif not all(isinstance(x, bool) for x in include_last):
        logger.error("All entries of 'include_last' must be True or False.")
        valid = False
    return valid


def _valid_neighbor_comparison(params: dict) -> bool:
    valid = True
    min_reg_lens = params['algo'].get('min_regime_length')
    if not min_reg_lens:
        logger.error(
            "'min_regime_length' (list[int]) must be provided if 'neighbor_comparison' "
            "is provided to 'post_processing_method'."
        )
        valid = False
    elif not isinstance(min_reg_lens, list):
        logger.error(
            "'min_regime_length' must be a nonempty list of positive integers if "
            "'neighbor_comparison' is provided to 'post_processing_method'."
        )
        valid = False
    elif not all(x > 0 for x in min_reg_lens):
        logger.error("All entries of 'min_regime_length' must be positive integers.")
        valid = False
    return valid


def _valid_entropy_post_processing(params: dict) -> bool:
    valid = True
    min_reg_lens = params['algo'].get('min_regime_length')
    if not min_reg_lens:
        logger.error(
            "'min_regime_length' (list[int]) must be provided if 'entropy' "
            "is provided to 'post_processing_method'."
        )
        valid = False
    elif not isinstance(min_reg_lens, list):
        logger.error(
            "'min_regime_length' must be a nonempty list of positive integers if "
            "'entropy' is provided to 'post_processing_method'."
        )
        valid = False
    elif not all(x > 0 for x in min_reg_lens):
        logger.error("All entries of 'min_regime_length' must be positive integers.")
        valid = False
    include_last = params['algo'].get('include_last_regime')
    if not include_last:
        logger.error(
            "'include_last_regime' (list[bool]) must be provided if 'entropy' "
            "is provided to 'post_processing_method'."
        )
        valid = False
    elif not isinstance(include_last, list):
        logger.error(
            "'include_last_regime' must be a nonempty list of bool if 'entropy' "
            "is provided to 'post_processing_method'."
        )
        valid = False
    elif not all(isinstance(x, bool) for x in include_last):
        logger.error("All entries of 'include_last' must be True or False.")
        valid = False
    window = params['algo'].get('entropy_window')
    if not window:
        logger.error(
            "'entropy_window' (list[int]) must be provided if 'entropy' "
            "is provided to 'post_processing_method'."
        )
        valid = False
    elif not isinstance(window, list):
        logger.error(
            "'entropy_window' must be a nonempty list of positive integers if "
            "'entropy' is provided to 'post_processing_method'."
        )
        valid = False
    elif not all(isinstance(x, int) and x > 0 for x in window):
        logger.error("All entries of 'entropy_window' must be positive integers.")
        valid = False
    return valid


def _valid_interval_overlap(params: dict) -> bool:
    return True


def _valid_no_post_processing(params: dict) -> bool:
    return True


# TODO: Should add a condition to this to check for a test that uses a lag.
# Need to allow it as long as at least one of those tests appears.
def _valid_lag_interval_post_processing(params: dict) -> bool:
    return True


def _valid_post_processing_params(params: dict) -> bool:
    if not common._is_nonempty_list(params['algo'], 'post_processing_method', str):
        return False
    valid = True
    valid_methods = {
        'majority_voting': _valid_majority_voting,
        'interval_overlap': _valid_interval_overlap,
        'interval_overlap_efficient': _valid_interval_overlap,
        'no_post_processing': _valid_no_post_processing,
        'neighbor_comparison': _valid_neighbor_comparison,
        'lag_interval': _valid_lag_interval_post_processing,
        'entropy': _valid_entropy_post_processing
    }
    for method in params['algo']['post_processing_method']:
        if method not in valid_methods:
            logger.error(
                "Unknown post_processing_method %s. Known methods are %s.",
                method, ', '.join(valid_methods.keys())
            )
            valid = False
        elif not valid_methods[method](params):
            valid = False
    return valid


def _valid_algo_params(params: dict) -> bool:
    if not common._provided_algo(params):
        return False
    validators = [
        _valid_recursive_method,
        _valid_depth,
        _valid_ruptures_kernel_params,
        _valid_post_processing_params
    ]
    valid = True
    for validator in validators:
        if not validator(params):
            valid = False
    return valid
