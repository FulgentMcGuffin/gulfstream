"""Input validation for parameters related to mapping a time series by a kernel feature map."""
import logging
from . import common

logger = logging.getLogger(__name__)

MAPPING_METHODS = ['rff', 'nystroem', 'inv_rff', 'raw']


def _provided_feature_map_approx_method(params: dict) -> bool:
    if not common._provided_algo(params):
        return False
    algo = params['algo']
    methods = algo.get('feature_map_approx_method')
    if not methods:
        logger.error("'feature_map_approx_method' must be provided.")
        return False
    elif not isinstance(methods, list):
        logger.error("'feature_map_approx_method' must be a list.")
        return False
    elif len(methods) == 0:
        logger.error("'feature_map_approx_method' must be a nonempty list.")
        return False
    invalid_methods = [x for x in methods if x not in MAPPING_METHODS]
    if invalid_methods:
        invalid_methods = list(set(invalid_methods))
        logger.error("Invalid 'feature_map_approx_method' entries: %s.", ', '.join(invalid_methods))
        return False
    return True


def _using_rff(params: dict) -> bool:
    if not _provided_feature_map_approx_method(params):
        return False
    if 'rff' in params['algo']['feature_map_approx_method']:
        return True
    return False


def _using_inv_rff(params: dict) -> bool:
    if not _provided_feature_map_approx_method(params):
        return False
    if 'inv_rff' in params['algo']['feature_map_approx_method']:
        return True
    return False


def _using_nystroem(params: dict) -> bool:
    if not _provided_feature_map_approx_method(params):
        return False
    if 'nystroem' in params['algo']['feature_map_approx_method']:
        return True
    return False


def _valid_num_features(params: dict) -> bool:
    num_features = params['algo'].get('num_features')
    kernel_approx_error = params['algo'].get('kernel_approx_error')
    if not num_features and not kernel_approx_error:
        logger.error("Must specify at least one of 'num_features' or 'kernel_approx_error'.")
        return False
    valid = True
    if num_features:
        if common._is_nonempty_list(params['algo'], 'num_features', int):
            if not all(x > 0 for x in params['algo']['num_features']):
                logger.error("All entries of 'num_features' must be positive integers.")
                valid = False
        else:
            valid = False
    if kernel_approx_error:
        if common._is_nonempty_list(params['algo'], 'kernel_approx_error', (int, float)):
            if not all(x > 0 for x in params['algo']['kernel_approx_error']):
                logger.error("All entries of 'kernel_approx_error' must be positive.")
                valid = False
        else:
            valid = False
    return valid


def _valid_num_mappings(params: dict) -> bool:
    num_mappings = params['algo'].get('num_mappings')
    if not num_mappings:
        logger.error("Must specify 'num_mappings'.")
        return False
    valid = True
    if common._is_nonempty_list(params['algo'], 'num_mappings', int):
        if not all(x > 0 for x in params['algo']['num_mappings']):
            logger.error("All entries of 'num_mappings' must be positive integers.")
            valid = False
    else:
        valid = False
    return valid


def _valid_rbf_kernel_params(kernel_params: dict) -> bool:
    """Return True if kernel_params is a valid set of kernel parameters for RBF kernel."""
    valid = True
    gamma_method = kernel_params.get('gamma_method')
    gamma = kernel_params.get('gamma')
    if not gamma_method:
        if not gamma:
            logger.error(
                "If 'gamma_method' is not specified for RBF kernel for a dict in "
                "'feature_map_kernel_params', then 'gamma' must be provided."
            )
            valid = False
        elif not isinstance(gamma, (int, float)):
            logger.error(
                "Parameter 'gamma' for 'rbf' kernel for a dict in "
                "'feature_map_kernel_params' must be type float."
            )
            valid = False
    elif gamma_method == 'user_specified':
        if not gamma:
            logger.error(
                "If 'gamma_method' is 'user_specified' for a dict in "
                "'feature_map_kernel_params', then 'gamma' must be provided."
            )
            valid = False
        elif not isinstance(gamma, (int, float)):
            logger.error(
                "Parameter 'gamma' for 'rbf' kernel for a dict in "
                "'feature_map_kernel_params' must be type float."
            )
            valid = False
    elif gamma_method not in ['median', 'sk_scale']:
        logger.error(
            "Unknown 'gamma_method' %s for a dict in 'feature_map_kernel_params'.", gamma_method
        )
        valid = False
    else:
        if gamma:
            logger.warning(
                "A dict in 'feature_map_kernel_params' provided 'gamma_method' != "
                "'user_specified', so the provided 'gamma' will be ignored."
            )
    return valid


def _valid_laplacian_kernel_params(kernel_params: dict) -> bool:
    gamma = kernel_params.get('gamma')
    if not gamma:
        logger.error(
            "A dict in 'feature_map_kernel_params' with kernel 'laplacian' is "
            "missing parameter 'gamma' (float)."
        )
        return False
    elif not isinstance(gamma, (int, float)):
        logger.error(
            "A dict in 'feature_map_kernel_params' with kernel 'laplacian' has "
            "parameter 'gamma' of incorrect type. Must be float."
        )
        return False
    return True


def _valid_chi2_kernel_params(kernel_params: dict) -> bool:
    gamma = kernel_params.get('gamma')
    if not gamma:
        logger.error(
            "A dict in 'feature_map_kernel_params' with kernel 'chi2' is "
            "missing parameter 'gamma' (float)."
        )
        return False
    elif not isinstance(gamma, (int, float)):
        logger.error(
            "A dict in 'feature_map_kernel_params' with kernel 'chi2' has "
            "parameter 'gamma' of incorrect type. Must be float."
        )
        return False
    return True


def _valid_poly_kernel_params(kernel_params: dict) -> bool:
    params_to_check = {
        'degree': kernel_params.get('degree'),
        'coef0': kernel_params.get('coef0'),
        'gamma': kernel_params.get('gamma')
    }
    valid = True
    for key, val in params_to_check.items():
        if not val:
            logger.error(
                "A dict in 'feature_map_kernel_params' with kernel 'poly' is "
                "missing parameter %s (float).", key
            )
            valid = False
        elif not isinstance(val, (int, float)):
            logger.error(
                "A dict in 'feature_map_kernel_params' with kernel 'poly' has "
                "parameter %s of incorrect type. Must be float.", key
            )
            valid = False
    return valid


def _valid_sigmoid_kernel_params(kernel_params: dict) -> bool:
    params_to_check = {
        'gamma': kernel_params.get('gamma'),
        'coef0': kernel_params.get('coef0')
    }
    valid = True
    for key, val in params_to_check.items():
        if not val:
            logger.error(
                "A dict in 'feature_map_kernel_params' with kernel 'sigmoid' is "
                "missing parameter %s (float).", key
            )
            valid = False
        elif not isinstance(val, (int, float)):
            logger.error(
                "A dict in 'feature_map_kernel_params' with kernel 'sigmoid' has "
                "parameter %s of incorrect type. Must be float.", key
            )
            valid = False
    return valid


def _valid_no_param_kernel_params(kernel_params: dict) -> bool:
    # Nothing to check for no params.
    return True


def _valid_kernels(params: dict) -> bool:
    kernel_params_list = params['algo'].get('feature_map_kernel_params')
    if not kernel_params_list:
        logger.error("Must provide 'feature_map_kernel_params'.")
        return False
    handlers = {
        'rbf': _valid_rbf_kernel_params,
        'laplacian': _valid_laplacian_kernel_params,
        'chi2': _valid_chi2_kernel_params,
        'poly': _valid_poly_kernel_params,
        'sigmoid': _valid_sigmoid_kernel_params,
        'additive_chi2': _valid_no_param_kernel_params,
        'linear': _valid_no_param_kernel_params,
        'cosine': _valid_no_param_kernel_params
    }
    valid = True
    for kernel_params in params['algo']['feature_map_kernel_params']:
        kernel = kernel_params.get('kernel')
        if not kernel:
            logger.error("A dict in 'feature_map_kernel_params' is missing key 'kernel'.")
            valid = False
        elif kernel not in handlers:
            logger.error(
                "A dict in 'feature_map_kernel_params' has unrecognized 'kernel' %s.", kernel
            )
            valid = False
        else:
            handler = handlers[kernel]
            if not handler(kernel_params):
                valid = False
    return valid


def _valid_kernel_feature_map_params(params: dict) -> bool:
    if not _provided_feature_map_approx_method(params):
        return False
    valid = True
    # Check if user provided necessary kernel params.
    if _using_rff(params) or _using_inv_rff(params):
        kernel_params_list = params['algo'].get('feature_map_kernel_params')
        if not kernel_params_list:
            logger.error(
                "Must provide nonempty list for 'feature_map_kernel_params' if using "
                "'feature_map_approx_method' other than 'raw'."
            )
            valid = False
        else:
            provided_rbf = False
            for kernel_params in kernel_params_list:
                if kernel_params.get('kernel') == 'rbf':
                    provided_rbf = True
                    if not _valid_rbf_kernel_params(kernel_params):
                        valid = False
            if not provided_rbf:
                logger.error(
                    "If using 'rff' or 'inv_rff' for 'feature_map_approx_method', must provide "
                    "at least one entry of 'feature_map_kernel_params' with 'rbf' kernel."
                )
                valid = False
    if _using_nystroem(params):
        if not _valid_kernels(params):
            valid = False
    # Check if user passed data on dimension and number of mappings.
    if _using_rff(params) or _using_inv_rff(params) or _using_nystroem(params):
        if not _valid_num_features(params):
            valid = False
        if not _valid_num_mappings(params):
            valid = False
    return valid
