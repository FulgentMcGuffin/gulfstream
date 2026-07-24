from __future__ import annotations
"""
Regime detection using ruptures methods other than binseg.
"""
import numpy as np
import polars as pl
import logging
import ruptures as rpt

from gulfstream.common import frames
from gulfstream.common import utils
from gulfstream.common.results import AlgoResults
from gulfstream.detectors import common_validation as common

logger = logging.getLogger(__name__)


def _find_bkpts_by_pelt(
    df: pl.DataFrame,
    ruptures_cost_params: dict,
    ruptures_penalty: float = None,
    **kwargs
) -> list[int]:
    """
    Pruned exact linear time algorithm for finding breakpoints.

    Parameters
    ----------
    ruptures_cost_params : dict, optional
        Parameters for cost function. Must contain key 'model' specifying the cost function to use.
        Valid choices are 'rbf', 'l2', and 'l1'. Then additional parameters for the specified cost.
        Currently, only cost with additional params is 'rbf', which requires entry 'gamma' (float)
        for the RBF kernel bandwidth.
    ruptures_penalty : float, optional
        Penalty per breakpoint. Will use log(len(df))*stdev(df)**2 if not provided.

    Notes
    -----
    See here for details: https://centre-borelli.github.io/ruptures-docs/user-guide/detection/pelt/
    """
    signal = frames.to_numpy(df)

    # Get parameters to pass to ruptures.
    required_params = {
        'rbf': ['gamma'],
        'l1': [],
        'l2': []
    }
    model = ruptures_cost_params['model']
    if model not in required_params:
        raise ValueError(f"Unknown ruptures cost function {model}.")
    missing_params = [param for param in required_params[model] if param not in ruptures_cost_params]
    if missing_params:
        raise ValueError(f"Missing required parameters for cost function {model}: {missing_params}.")
    filtered_params = {key: value for key, value in ruptures_cost_params.items() if key in required_params[model]}
    algo_instance = rpt.Pelt(model=model, params=filtered_params)

    ruptures_penalty = ruptures_penalty if ruptures_penalty is not None else np.log(len(signal)) * np.std(signal) ** 2
    bkpts = algo_instance.fit_predict(signal, pen=ruptures_penalty)
    # ruptures sometimes returns an endpoint, and it causes problems down the line
    # since these aren't breakpoints.
    bkpts = [bkpt for bkpt in bkpts if bkpt > 0 and bkpt < df.height - 1]
    return bkpts


def _find_bkpts_by_window(
    df: pl.DataFrame,
    ruptures_window: int,
    ruptures_cost_params: dict,
    regimes: int = None,
    ruptures_penalty: float = None,
    ruptures_rec_error: float = None,
    **kwargs
) -> list[int]:
    """
    Find breakpoints by sliding two adjacent windows along a time series, and detecting
    differences in the samples within the two windows.

    Parameters
    ----------
    ruptures_window : int
        Size of sliding window.
    ruptures_cost_params : dict, optional
        Parameters for cost function. Must contain key 'model' specifying the cost function to use.
        Valid choices are 'rbf', 'l2', and 'l1'. Then additional parameters for the specified cost.
        Currently, only cost with additional params is 'rbf', which requires entry 'gamma' (float)
        for the RBF kernel bandwidth.
    regimes : int, optional
        Number of regimes to find. Will use penalty or reconstruction error if not provided.
    ruptures_penalty : float, optional
        Penalty per breakpoint. Will use regimes or reconstruction error if not provided.
    ruptures_rec_error : float, optional
        Reconstruction error. Continue placing breakpoints until error falls below
        this value. Will use regimes or penalty if not provided.

    Notes
    -----
    See here for details: https://centre-borelli.github.io/ruptures-docs/user-guide/detection/window/
    """
    num_not_none = sum(x is not None for x in (regimes, ruptures_penalty, ruptures_rec_error))
    if num_not_none == 0:
        raise ValueError("One of regimes, ruptures_penalty, and ruptures_rec_error must be provided.")
    elif num_not_none > 1:
        raise ValueError("Only one of regimes, ruptures_penalty, and ruptures_rec_error should be provided.")

    signal = frames.to_numpy(df)

    # Get parameters to pass to ruptures.
    required_params = {
        'rbf': ['gamma'],
        'l1': [],
        'l2': []
    }
    model = ruptures_cost_params['model']
    if model not in required_params:
        raise ValueError(f"Unknown ruptures cost function {model}.")
    missing_params = [param for param in required_params[model] if param not in ruptures_cost_params]
    if missing_params:
        raise ValueError(f"Missing required parameters for cost function {model}: {missing_params}.")
    filtered_params = {key: value for key, value in ruptures_cost_params.items() if key in required_params[model]}
    algo_instance = rpt.Window(width=ruptures_window, model=model, params=filtered_params)

    num_bkpts = regimes - 1 if regimes is not None else None
    bkpts = algo_instance.fit_predict(signal, n_bkps=num_bkpts, pen=ruptures_penalty, epsilon=ruptures_rec_error)
    # ruptures sometimes returns an endpoint, and it causes problems down the line
    # since these aren't breakpoints.
    bkpts = [bkpt for bkpt in bkpts if bkpt > 0 and bkpt < df.height - 1]
    return bkpts


def _find_bkpts_by_dynp(
    df: pl.DataFrame,
    regimes: int,
    ruptures_cost_params: dict = None,
    **kwargs
) -> list[int]:
    """
    Find optimal set of breakpoints by dynamic programming.

    Parameters
    ----------
    ruptures_cost_params : dict, optional
        Parameters for cost function. Must contain key 'model' specifying the cost function to use.
        Valid choices are 'rbf', 'l2', and 'l1'. Then additional parameters for the specified cost.
        Currently, only cost with additional params is 'rbf', which requires entry 'gamma' (float)
        for the RBF kernel bandwidth.

    Notes
    -----
    Returned set of breakpoints is exactly optimal for the specified cost function and number of
    breakpoints, but is not stable upon adding additional breakpoints. See here for details:
    https://centre-borelli.github.io/ruptures-docs/user-guide/detection/dynp/
    """
    signal = frames.to_numpy(df)

    # Get parameters to pass to ruptures.
    required_params = {
        'rbf': ['gamma'],
        'l1': [],
        'l2': []
    }
    model = ruptures_cost_params['model']
    if model not in required_params:
        raise ValueError(f"Unknown ruptures cost function {model}.")
    missing_params = [param for param in required_params[model] if param not in ruptures_cost_params]
    if missing_params:
        raise ValueError(f"Missing required parameters for cost function {model}: {missing_params}.")
    filtered_params = {key: value for key, value in ruptures_cost_params.items() if key in required_params[model]}
    algo_instance = rpt.Dynp(model=model, params=filtered_params)
    bkpts = algo_instance.fit_predict(signal, n_bkps=regimes - 1)
    # ruptures sometimes returns an endpoint, and it causes problems down the line
    # since these aren't breakpoints.
    bkpts = [bkpt for bkpt in bkpts if bkpt > 0 and bkpt < df.height - 1]
    return bkpts


def ruptures_predict_regimes(
    df: pl.DataFrame,
    ruptures_algorithm: str,
    **kwargs
) -> AlgoResults:
    """
    Predict regimes using ruptures.

    Parameters
    ----------
    ruptures_algorithm : str
        Supported options are 'pelt', 'window', and 'dynp'.
    kwargs : dict
        Dictionary of parameters for whichever ruptures method you are using.
    """
    algo_dispatch = {
        'pelt': _find_bkpts_by_pelt,
        'window': _find_bkpts_by_window,
        'dynp': _find_bkpts_by_dynp
    }
    handler = algo_dispatch.get(ruptures_algorithm.lower())
    if not handler:
        raise ValueError(
            f"Unknown ruptures algorithm {ruptures_algorithm}. "
            f"Known methods are {', '.join(algo_dispatch.keys())}."
        )
    bkpts = handler(df, **kwargs)
    labels = utils._convert_breakpoints_to_labels(df.height, bkpts)
    return AlgoResults(bkpts=bkpts, labels=labels)


def _pelt_param_generator(params: dict):
    if 'pelt' in params['algo'].get('ruptures_algorithm', []):
        for ruptures_cost_params in params['algo']['ruptures_cost_params']:
            for ruptures_penalty in params['algo']['ruptures_penalty']:
                yield {
                    'regime_detection_algorithm': 'ruptures',
                    'ruptures_algorithm': 'pelt',
                    'ruptures_cost_params': ruptures_cost_params,
                    'ruptures_penalty': ruptures_penalty
                }


def _window_param_generator(params: dict):
    stopping_conditions = ['regimes', 'ruptures_penalty', 'ruptures_rec_error']
    if 'window' in params['algo'].get('ruptures_algorithm', []):
        for window in params['algo']['ruptures_window']:
            for ruptures_cost_params in params['algo']['ruptures_cost_params']:
                for cond in stopping_conditions:
                    for choice in params['algo'].get(cond, []):
                        yield {
                            'regime_detection_algorithm': 'ruptures',
                            'ruptures_algorithm': 'window',
                            'ruptures_window': window,
                            'ruptures_cost_params': ruptures_cost_params,
                            cond: choice
                        }


def _dynp_param_generator(params: dict):
    if 'dynp' in params['algo'].get('ruptures_algorithm', []):
        for regimes in params['algo']['regimes']:
            for ruptures_cost_params in params['algo']['ruptures_cost_params']:
                yield {
                    'regime_detection_algorithm': 'ruptures',
                    'ruptures_algorithm': 'dynp',
                    'ruptures_cost_params': ruptures_cost_params,
                    'regimes': regimes
                }


def ruptures_param_generator(params: dict):
    """
    Yield all valid combinations of parameters for ruptures regime detection.
    """
    handlers = [_pelt_param_generator, _window_param_generator, _dynp_param_generator]
    for handler in handlers:
        yield from handler(params)


def ruptures_params_printout() -> dict:
    """
    Dict for writing heading of this algorithm. Each entry will be a
    column heading in an Excel sheet. Values are singleton lists with
    the heading to be used for the associated column. Keys should use
    the format "algoname_paramname".
    """
    return {
        'ruptures_ruptures_algorithm': ['ruptures algorithm'],
        'ruptures_ruptures_cost_params': ['ruptures cost function'],
        'ruptures_ruptures_window': ['rolling window size'],
        'ruptures_regimes': ['number of regimes'],
        'ruptures_ruptures_penalty': ['penalty per breakpoint'],
        'ruptures_ruptures_rec_error': ['reconstruction error']
    }


def _valid_penalty(algo_params: dict) -> bool:
    penalty = algo_params.get('ruptures_penalty')
    if penalty is None:
        logger.error("'ruptures_penalty' (list[float]) must be provided.")
        return False
    elif not isinstance(penalty, list):
        logger.error("'ruptures_penalty' needs to be type list[float]. Got type %s.", type(penalty))
        return False
    elif not all(isinstance(x, (int, float)) and x >= 0 for x in penalty):
        logger.error("All entries of 'ruptures_penalty' must be non-negative floats.")
        return False
    return True


def _valid_rec_error(algo_params: dict) -> bool:
    ruptures_rec_error = algo_params.get('ruptures_rec_error')
    if ruptures_rec_error is None:
        logger.error("'ruptures_rec_error' (list[float]) must be provided.")
        return False
    elif not isinstance(ruptures_rec_error, list):
        logger.error("'ruptures_rec_error' must be type list[float]. Got type %s.", type(ruptures_rec_error))
        return False
    elif not all(isinstance(x, (int, float)) and x > 0 for x in ruptures_rec_error):
        logger.error("All entries of 'ruptures_rec_error' must be positive floats.")
        return False
    return True


def _valid_rbf_ruptures(ruptures_cost_params: dict) -> bool:
    gamma = ruptures_cost_params.get('gamma')
    if gamma is None:
        logger.error("'gamma' (float) must be provided when using 'rbf' cost for 'ruptures_cost_params'.")
        return False
    elif not isinstance(gamma, (int, float)) or gamma < 0:
        logger.error("'gamma' for RBF kernel bandwidth for 'ruptures_cost_params' must be non-negative float.")
        return False
    return True


def _valid_no_extra_param_ruptures_cost(ruptures_cost_params: dict) -> bool:
    return True


def _valid_ruptures_cost_params_single(ruptures_cost_params: dict) -> bool:
    validators = {
        'rbf': _valid_rbf_ruptures,
        'l1': _valid_no_extra_param_ruptures_cost,
        'l2': _valid_no_extra_param_ruptures_cost
    }
    valid = True
    model = ruptures_cost_params.get('model')
    if model is None:
        logger.error("Each dict in 'ruptures_cost_params' must include entry 'model' (str).")
        valid = False
    elif model not in validators:
        logger.error("Unknown ruptures model %s.", model)
        valid = False
    else:
        if not validators[model](ruptures_cost_params):
            valid = False
    return valid


def _valid_ruptures_cost_params(algo_params: dict) -> bool:
    costs = algo_params.get('ruptures_cost_params')
    if costs is None:
        logger.error("'ruptures_cost_params' (list[dict]) must be specified.")
        return False
    elif not isinstance(costs, list):
        logger.error("'ruptures_cost_params' must be type list[dict]. Got type %s.", type(costs))
        return False
    elif not all(isinstance(d, dict) for d in costs):
        logger.error("All entries of 'ruptures_cost_params' must be dicts.")
        return False
    valid = True
    for d in costs:
        if not _valid_ruptures_cost_params_single(d):
            valid = False
    return valid


def _valid_window(algo_params: dict) -> bool:
    window = algo_params.get('ruptures_window')
    if window is None:
        logger.error(
            "'ruptures_window' (list[int]) in 'algo' must be provided "
            "for 'window' for 'ruptures_algorithm'."
        )
        return False
    elif not isinstance(window, list):
        logger.error("'ruptures_window' in 'algo' must be type list[int]. Got type %s.", type(window))
        return False
    elif not all(isinstance(x, int) and x > 0 for x in window):
        logger.error("All entries of 'ruptures_window' must be positive integers.")
        return False
    return True


def _valid_pelt_params(algo_params: dict) -> bool:
    valid = True
    if not _valid_ruptures_cost_params(algo_params):
        valid = False
    if not _valid_penalty(algo_params):
        valid = False
    return valid


def _valid_window_params(algo_params: dict) -> bool:
    regimes = algo_params.get('regimes')
    ruptures_penalty = algo_params.get('ruptures_penalty')
    ruptures_rec_error = algo_params.get('ruptures_rec_error')
    if regimes is None and ruptures_penalty is None and ruptures_rec_error is None:
        logger.error(
            "At least one of 'regimes', 'ruptures_penalty', or 'ruptures_rec_error' "
            "must be provided to use 'window' for 'ruptures_algorithm'."
        )
        return False
    valid = True
    if regimes is not None:
        if not common._valid_regimes(algo_params):
            valid = False
    if ruptures_penalty is not None:
        if not _valid_penalty(algo_params):
            valid = False
    if ruptures_rec_error is not None:
        if not _valid_rec_error(algo_params):
            valid = False
    if not _valid_ruptures_cost_params(algo_params):
        valid = False
    if not _valid_window(algo_params):
        valid = False
    return valid


def _valid_dynp_params(algo_params: dict) -> bool:
    valid = True
    if not common._valid_regimes(algo_params):
        valid = False
    if not _valid_ruptures_cost_params(algo_params):
        valid = False
    return valid


def ruptures_input_validator(algo_params: dict) -> bool:
    validators = {
        'pelt': _valid_pelt_params,
        'window': _valid_window_params,
        'dynp': _valid_dynp_params
    }
    valid = True
    algo = algo_params.get('ruptures_algorithm')
    if algo is None:
        logger.error("'ruptures_algorithm' (list[str]) must be specified for ruptures.")
        valid = False
    elif not isinstance(algo, list):
        logger.error("'ruptures_algorithm' must be type list[str]. Got type %s.", type(algo))
        valid = False
    elif not all(x in validators for x in algo):
        logger.error(
            "At least one unknown choice of 'ruptures_algorithm' provided. "
            "Valid options are %s.", ', '.join(validators.keys())
        )
        valid = False
    elif len(algo) == 0:
        logger.error("'ruptures_algorithm' must be a nonempty list.")
        valid = False

    if valid:
        for a in algo:
            if a in validators:
                if not validators[a](algo_params):
                    valid = False
    return valid
