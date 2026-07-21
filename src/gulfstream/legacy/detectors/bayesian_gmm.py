from __future__ import annotations
"""
Regime detection using Bayesian Gaussian mixture model.
"""
import polars as pl
import logging
from sklearn.mixture import BayesianGaussianMixture

from gulfstream.common import frames
from gulfstream.common import utils
from gulfstream.common.results import AlgoResults
from gulfstream.legacy.detectors import common_validation as common

logger = logging.getLogger(__name__)


def bayesian_gmm_predict_regimes(df: pl.DataFrame, regimes: int, reg_covar: float = 1e-5, **kwargs) -> AlgoResults:
    """
    Fit BayesianGMM to df to predict regimes.

    Parameters
    ----------
    regimes : int
        Number of distributions in the mixture. Usually the number of regimes, but
        it is possible the model finds fewer regimes.
    reg_covar : float, optional
        Regularization for the diagonal of covariance matrices. Default is 1e-5.

    Returns
    -------
    AlgoResults
        Container containing the breakpoints and labels.
    """
    bgmm = BayesianGaussianMixture(
        n_components=regimes,
        covariance_type='full',
        random_state=42,
        reg_covar=reg_covar
    )
    bgmm.fit(frames.to_numpy(df))
    labels = utils._map_labels_to_ordered_integers(bgmm.predict(frames.to_numpy(df)))
    bkpts = utils._convert_labels_to_bkpts(labels)
    return AlgoResults(bkpts=bkpts, labels=labels)


def bayesian_gmm_param_generator(params: dict):
    """
    Yield all valid combinations of parameters for BayesianGMM regime detection.
    """
    if 'bayesian_gmm' in params['algo']['regime_detection_algorithm']:
        for regimes in params['algo']['regimes']:
            for reg_covar in params['algo'].get('reg_covar', [1e-5]):
                yield {
                    'regime_detection_algorithm': 'bayesian_gmm',
                    'regimes': regimes,
                    'reg_covar': reg_covar
                }


def bayesian_gmm_params_printout() -> dict:
    """
    Dict for writing heading of this algorithm. Each entry will be a
    column heading in an Excel sheet. Values are singleton lists with
    the heading to be used for the associated column. Keys should use
    the format "algoname_paramname".
    """
    return {
        'bayesian_gmm_regimes': ['maximum number of regimes'],
        'bayesian_gmm_reg_covar': ['regularization for covariance matrices']
    }


def _valid_reg_covar(algo_params: dict) -> bool:
    reg_covar = algo_params.get('reg_covar')
    if reg_covar is None:
        return True  # Optional parameter.
    elif not isinstance(reg_covar, list):
        logger.error("'reg_covar' in 'algo' must be type list[float]. Got type %s.", type(reg_covar))
        return False
    elif not all(isinstance(x, (int, float)) and x >= 0 for x in reg_covar):
        logger.error("All entries of 'reg_covar' must be non-negative floats.")
        return False
    return True


def bayesian_gmm_input_validator(algo_params: dict) -> bool:
    """
    Returns True if algo_params contains a valid set of parameters
    for Bayesian GMM regime detection. May contain additional
    parameters as well.
    """
    valid = True
    if not common._valid_regimes(algo_params):
        valid = False
    if not _valid_reg_covar(algo_params):
        valid = False
    return valid
