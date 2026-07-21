from __future__ import annotations
"""
Regime detection using hidden Markov models.
Currently just Gaussian HMM, but could add others.
"""
import polars as pl
import logging
import hmmlearn.hmm as hmm

from gulfstream.common import frames
from gulfstream.common import utils
from gulfstream.common.results import AlgoResults
from gulfstream.legacy.detectors import common_validation as common

logger = logging.getLogger(__name__)


def gaussian_hmm_predict_regimes(df: pl.DataFrame, regimes: int, hmm_n_iter: int = 100, **kwargs) -> AlgoResults:
    """
    Find regimes using an HMM with Gaussian emissions.

    Parameters
    ----------
    regimes : int
        Number of hidden states. Also equal to the number of regimes.

    Returns
    -------
    AlgoResults
        Container containing the breakpoints and labels (params attribute is None).
    """
    model = hmm.GaussianHMM(
        n_components=regimes,
        n_iter=hmm_n_iter,
        covariance_type='full',
        min_covar=1e-2
    )
    model.fit(frames.to_numpy(df))
    hidden_states = model.predict(frames.to_numpy(df))
    labels = utils._map_labels_to_ordered_integers(hidden_states)
    bkpts = utils._convert_labels_to_bkpts(labels)
    return AlgoResults(bkpts=bkpts, labels=labels)


def hmm_predict_regimes(df: pl.DataFrame, hmm_emissions: str, **kwargs) -> AlgoResults:
    """
    Find regimes using HMMs.

    Parameters
    ----------
    kwargs : dict
        Contains the required keyword arguments for whichever HMM method to use.
    """
    hmm_dispatch = {
        'gaussian': gaussian_hmm_predict_regimes
    }
    handler = hmm_dispatch.get(hmm_emissions)
    if not handler:
        raise ValueError(f"Unknown HMM emissions {hmm_emissions}.")
    return handler(df, **kwargs)


def gaussian_hmm_param_generator(params: dict):
    """
    Yield all valid combinations of parameters for GaussianHMM regime detection.
    """
    for regimes in params['algo']['regimes']:
        for hmm_n_iter in params['algo'].get('hmm_n_iter', [100]):
            yield {
                'regime_detection_algorithm': 'hmm',
                'hmm_emissions': 'gaussian',
                'regimes': regimes,
                'hmm_n_iter': hmm_n_iter
            }


def hmm_param_generator(params: dict):
    generators = {
        'gaussian': gaussian_hmm_param_generator
    }
    if 'hmm' in params['algo']['regime_detection_algorithm']:
        for hmm_emissions in params['algo']['hmm_emissions']:
            handler = generators.get(hmm_emissions)
            if not handler:
                raise ValueError(f"Unknown HMM emissions type {hmm_emissions}.")
            yield from handler(params)


def hmm_params_printout() -> dict:
    """
    Dict for writing heading of this algorithm. Each entry will be a
    column heading in an Excel sheet. Values are singleton lists with
    the heading to be used for the associated column. Keys should use
    the format "algoname_paramname".
    """
    return {
        'hmm_hmm_emissions': ['type of emissions'],
        'hmm_regimes': ['number of regimes'],
        'hmm_hmm_n_iter': ['number of iterations']
    }


def _valid_hmm_n_iter(algo_params: dict) -> bool:
    hmm_n_iter = algo_params.get('hmm_n_iter')
    if hmm_n_iter is None:
        return True  # Optional parameter.
    elif not isinstance(hmm_n_iter, list):
        logger.error("'hmm_n_iter' in 'algo' must be type list[int]. Got type %s.", type(hmm_n_iter))
        return False
    elif not all(isinstance(x, int) and x > 0 for x in hmm_n_iter):
        logger.error("All entries of 'hmm_n_iter' must be positive ints.")
        return False
    return True


def gaussian_hmm_input_validator(algo_params: dict) -> bool:
    """
    Returns True if algo_params contains a valid set of parameters
    for Gaussian HMM regime detection. May contain additional
    parameters as well.
    """
    valid = True
    if not common._valid_regimes(algo_params):
        valid = False
    if not _valid_hmm_n_iter(algo_params):
        valid = False
    return valid


def hmm_input_validator(algo_params: dict) -> bool:
    """
    Validates algo_params for all HMM algorithms.
    """
    validators = {
        'gaussian': gaussian_hmm_input_validator
    }
    valid = True
    hmm_emissions = algo_params.get('hmm_emissions')
    if hmm_emissions is None:
        logger.error("'hmm_emissions' (list[str]) must be specified for HMM.")
        valid = False
    elif not isinstance(hmm_emissions, list):
        logger.error("'hmm_emissions' must be type list[str]. Got type %s.", type(hmm_emissions))
        valid = False
    elif not all(x in validators for x in hmm_emissions):
        logger.error("At least one unknown choice of 'hmm_emissions' provided. Valid options are %s.",
                     ', '.join(validators.keys()))
        valid = False
    elif len(hmm_emissions) == 0:
        logger.error("'hmm_emissions' must be a nonempty list.")
        valid = False

    if valid:
        for e in hmm_emissions:
            if e in validators:
                if not validators[e](algo_params):
                    valid = False
    return valid


# Back-compat alias
_hmm_input_validator = hmm_input_validator
