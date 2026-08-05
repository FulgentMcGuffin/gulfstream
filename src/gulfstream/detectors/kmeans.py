from __future__ import annotations
"""
Regime detection using k-means.
"""
import numpy as np
import polars as pl
import logging
from sklearn.cluster import KMeans

from gulfstream.common.results import AlgoResults
from gulfstream.detectors import common_validation as common
from gulfstream.common import frames
from gulfstream.common import utils

logger = logging.getLogger(__name__)


def kmeans_predict_regimes(
    df: pl.DataFrame | np.ndarray,
    regimes: int,
    random_state: int = None,
    **kwargs
) -> AlgoResults:
    """
    Predict regimes by k-means clustering on df.
    """
    X = frames.to_numpy(df) if isinstance(df, pl.DataFrame) else np.asarray(df)
    kmeans = KMeans(n_clusters=regimes, init='k-means++', random_state=random_state)
    labels = utils._map_labels_to_ordered_integers(kmeans.fit_predict(X))
    bkpts = utils._convert_labels_to_bkpts(labels)
    return AlgoResults(bkpts=bkpts, labels=labels)


def kmeans_param_generator(params: dict):
    """
    Yield all valid combinations of parameters for k-means clustering regime detection.
    """
    if 'kmeans' in params['algo']['regime_detection_algorithm']:
        for regimes in params['algo']['regimes']:
            for random_state in common.algo_grid(params, "random_state", [None]):
                yield {
                    'regime_detection_algorithm': 'kmeans',
                    'regimes': regimes,
                    'random_state': random_state
                }


def kmeans_params_printout() -> dict:
    """
    Dict for writing heading of this algorithm. Each entry will be a
    column heading in an Excel sheet. Values are singleton lists with
    the heading to be used for the associated column. Keys should use
    the format "algoname_paramname".
    """
    return {
        'kmeans_regimes': ['number of regimes'],
        'kmeans_random_state': ['random_state']
    }


def _valid_random_state(algo_params: dict) -> bool:
    random_state = algo_params.get('random_state')
    if random_state is None:
        return True  # Optional parameter.
    elif not isinstance(random_state, list):
        logger.error("'random_state' in 'algo' must be type list[int]. Got type %s.", type(random_state))
        return False
    elif not all(isinstance(x, int) and x > 0 for x in random_state):
        logger.error("All entries of 'random_state' must be positive ints.")
        return False
    return True


def kmeans_input_validator(algo_params: dict) -> bool:
    """
    Returns True if algo_params contains a valid set of parameters
    for k-means regime detection. May contain additional parameters as well.
    """
    valid = True
    if not common._valid_regimes(algo_params):
        valid = False
    if not _valid_random_state(algo_params):
        valid = False
    return valid
