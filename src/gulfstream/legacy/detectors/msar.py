from __future__ import annotations
"""
Regime detection with Markov switching autoregressive model.
"""
import numpy as np
import polars as pl
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from sklearn.decomposition import PCA

from gulfstream.common import frames
from gulfstream.common import utils
from gulfstream.common.results import AlgoResults
from gulfstream.legacy.detectors import common_validation as common


def msar_predict_regimes(df: pl.DataFrame, regimes: int, **kwargs) -> AlgoResults:
    """
    Fit a Markov switching AR model to predict regimes.

    Notes
    -----
    The model only supports univariate time series, so we fit PCA prior to
    fitting the model. statsmodels accepts a 1-d numpy endog array.
    """
    X = frames.to_numpy(df)
    pca = PCA(n_components=1)
    y = np.asarray(pca.fit_transform(X).ravel(), dtype=float)

    model = MarkovRegression(y, k_regimes=regimes, trend="c", switching_variance=True)
    result = model.fit()

    regime_probabilities = result.smoothed_marginal_probabilities
    raw_labels = np.argmax(np.asarray(regime_probabilities), axis=1)

    labels = utils._map_labels_to_ordered_integers(raw_labels)
    bkpts = utils._convert_labels_to_bkpts(labels)
    return AlgoResults(bkpts=bkpts, labels=labels)


def msar_params_generator(params: dict):
    """
    Yield all valid combinations of parameters for MSAR regime detection.
    """
    if "msar" in params["algo"]["regime_detection_algorithm"]:
        for regimes in params["algo"]["regimes"]:
            yield {
                "regime_detection_algorithm": "msar",
                "regimes": regimes,
            }


def msar_params_printout() -> dict:
    return {"msar_regimes": ["number of regimes"]}


def msar_input_validator(algo_params: dict) -> bool:
    valid = True
    if not common._valid_regimes(algo_params):
        valid = False
    return valid
