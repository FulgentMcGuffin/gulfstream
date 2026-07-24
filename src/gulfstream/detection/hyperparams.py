"""Hyperparameter selection for the gulfstream regime detection algorithm."""
from collections import namedtuple
from typing import Tuple
import logging

import numpy as np
import polars as pl
from sklearn.decomposition import PCA
from statsmodels.tsa.stattools import acf

from gulfstream.common import frames
from gulfstream.common.results import HyperparameterResults
from gulfstream.detection.time_index import (
    convert_original_index_to_dimred_index,
    convert_dimred_index_to_original_index,
)

logger = logging.getLogger(__name__)

LagConfig = namedtuple("LagConfig", ["lag_window", "max_lag", "pca_explained_variance"])
DEFAULT_LAG_CONFIG = LagConfig(lag_window=100, max_lag=40, pca_explained_variance=0.95)

SampleSizeConfig = namedtuple("SampleSizeConfig", ["min_sample_size"])
DEFAULT_SAMPLESIZE_CONFIG = SampleSizeConfig(min_sample_size=50)


def _select_hyperparameters(
    params: dict,
    date: str,
    df: pl.DataFrame = None,
    bkpt: int = None,
    **kwargs
) -> HyperparameterResults:
    """Select all necessary hyperparameters for the in-house regime
    detection algorithm specified in params.

    Parameters
    ----------
    params : dict
        'test' section of the dictionary of parameters for one run of the breakpoint algorithm.
        Format as in the README, but each entry is a single value instead of a list.
    date : str
        String to print for the breakpoint hyperparameters are being selected for. Intended to
        be the date of the breakpoint in the original raw time series. Used for logging purposes.
    df : pl.DataFrame, optional
        PCA of time series to use for lag selection with ACF decay. Default is None.
    bkpt : int, optional
        Breakpoint to calculate lag for. Should pass bkpt in units of the time series the
        statistical tests are being applied to (any necessary unit conversions will be carried
        out by this function). bkpt is only required if using ACF decay for lag selection.
        Default is None.

    Returns
    -------
    HyperparameterResults
        Container containing the requested hyperparameters.
    """
    res = HyperparameterResults()
    test_params = params.get('test')
    if not test_params:
        raise KeyError("'test' must be specified.")

    handlers = {
        'lag': select_lag,
        'sample_size': select_sample_size,
        'window': select_window
    }

    lag_params = test_params.get('lag')
    if lag_params:
        handler = handlers['lag']
        lag_args = {
            'df': df,
            'bkpt': bkpt,
            'params': params,
            'date': date
        }
        try:
            res.lag = handler(**lag_args)
        except:
            raise

    window_params = test_params.get('window')
    if window_params:
        handler = handlers['window']
        window_args = {'params': params}
        try:
            res.window = handler(**window_args)
        except:
            raise

    sample_size_params = test_params.get('sample_size')
    if sample_size_params:
        handler = handlers['sample_size']
        window = res.window.get('window')
        if not window:
            raise ValueError("res.window.window is None. Must be int.")
        sample_args = {
            'params': params,
            'window': window
        }
        try:
            res.sample_size = handler(**sample_args)
        except:
            raise

    return res


def select_hyperparameters(
    params: dict,
    date: str,
    df: pl.DataFrame = None,
    bkpt: int = None,
    **kwargs
) -> HyperparameterResults:
    """Select all necessary hyperparameters for the in-house regime
    detection algorithm specified in params.

    Parameters
    ----------
    params : dict
        'test' section of the dictionary of parameters for one run of the breakpoint algorithm.
        Format as in the README, but each entry is a single value instead of a list.
    date : str
        String to print for the breakpoint hyperparameters are being selected for.
    df : pl.DataFrame, optional
        PCA of time series to use for lag selection with ACF decay. Default is None.
    bkpt : int, optional
        Breakpoint to calculate lag for. Default is None.

    Returns
    -------
    HyperparameterResults
        Container containing the requested hyperparameters.
    """
    return _select_hyperparameters(params, date, df=df, bkpt=bkpt, **kwargs)


def user_specified_lag(params: dict, **kwargs) -> dict:
    """TODO: should add some input validation to this.

    Notes
    -----
    Not doing unit conversions for the user here since it would be confusing
    (seeing that their user-specified lag isn't being used when requested).
    """
    return {
        'method': 'user_specified',
        'lag': params['test']['lag']['lag'],
        'lag_days': params['test']['lag']['lag']
    }


def select_max_acf_drop_lag(
    df: pl.DataFrame,
    bkpt: int,
    params: dict,
    date: str,
    **kwargs
) -> dict:
    """Convert bkpt into units of the raw time series.
    Then choose the best lag using the following procedure.
    Take the first rank PCs. For each, use statsmodels acf
    function to compute the autocorrelation function.
    Choose first lag such that all autocorrelations are below threshold.
    Convert this lag back to units of the time series statistical tests
    are being applied to.

    Parameters
    ----------
    df : pl.DataFrame
        PCA of time series.
    bkpt : int
        Point to calculate lag around. Should be in units of the time series
        statistical tests are being applied to.
    params : dict
        Dict of parameters for one instance of the breakpoint algorithm.
        See README for formatting.
    date : str
        Date string for logging.
    kwargs : dict, optional
        Only the listed parameters are used.

    Returns
    -------
    dict
        Contains the following entries:
        'method' (str): Always equals 'acf_decay'.
        'acf_decay' (float): Value of ACF at the resulting lag.
        'lag' (int): Number of transition days. In units of the time series
            statistical tests are being applied to.
        'window' (int): Window size used for ACF calculation.
        'max_lag' (int): Maximum allowed lag.
    """
    # Since this method uses PCA of raw time series, need to
    # convert units in case of e.g. DMD.
    try:
        conv_bkpt = convert_dimred_index_to_original_index(bkpt, params)
    except:
        raise

    lag_params = params['test']['lag']
    window = lag_params.get('window', DEFAULT_LAG_CONFIG.lag_window)
    max_lag = lag_params.get('max_lag', DEFAULT_LAG_CONFIG.max_lag)
    threshold = lag_params.get('threshold')
    if not threshold:
        raise KeyError("'threshold' must be provided.")
    if conv_bkpt - window < 0 or conv_bkpt + window > df.height:
        raise ValueError(f"Breakpoint {date} is too close to the boundary.")
    if 2 * window < max_lag + 1:
        raise ValueError(f"window {window} too small for max_lag {max_lag}.")

    # Get windowed region.
    X = frames.slice_rows(df, conv_bkpt - window, conv_bkpt + window)
    acf_fallpoints = []
    for col in frames.feature_columns(X):
        res = acf(X[col].to_numpy(), nlags=max_lag, fft=True)
        # Find first time ACF falls below threshold.
        i = 0
        while i < len(res):
            if abs(res[i]) < threshold:
                acf_fallpoints.append((i, res[i]))
                break
            i += 1
        if i == len(res):
            # Failed. Take largest lag.
            acf_fallpoints.append((len(res) - 1, res[-1]))

    # Take the largest number of days it took to fall below threshold.
    lag_tuple = max(acf_fallpoints, key=lambda x: x[0])
    # Convert 'lag' back to units of time series we are applying stat tests to.
    return {
        'method': 'acf_decay',
        'acf_decay': lag_tuple[1],
        'lag': convert_original_index_to_dimred_index(lag_tuple[0], params),
        'lag_days': lag_tuple[0],  # Lag in raw time series units.
        'window': window,
        'max_lag': max_lag
    }


def select_lag(params: dict, date: str, df: pl.DataFrame = None,
                          bkpt: int = None, **kwargs) -> dict:
    """Dispatcher for lag selection."""
    handlers = {
        'user_specified': user_specified_lag,
        'acf_decay': select_max_acf_drop_lag
    }
    test_params = params.get('test')
    if not test_params:
        raise KeyError("'test' must be specified.")
    lag_params = test_params.get('lag')
    if not lag_params:
        raise KeyError("'lag' must be specified.")
    method = lag_params.get('method')
    if not method:
        raise KeyError("'method' must be specified.")
    handler = handlers.get(method)
    if not handler:
        raise ValueError(f"Unknown method {method}.")
    args = {
        'params': params,
        'df': df,
        'bkpt': bkpt,
        'date': date
    }
    try:
        res = handler(**args)
    except:
        raise
    return res


def user_specified_sample_size(
    params: dict,
    window: int,
    **kwargs
) -> dict:
    """Notes
    -----
    window should be in units of the time series stat tests are being applied to.
    """
    sample_params = params['test']['sample_size']
    num_samples = sample_params.get('num_samples')
    if not num_samples:
        raise KeyError("'num_samples' must be specified.")
    if num_samples > 2 * window:
        logger.warning(
            "'num_samples' %s is larger than windowed region %s, "
            "using sample size %s.", num_samples, 2 * window, 2 * window
        )
        num_samples = 2 * window
    return {
        'method': 'user_specified',
        'num_samples': num_samples
    }


def select_sample_size_by_proportion(
    params: dict,
    window: int,
    **kwargs
) -> dict:
    """Returns the proportion of the window specified in
    params['test']['sample_size']['sample_proportion'].

    Parameters
    ----------
    params : dict
        Dict of parameters for one instance of the in-house regime detection
        algorithm. See README for formatting.
    window : int
        Length of segments on each side of a bkpt for testing.
        Should be in units of the time series stat tests are being applied to.

    Returns
    -------
    dict
        Contains: 'method', 'sample_proportion', 'num_samples'.
    """
    sample_params = params['test']['sample_size']
    min_sample_size = sample_params.get(
        'min_sample_size', DEFAULT_SAMPLESIZE_CONFIG.min_sample_size
    )
    proportion = sample_params.get('sample_proportion')
    if not proportion:
        raise KeyError("'sample_proportion' must be specified.")
    if proportion <= 0 or proportion > 1:
        raise ValueError("'sample_proportion' must be between 0 and 1.")
    window_size = 2 * window
    sample_size = round(proportion * window_size)
    if sample_size < min_sample_size:
        logger.warning(
            "Requested proportion %s of windowed region %s is %s, "
            "which is smaller than min_sample_size %s.",
            proportion, window_size, sample_size, min_sample_size
        )
        if min_sample_size > window_size:
            logger.warning(
                "min_sample_size %s is larger than window %s. "
                "Using full window %s.", min_sample_size, window_size, window_size
            )
            return {'method': 'proportional', 'sample_proportion': 1.0, 'num_samples': window_size}
        return {
            'method': 'proportional',
            'sample_proportion': min_sample_size / window_size,
            'num_samples': min_sample_size
        }
    return {'method': 'proportional', 'sample_proportion': proportion, 'num_samples': sample_size}


def select_sample_size(params: dict, **kwargs) -> dict:
    handlers = {
        'user_specified': user_specified_sample_size,
        'proportional': select_sample_size_by_proportion
    }
    sample_params = params['test']['sample_size']
    method = sample_params.get('method')
    if not method:
        raise KeyError("'method' must be specified.")
    handler = handlers.get(method)
    if not handler:
        raise ValueError(f"Unknown method {method}.")
    return handler(params, **kwargs)


def user_specified_window(params: dict, **kwargs) -> dict:
    """Return the user-specified window size (no unit conversion)."""
    return {
        "method": "user_specified",
        "window": params["test"]["window"]["window"],
        "window_days": params["test"]["window"]["window"],
    }


def ess_window(params: dict, *, df: pl.DataFrame | None = None, **kwargs) -> dict:
    """Effective-sample-size heuristic for the MMD window.

    Uses a simple AR(1)-style ESS estimate on the first feature (or PCA score
    when ``df`` is provided): ``ess ≈ n * (1-ρ)/(1+ρ)``, then takes a fraction
    of ESS as the half-window on each side of a candidate breakpoint.
    """
    window_cfg = params["test"]["window"]
    frac = float(window_cfg.get("ess_fraction", 0.25))
    min_window = int(window_cfg.get("min_window", 20))
    max_window = int(window_cfg.get("max_window", 120))
    if df is None or df.height < 10:
        w = max(min_window, min(max_window, 40))
        return {"method": "ess", "window": w, "window_days": w, "ess": None}

    x = frames.to_numpy(frames.select_features(df, frames.feature_columns(df)[:1])).ravel()
    x = x - np.nanmean(x)
    if len(x) < 3 or np.nanstd(x) == 0:
        rho = 0.0
    else:
        rho = float(np.corrcoef(x[:-1], x[1:])[0, 1])
        if not np.isfinite(rho):
            rho = 0.0
        rho = float(np.clip(rho, -0.99, 0.99))
    ess = len(x) * (1.0 - rho) / (1.0 + rho)
    w = int(round(frac * ess / 2.0))
    w = max(min_window, min(max_window, w))
    return {"method": "ess", "window": w, "window_days": w, "ess": float(ess), "rho": rho}


def select_window(params: dict, **kwargs) -> dict:
    """Dispatcher for window size selection."""
    handlers = {
        "user_specified": user_specified_window,
        "ess": ess_window,
    }
    method = params["test"]["window"].get("method")
    if not method:
        raise KeyError("'method' must be specified.")
    handler = handlers.get(method)
    if not handler:
        raise ValueError(f"Unknown window method {method}.")
    return handler(params, **kwargs)


def asked_for_acf_lag_selection(params: dict) -> bool:
    if not params.get('test'):
        raise KeyError("'test' must be specified.")
    lag = params['test'].get('lag')
    if not lag:
        return False
    for d in lag:
        method = d.get('method')
        if method == 'acf_decay':
            return True
    return False


def calculate_pca_for_lag_selection(df: pl.DataFrame) -> pl.DataFrame:
    """Calculates PCA for use in lag hyperparameter selection with ACF decay.

    Parameters
    ----------
    df : pl.DataFrame
        Raw time series.

    Returns
    -------
    pl.DataFrame
        PCA of df with rank selected to explain
        DEFAULT_LAG_CONFIG.pca_explained_variance of the variance of df.
    """
    logger.info(
        'Calculating PCA of time series for lag selection based on '
        'autocorrelation function decay.'
    )
    p = PCA(
        n_components=DEFAULT_LAG_CONFIG.pca_explained_variance,
        svd_solver='auto'
    )
    df_pca = frames.with_same_dates(p.fit_transform(frames.to_numpy(df)), df)
    logger.info('Number of PCs for lag selection: %s.', frames.n_features(df_pca))
    return df_pca
