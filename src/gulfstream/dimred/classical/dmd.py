from __future__ import annotations
"""
Dynamic mode decomposition.
"""

import numpy as np
import polars as pl
from typing import Literal, Tuple
import logging
from gulfstream.common import frames
from gulfstream.common.results import DimredResults
from gulfstream.dimred.classical import _common as common

logger = logging.getLogger(__name__)


def _dmd_dimred(
    df: pl.DataFrame,
    *,
    dmd_stride: int,
    dmd_rolling_window: int,
    rank_selection_method: Literal['entropy', 'explained_variance', 'svht', 'user_specified'],
    rank: int = None,
    threshold: float = None,
    **kwargs
) -> DimredResults:
    """
    Computes DMD features for df, as well as the rank if rank is not provided.

    Parameters
    ----------
    df : pl.DataFrame
        Time series to calculate DMD features for.
    dmd_stride : int
        Step size for rolling window calculations.
    dmd_rolling_window : int
        Size of rolling window.
    rank_selection_method : str
        Method for selecting the rank. Valid choices are 'entropy', 'svht' (singular
        value hard-thresholding), 'explained_variance', and 'user_specified'.
    rank : int, optional
        DMD rank. Resulting dataframe will have 4*rank columns. Only
        required in rank_selection_method = 'user_specified'.
    threshold : float, optional
        Threshold for selecting the rank using entropy or explained variance.
        Should be between 0 and 1. Must be specified if rank is None and
        rank_selection_method is 'entropy' or 'explained_variance'.
    kwargs : dict, optional
        Only the arguments listed above are used. kwargs is
        included for convenience.

    Returns
    -------
    DimredResults
        Container containing the results of dimension reduction.
    """
    n_feat = frames.n_features(df)
    X = frames.to_numpy(df)
    if dmd_rolling_window > df.height or dmd_rolling_window < 1:
        raise ValueError("'dmd_rolling_window' must be positive and less than length of df.")
    if dmd_stride > df.height or dmd_stride < 1:
        raise ValueError("'dmd_stride' must be positive and less than length of df.")
    if rank_selection_method != 'user_specified':
        logger.info("Calculating rank for DMD.")
        ranks = _dmd_rank_over_time(
            X,
            dmd_rolling_window=dmd_rolling_window,
            dmd_stride=dmd_stride,
            threshold=threshold,
            method=rank_selection_method
        )
        # The algorithm is pretty quick with DMD, so being conservative here
        # and taking max rank instead of an average doesn't hurt performance much.
        my_rank = max(ranks)
        logger.info("Found rank %s for DMD.", my_rank)
    else:
        if rank is None:
            raise ValueError("Must specify 'rank' if using 'user_specified' rank selection.")
        elif rank > min(df.height, n_feat) or rank < 1:
            raise ValueError("'rank' must be positive and not exceed number of rows and number of columns of raw data.")
        my_rank = rank

    # Sliding-window DMD yields fewer rows than the input; synthesize dates.
    df_dimred = frames.from_numpy(
        _sliding_window_features(
            X,
            dmd_rolling_window=dmd_rolling_window,
            dmd_stride=dmd_stride,
            rank=my_rank
        )
    )
    res = DimredResults(df=df_dimred,
                        dimred='dmd',
                        rank_selection_method=rank_selection_method,
                        rank=my_rank, dmd_stride=dmd_stride, dmd_rolling_window=dmd_rolling_window)
    if rank_selection_method in ['explained_variance', 'entropy']:
        res.threshold = threshold
    return res


def _dmd_generator(df: pl.DataFrame, params: dict):
    """
    Yield an iterator for all requested dimension reductions of df
    using DMD.

    Parameters
    ----------
    df : pl.DataFrame
        Data matrix to perform dimension reduction on.
    params : dict
        Dictionary of parameters with formatting as in the README.

    Yields
    ------
    DimredResults
        Results of dimension reduction with each valid combination of parameters.
    """
    rank_generators = [
        common._user_specified_rank_generator,
        common._entropy_param_generator,
        common._explained_variance_param_generator,
        common._svht_param_generator
    ]
    if 'dmd' in params['algo']['dimred']:
        for dmd_rolling_window in params['algo']['dmd_rolling_window']:
            for dmd_stride in params['algo']['dmd_stride']:
                for handler in rank_generators:
                    for d in handler(params):
                        d['dimred'] = 'dmd'
                        d['dmd_rolling_window'] = dmd_rolling_window
                        d['dmd_stride'] = dmd_stride
                        yield _dmd_dimred(df, **d)


#####################
# DMD IMPLEMENTATION
#####################

def _dmd(
    X1: np.ndarray,
    X2: np.ndarray,
    rank: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform Dynamic Mode Decomposition (DMD) on the input matrices X1 and X2.

    Parameters
    ----------
    X1, X2 : np.ndarray
        Shape (T, d) each, typically with T >= d.
    rank : int, optional
        Number of top singular values to truncate to.
        Default is None (use all singular values).

    Returns
    -------
    np.ndarray
        Shape (rank,) or (min(T, d),) if rank not specified.
        Eigenvalues of the transition matrix driving the
        underlying dynamical system.
    np.ndarray
        Shape (T, rank) or (T, d) if rank not specified. DMD modes.
    """
    U, Sigma, VT = np.linalg.svd(X1, full_matrices=False)
    if rank is not None:
        U = U[:, :rank]
        Sigma = np.diag(Sigma[:rank])
        VT = VT[:rank, :]
    else:
        Sigma = np.diag(Sigma)
    A_tilde = U.T @ X2 @ VT.T @ np.linalg.inv(Sigma)
    eigenvalues, eigenvectors = np.linalg.eig(A_tilde)
    modes = X2 @ VT.T @ np.linalg.inv(Sigma) @ eigenvectors
    return eigenvalues, modes


def _extract_enriched_dmd_features(
    X1: np.ndarray,
    X2: np.ndarray,
    rank: int = 3
) -> np.ndarray:
    """
    This is a heuristic of Babis' with good intuition
    that works well. Extract enriched features from DMD, including:
    - real(eigenvalues), imag(eigenvalues)
    - log-transformed mode amplitudes
    - leading singular values (from X1)
    Returns a 1D array (feature vector) for a single window.

    Parameters
    ----------
    X1, X2 : np.ndarray
        Shape (T, d) each, typically with T >= d.
    rank : int, optional
        Uses top rank singular values to calculate DMD
        features. Default is 3.

    Returns
    -------
    np.ndarray
        features. Shape (4*rank,). Stacked left to right, consists of
        - real parts of eigenvalues
        - imaginary parts of eigenvalues
        - log(1 + mode amplitudes)
        - top rank singular values of X1.
    """
    eigenvalues, modes = _dmd(X1, X2, rank=rank)
    mode_amplitudes = np.abs(modes).mean(axis=0)
    log_amplitudes = np.log1p(mode_amplitudes)  # log(1 + amplitude)
    sv = np.linalg.svd(X1, compute_uv=False)
    sv_trunc = sv[:rank] if len(sv) >= rank else sv  # handle short array
    features = np.hstack([
        np.real(eigenvalues),
        np.imag(eigenvalues),
        log_amplitudes,
        sv_trunc
    ])
    return features


def _sliding_window_features(
    data: np.ndarray,
    dmd_rolling_window: int = 30,
    dmd_stride: int = 3,
    rank: int = 3
) -> np.ndarray:
    """
    Slide a window over 'data', collecting one DMD feature vector per window.

    Parameters
    ----------
    data : np.ndarray
        Data matrix to calculate DMD features for.
    dmd_rolling_window : int, optional
        Size of rolling window. Default is 30.
    dmd_stride : int, optional
        Rolling window moves by this many steps. Default is 3.

    Returns
    -------
    np.ndarray
        DMD features. Shape is (num_windows, 4*rank).
    """
    T = data.shape[0]
    features_list = []
    # We'll go up to T - dmd_rolling_window (inclusive)
    for start in range(0, T - dmd_rolling_window + 1, dmd_stride):
        X1 = data[start: start + dmd_rolling_window - 1, :]
        X2 = data[start + 1: start + dmd_rolling_window, :]
        feat = _extract_enriched_dmd_features(X1, X2, rank=rank)
        features_list.append(feat)
    return np.array(features_list)


#################
# DMD RANK SELECTOR
#################

def _dmd_rank_over_time(
    data: np.ndarray,
    dmd_rolling_window: int = 30,
    dmd_stride: int = 3,
    threshold: float = 0.95,
    method: Literal['entropy', 'explained_variance'] = 'entropy'
) -> list:
    """
    Computes the optimal ranks of rolling windows of data with the specified
    dmd_rolling_window and steps of size dmd_stride. Returns a list of ranks.

    Parameters
    ----------
    data : np.ndarray
        Data matrix.
    dmd_rolling_window : int, optional
        Size of rolling window. Default is 30.
    dmd_stride : int, optional
        Rolling window moves by this many days for each step.
        Default is 3.
    threshold : float, optional
        Explained variance threshold or explained entropy threshold
        for selecting rank. Default is 0.95.
    method : str, optional
        Valid choices are 'entropy', 'explained_variance', or 'svht'.
        Method for selecting rank. Default is 'entropy'.

    Returns
    -------
    list[int]
        Optimal ranks using requested method for each rolling window.
    """
    handlers = {
        'entropy': common._rank_by_entropy,
        'explained_variance': common._rank_by_explained_variance,
        'svht': common._rank_by_svht
    }
    handler = handlers.get(method, None)
    if handler is None:
        raise ValueError(f"Unknown method {method}.")

    T = data.shape[0]
    ranks = []
    for start in range(0, T - dmd_rolling_window + 1, dmd_stride):
        X1 = data[start:start + dmd_rolling_window - 1, :]
        _, Sigma, _ = np.linalg.svd(X1, full_matrices=False)
        ranks.append(
            handler(
                eigvals=Sigma ** 2,
                threshold=threshold,
                row=X1.shape[0],
                column=X1.shape[1],
            )
        )
    return ranks
