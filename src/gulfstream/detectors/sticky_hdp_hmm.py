"""Truncated sticky HDP-HMM regime detection.

Approximates Fox et al. sticky HDP-HMM with a finite truncation:
stick-breaking weights for the initial distribution, sticky self-transition
bias κ, Gaussian emissions via ``hmmlearn``, then prune nearly empty states.
"""
from __future__ import annotations

import logging

import hmmlearn.hmm as hmm
import numpy as np
import polars as pl

from gulfstream.common import frames, utils
from gulfstream.common.results import AlgoResults
from gulfstream.detectors import common_validation as common

logger = logging.getLogger(__name__)


def _stick_breaking(alpha: float, k: int, rng: np.random.Generator) -> np.ndarray:
    """Finite stick-breaking weights of length ``k`` (renormalized)."""
    betas = rng.beta(1.0, alpha, size=k)
    remaining = 1.0
    weights = np.zeros(k, dtype=float)
    for i in range(k - 1):
        weights[i] = betas[i] * remaining
        remaining *= 1.0 - betas[i]
    weights[-1] = remaining
    weights = np.clip(weights, 1e-8, None)
    return weights / weights.sum()


def _sticky_transition(
    weights: np.ndarray,
    kappa: float,
    alpha: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Row-stochastic sticky transition matrix (truncated HDP style)."""
    k = len(weights)
    # Shared DP base measure β, sticky κ on diagonal.
    trans = np.zeros((k, k), dtype=float)
    for i in range(k):
        # Dirichlet(α β + κ e_i) approximated by normalized (α β + κ e_i)
        row = alpha * weights.copy()
        row[i] += kappa
        row = np.clip(row, 1e-8, None)
        # light Dirichlet noise for diversity
        row = rng.dirichlet(row * 10.0 + 1e-3)
        trans[i] = row
    return trans


def sticky_hdp_hmm_predict_regimes(
    df: pl.DataFrame | np.ndarray,
    regimes: int | None = None,
    hdp_max_states: int = 8,
    sticky_kappa: float = 10.0,
    hdp_alpha: float = 1.0,
    hdp_gamma: float = 1.0,
    hmm_n_iter: int = 100,
    min_state_frac: float = 0.02,
    random_state: int | None = 42,
    **kwargs,
) -> AlgoResults:
    """Fit a truncated sticky HDP-HMM and return ordered regime labels.

    Parameters
    ----------
    regimes :
        Unused for the HDP truncation (kept for YAML grid compatibility).
        Effective number of states is discovered up to ``hdp_max_states``.
    hdp_max_states :
        Truncation level K.
    sticky_kappa :
        Self-transition stickiness (Fox κ).
    hdp_alpha / hdp_gamma :
        Transition / top-level DP concentration approximations.
    min_state_frac :
        Drop states whose occupancy is below this fraction after fitting.
    """
    X = frames.to_numpy(df) if isinstance(df, pl.DataFrame) else np.asarray(df, dtype=float)
    n = X.shape[0]
    k = int(hdp_max_states)
    if k < 2:
        raise ValueError("hdp_max_states must be >= 2")
    rng = np.random.default_rng(random_state)

    beta = _stick_breaking(float(hdp_gamma), k, rng)
    startprob = _stick_breaking(float(hdp_alpha), k, rng)
    transmat = _sticky_transition(beta, float(sticky_kappa), float(hdp_alpha), rng)

    model = hmm.GaussianHMM(
        n_components=k,
        n_iter=int(hmm_n_iter),
        covariance_type="full",
        min_covar=1e-2,
        init_params="mc",  # means + covars random; we supply sticky start/trans
        params="stmc",
        random_state=random_state,
    )
    model.startprob_ = startprob
    model.transmat_ = transmat
    model.fit(X)
    raw = model.predict(X)

    # Prune rare states
    counts = np.bincount(raw, minlength=k).astype(float)
    keep = np.where(counts / max(n, 1) >= float(min_state_frac))[0]
    if len(keep) == 0:
        keep = np.array([int(np.argmax(counts))])
    remap = {old: new for new, old in enumerate(keep.tolist())}
    # Map dropped states to nearest kept mean
    kept_means = model.means_[keep]
    pruned = np.empty(n, dtype=int)
    for t, s in enumerate(raw):
        if s in remap:
            pruned[t] = remap[s]
        else:
            dist = np.sum((kept_means - X[t]) ** 2, axis=1)
            pruned[t] = int(np.argmin(dist))

    labels = utils._map_labels_to_ordered_integers(pruned)
    bkpts = utils._convert_labels_to_bkpts(labels)
    return AlgoResults(bkpts=bkpts, labels=labels)


def sticky_hdp_hmm_param_generator(params: dict):
    if "sticky_hdp_hmm" not in params["algo"].get("regime_detection_algorithm", []):
        return
    # regimes list may be present for other algos; HDP uses truncation level
    max_states = params["algo"].get("hdp_max_states", params["algo"].get("regimes", [8]))
    for k in max_states:
        for kappa in params["algo"].get("sticky_kappa", [10.0]):
            for alpha in params["algo"].get("hdp_alpha", [1.0]):
                for gamma in params["algo"].get("hdp_gamma", [1.0]):
                    for n_iter in params["algo"].get("hmm_n_iter", [100]):
                        for rs in common.algo_grid(params, "random_state", [42]):
                            yield {
                                "regime_detection_algorithm": "sticky_hdp_hmm",
                                "hdp_max_states": k,
                                "sticky_kappa": kappa,
                                "hdp_alpha": alpha,
                                "hdp_gamma": gamma,
                                "hmm_n_iter": n_iter,
                                "random_state": rs,
                                # placate validators that expect regimes
                                "regimes": k,
                            }


def sticky_hdp_hmm_params_printout() -> dict:
    return {
        "sticky_hdp_hmm_hdp_max_states": ["truncation level"],
        "sticky_hdp_hmm_sticky_kappa": ["sticky kappa"],
        "sticky_hdp_hmm_hmm_n_iter": ["EM iterations"],
    }


def sticky_hdp_hmm_input_validator(algo_params: dict) -> bool:
    # Accept either regimes or hdp_max_states
    if algo_params.get("hdp_max_states") is not None:
        vals = algo_params["hdp_max_states"]
        if isinstance(vals, list) and all(isinstance(x, int) and x >= 2 for x in vals):
            return True
    return common._valid_regimes(algo_params)
