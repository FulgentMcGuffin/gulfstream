"""Model-based dimred backends: Bayesian GMM, HMM, k-means, HDBSCAN, OPTICS,
MSAR, Wasserstein, ruptures, and TFT.

These reuse the legacy detector models but emit continuous embeddings
(soft responsibilities / distances / attention) so Graph 1 / Graph 2 can
treat them like PCA/kPCA/raw/DMD/t-SNE/UMAP.
"""
from __future__ import annotations

import logging
import math

import numpy as np
import polars as pl
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import BayesianGaussianMixture

from gulfstream.dimred import density as dens
from gulfstream.common import frames
from gulfstream.common.results import DimredResults
from gulfstream.legacy.detectors import ruptures_methods as ruptures_mod
from gulfstream.legacy.detectors import tft as tft_mod
from gulfstream.legacy.detectors import wasserstein as wass_mod

logger = logging.getLogger(__name__)

try:
    import hmmlearn.hmm as hmm
except ImportError:  # pragma: no cover
    hmm = None


def _require_regimes(regimes: int | None) -> int:
    if regimes is None or not isinstance(regimes, int) or regimes < 1:
        raise ValueError("'regimes' must be a positive int for model-based dimred.")
    return regimes


def _as_frame(x: np.ndarray, template: pl.DataFrame, prefix: str) -> pl.DataFrame:
    cols = [f"{prefix}{i}" for i in range(x.shape[1])]
    return frames.with_same_dates(x, template, columns=cols)


# ---------------------------------------------------------------------------
# Bayesian GMM
# ---------------------------------------------------------------------------

def _bayesian_gmm_dimred(
    df: pl.DataFrame,
    regimes: int,
    reg_covar: float = 1e-5,
    random_state: int | None = 42,
    **kwargs,
) -> DimredResults:
    """Soft mixture responsibilities as embedding columns."""
    n = _require_regimes(regimes)
    X = frames.to_numpy(df)
    model = BayesianGaussianMixture(
        n_components=n,
        covariance_type="full",
        random_state=random_state,
        reg_covar=reg_covar,
    )
    model.fit(X)
    probs = model.predict_proba(X)
    return DimredResults(
        df=_as_frame(probs, df, "bgmm_"),
        dimred="bayesian_gmm",
        rank=probs.shape[1],
        rank_selection_method="regimes",
        model=model,
    )


def _bayesian_gmm_generator(df: pl.DataFrame, params: dict):
    if "bayesian_gmm" not in params.get("algo", {}).get("dimred", []):
        return
    for regimes in params["algo"]["regimes"]:
        for reg_covar in params["algo"].get("reg_covar", [1e-5]):
            yield _bayesian_gmm_dimred(df, regimes=regimes, reg_covar=reg_covar)


# ---------------------------------------------------------------------------
# HMM
# ---------------------------------------------------------------------------

def _hmm_dimred(
    df: pl.DataFrame,
    regimes: int,
    hmm_emissions: str = "gaussian",
    hmm_n_iter: int = 100,
    **kwargs,
) -> DimredResults:
    """Posterior state probabilities as embedding columns."""
    if hmm is None:
        raise ImportError("hmmlearn is required for HMM dimred.")
    if hmm_emissions != "gaussian":
        raise ValueError(f"Unknown HMM emissions for dimred: {hmm_emissions}")
    n = _require_regimes(regimes)
    X = frames.to_numpy(df)
    model = hmm.GaussianHMM(
        n_components=n,
        n_iter=hmm_n_iter,
        covariance_type="full",
        min_covar=1e-2,
    )
    model.fit(X)
    probs = model.predict_proba(X)
    return DimredResults(
        df=_as_frame(probs, df, "hmm_"),
        dimred="hmm",
        rank=probs.shape[1],
        rank_selection_method="regimes",
        model=model,
    )


def _hmm_generator(df: pl.DataFrame, params: dict):
    if "hmm" not in params.get("algo", {}).get("dimred", []):
        return
    for regimes in params["algo"]["regimes"]:
        for emissions in params["algo"].get("hmm_emissions", ["gaussian"]):
            for n_iter in params["algo"].get("hmm_n_iter", [100]):
                yield _hmm_dimred(
                    df,
                    regimes=regimes,
                    hmm_emissions=emissions,
                    hmm_n_iter=n_iter,
                )


# ---------------------------------------------------------------------------
# k-means
# ---------------------------------------------------------------------------

def _kmeans_dimred(
    df: pl.DataFrame,
    regimes: int,
    random_state: int | None = None,
    **kwargs,
) -> DimredResults:
    """Distances to cluster centers as embedding columns."""
    n = _require_regimes(regimes)
    X = frames.to_numpy(df)
    model = KMeans(n_clusters=n, init="k-means++", random_state=random_state, n_init=10)
    model.fit(X)
    dists = model.transform(X)
    return DimredResults(
        df=_as_frame(dists, df, "kmeans_"),
        dimred="kmeans",
        rank=dists.shape[1],
        rank_selection_method="regimes",
        model=model,
    )


def _kmeans_generator(df: pl.DataFrame, params: dict):
    if "kmeans" not in params.get("algo", {}).get("dimred", []):
        return
    for regimes in params["algo"]["regimes"]:
        for random_state in params["algo"].get("random_state", [None]):
            yield _kmeans_dimred(df, regimes=regimes, random_state=random_state)


# ---------------------------------------------------------------------------
# HDBSCAN
# ---------------------------------------------------------------------------

def _hdbscan_dimred(
    df: pl.DataFrame,
    min_cluster_size: int = 5,
    min_samples: int | None = None,
    metric: str = "euclidean",
    cluster_selection_epsilon: float = 0.0,
    cluster_selection_method: str = "eom",
    allow_single_cluster: bool = False,
    **kwargs,
) -> DimredResults:
    """Distances to HDBSCAN centroids as embedding columns."""
    X = frames.to_numpy(df)
    model, labels = dens.fit_hdbscan(
        X,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_epsilon=cluster_selection_epsilon,
        cluster_selection_method=cluster_selection_method,
        allow_single_cluster=allow_single_cluster,
    )
    dists = dens.density_embedding(X, labels, model=model)
    return DimredResults(
        df=_as_frame(dists, df, "hdbscan_"),
        dimred="hdbscan",
        rank=dists.shape[1],
        rank_selection_method="discovered",
        model=model,
    )


def _hdbscan_generator(df: pl.DataFrame, params: dict):
    if "hdbscan" not in params.get("algo", {}).get("dimred", []):
        return
    for min_cluster_size in params["algo"].get("hdbscan_min_cluster_size", [5]):
        for min_samples in params["algo"].get("hdbscan_min_samples", [None]):
            for metric in params["algo"].get("hdbscan_metric", ["euclidean"]):
                for eps in params["algo"].get("hdbscan_cluster_selection_epsilon", [0.0]):
                    for method in params["algo"].get(
                        "hdbscan_cluster_selection_method", ["eom"]
                    ):
                        for allow in params["algo"].get(
                            "hdbscan_allow_single_cluster", [False]
                        ):
                            yield _hdbscan_dimred(
                                df,
                                min_cluster_size=min_cluster_size,
                                min_samples=min_samples,
                                metric=metric,
                                cluster_selection_epsilon=eps,
                                cluster_selection_method=method,
                                allow_single_cluster=allow,
                            )


# ---------------------------------------------------------------------------
# OPTICS
# ---------------------------------------------------------------------------

def _optics_dimred(
    df: pl.DataFrame,
    min_samples: int = 5,
    max_eps: float = math.inf,
    metric: str = "minkowski",
    p: int = 2,
    cluster_method: str = "xi",
    xi: float = 0.05,
    min_cluster_size: int | None = None,
    eps: float | None = None,
    **kwargs,
) -> DimredResults:
    """Distances to OPTICS cluster centroids as embedding columns."""
    X = frames.to_numpy(df)
    model, labels = dens.fit_optics(
        X,
        min_samples=min_samples,
        max_eps=max_eps,
        metric=metric,
        p=p,
        cluster_method=cluster_method,
        xi=xi,
        min_cluster_size=min_cluster_size,
        eps=eps,
    )
    dists = dens.density_embedding(X, labels, model=model)
    return DimredResults(
        df=_as_frame(dists, df, "optics_"),
        dimred="optics",
        rank=dists.shape[1],
        rank_selection_method="discovered",
        model=model,
    )


def _optics_generator(df: pl.DataFrame, params: dict):
    if "optics" not in params.get("algo", {}).get("dimred", []):
        return
    for min_samples in params["algo"].get("optics_min_samples", [5]):
        for max_eps in params["algo"].get("optics_max_eps", [math.inf]):
            for metric in params["algo"].get("optics_metric", ["minkowski"]):
                for p in params["algo"].get("optics_p", [2]):
                    for cluster_method in params["algo"].get(
                        "optics_cluster_method", ["xi"]
                    ):
                        for xi in params["algo"].get("optics_xi", [0.05]):
                            for min_cluster_size in params["algo"].get(
                                "optics_min_cluster_size", [None]
                            ):
                                for eps in params["algo"].get("optics_eps", [None]):
                                    yield _optics_dimred(
                                        df,
                                        min_samples=min_samples,
                                        max_eps=max_eps,
                                        metric=metric,
                                        p=p,
                                        cluster_method=cluster_method,
                                        xi=xi,
                                        min_cluster_size=min_cluster_size,
                                        eps=eps,
                                    )


# ---------------------------------------------------------------------------
# MSAR
# ---------------------------------------------------------------------------

def _msar_dimred(df: pl.DataFrame, regimes: int, **kwargs) -> DimredResults:
    """Smoothed MSAR regime probabilities (after PCA→1D) as embedding columns."""
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

    n = _require_regimes(regimes)
    X = frames.to_numpy(df)
    pca = PCA(n_components=1)
    y = np.asarray(pca.fit_transform(X).ravel(), dtype=float)
    model = MarkovRegression(y, k_regimes=n, trend="c", switching_variance=True)
    result = model.fit()
    probs = np.asarray(result.smoothed_marginal_probabilities, dtype=float)
    if probs.ndim == 1:
        probs = probs.reshape(-1, 1)
    return DimredResults(
        df=_as_frame(probs, df, "msar_"),
        dimred="msar",
        rank=probs.shape[1],
        rank_selection_method="regimes",
        model=result,
    )


def _msar_generator(df: pl.DataFrame, params: dict):
    if "msar" not in params.get("algo", {}).get("dimred", []):
        return
    for regimes in params["algo"]["regimes"]:
        yield _msar_dimred(df, regimes=regimes)


# ---------------------------------------------------------------------------
# Wasserstein
# ---------------------------------------------------------------------------

def _expand_window_features(
    df: pl.DataFrame,
    window_feats: np.ndarray,
    window: int,
    stride: int,
) -> np.ndarray:
    """Map per-window feature rows onto the original series length."""
    data_len = df.height
    n_feat = window_feats.shape[1]
    out = np.full((data_len, n_feat), np.nan, dtype=float)
    m_windows = window_feats.shape[0]
    for m in range(1, m_windows + 1):
        start, end = wass_mod.get_lifted_indices(df, m, window, stride)
        out[start:end] = window_feats[m - 1]
    # Forward-fill any trailing uncovered rows.
    if np.isnan(out).any():
        last = None
        for i in range(data_len):
            if not np.isnan(out[i, 0]):
                last = out[i].copy()
            elif last is not None:
                out[i] = last
            else:
                out[i] = 0.0
    return out


def _wasserstein_dimred(
    df: pl.DataFrame,
    wass_window: int,
    wass_stride: int,
    regimes: int | None = None,
    opt_transport_reg: float = 0.001,
    **kwargs,
) -> DimredResults:
    """Per-time Wasserstein distances to cluster centroids (expanded from windows)."""
    raw_distributions = wass_mod.lift_datastream(df, wass_window, wass_stride)
    if not raw_distributions:
        raise ValueError("Wasserstein dimred produced no windows; check window/stride.")
    distributions = wass_mod.process_distributions(raw_distributions)
    distances = wass_mod.compute_distances(distributions, opt_transport_reg)

    n_regimes = regimes
    if n_regimes is None:
        logger.info("Wasserstein dimred: selecting regimes via silhouette.")
        n_regimes = wass_mod._determine_optimal_clusters_silhouette(
            distributions, distances
        )
    n_regimes = _require_regimes(n_regimes)

    labels = np.asarray(
        wass_mod._cluster_wasserstein_using_distributions_known_clusters(
            distributions, n_regimes, distance_matrix=distances
        ),
        dtype=int,
    )
    # Soft features: mean distance from each window to members of each cluster.
    m = len(labels)
    window_feats = np.zeros((m, n_regimes), dtype=float)
    for c in range(n_regimes):
        members = np.where(labels == c)[0]
        if members.size == 0:
            continue
        window_feats[:, c] = distances[:, members].mean(axis=1)

    expanded = _expand_window_features(df, window_feats, wass_window, wass_stride)
    return DimredResults(
        df=_as_frame(expanded, df, "wass_"),
        dimred="wasserstein",
        rank=n_regimes,
        rank_selection_method="regimes" if regimes is not None else "silhouette",
        model={"labels": labels.tolist(), "distances": distances},
    )


def _wasserstein_generator(df: pl.DataFrame, params: dict):
    if "wasserstein" not in params.get("algo", {}).get("dimred", []):
        return
    for wass_window in params["algo"]["wass_window"]:
        for wass_stride in params["algo"]["wass_stride"]:
            for regimes in params["algo"].get("regimes", [None]):
                for reg in params["algo"].get("opt_transport_reg", [0.001]):
                    yield _wasserstein_dimred(
                        df,
                        wass_window=wass_window,
                        wass_stride=wass_stride,
                        regimes=regimes,
                        opt_transport_reg=reg,
                    )


# ---------------------------------------------------------------------------
# Ruptures (pelt / window / dynp)
# ---------------------------------------------------------------------------

def _ruptures_dimred(
    df: pl.DataFrame,
    ruptures_algorithm: str,
    ruptures_cost_params: dict | None = None,
    regimes: int | None = None,
    ruptures_penalty: float | None = None,
    ruptures_window: int | None = None,
    ruptures_rec_error: float | None = None,
    **kwargs,
) -> DimredResults:
    """Distances to ruptures-segment means as embedding columns."""
    cost = ruptures_cost_params or {"model": "l2"}
    res = ruptures_mod.ruptures_predict_regimes(
        df,
        ruptures_algorithm=ruptures_algorithm,
        ruptures_cost_params=cost,
        regimes=regimes,
        ruptures_penalty=ruptures_penalty,
        ruptures_window=ruptures_window,
        ruptures_rec_error=ruptures_rec_error,
    )
    labels = np.asarray(res.labels, dtype=int)
    X = frames.to_numpy(df)
    uniq = sorted(set(labels.tolist()))
    if not uniq:
        uniq = [0]
        labels = np.zeros(df.height, dtype=int)

    means = []
    for lab in uniq:
        mask = labels == lab
        means.append(X[mask].mean(axis=0))
    means = np.vstack(means)
    dists = np.linalg.norm(X[:, None, :] - means[None, :, :], axis=2)
    return DimredResults(
        df=_as_frame(dists, df, "ruptures_"),
        dimred="ruptures",
        rank=dists.shape[1],
        rank_selection_method="regimes",
        model={
            "bkpts": res.bkpts,
            "labels": labels.tolist(),
            "algorithm": ruptures_algorithm,
        },
    )


def _ruptures_generator(df: pl.DataFrame, params: dict):
    if "ruptures" not in params.get("algo", {}).get("dimred", []):
        return
    algos = params["algo"].get("ruptures_algorithm", ["pelt"])
    costs = params["algo"].get("ruptures_cost_params", [{"model": "l2"}])
    for algo in algos:
        for cost in costs:
            if algo == "pelt":
                for pen in params["algo"].get("ruptures_penalty", [None]):
                    yield _ruptures_dimred(
                        df,
                        ruptures_algorithm=algo,
                        ruptures_cost_params=cost,
                        ruptures_penalty=pen,
                    )
            elif algo == "dynp":
                for regimes in params["algo"]["regimes"]:
                    yield _ruptures_dimred(
                        df,
                        ruptures_algorithm=algo,
                        ruptures_cost_params=cost,
                        regimes=regimes,
                    )
            elif algo == "window":
                for window in params["algo"]["ruptures_window"]:
                    if "regimes" in params["algo"]:
                        for regimes in params["algo"]["regimes"]:
                            yield _ruptures_dimred(
                                df,
                                ruptures_algorithm=algo,
                                ruptures_cost_params=cost,
                                ruptures_window=window,
                                regimes=regimes,
                            )
                    elif "ruptures_penalty" in params["algo"]:
                        for pen in params["algo"]["ruptures_penalty"]:
                            yield _ruptures_dimred(
                                df,
                                ruptures_algorithm=algo,
                                ruptures_cost_params=cost,
                                ruptures_window=window,
                                ruptures_penalty=pen,
                            )
                    else:
                        for eps in params["algo"]["ruptures_rec_error"]:
                            yield _ruptures_dimred(
                                df,
                                ruptures_algorithm=algo,
                                ruptures_cost_params=cost,
                                ruptures_window=window,
                                ruptures_rec_error=eps,
                            )
            else:
                raise ValueError(f"Unknown ruptures_algorithm for dimred: {algo}")


# ---------------------------------------------------------------------------
# TFT (attention embeddings)
# ---------------------------------------------------------------------------

def _attention_to_matrix(attn: dict, n_times: int) -> np.ndarray:
    """Average attention vectors across series categories onto a time×dim grid."""
    buckets: dict[int, list[np.ndarray]] = {}
    for _cat, by_t in attn.items():
        for t, vec in by_t.items():
            t_i = int(t)
            if 0 <= t_i < n_times:
                buckets.setdefault(t_i, []).append(np.asarray(vec, dtype=float).ravel())
    if not buckets:
        raise ValueError("TFT produced no attention vectors; check encoder/data length.")
    dim = next(iter(next(iter(buckets.values())))).shape[0]
    out = np.full((n_times, dim), np.nan, dtype=float)
    for t, vecs in buckets.items():
        out[t] = np.mean(np.vstack(vecs), axis=0)
    # Forward/back fill NaNs without pandas as primary frame type.
    for j in range(dim):
        col = out[:, j]
        last = np.nan
        for i in range(n_times):
            if not np.isnan(col[i]):
                last = col[i]
            elif not np.isnan(last):
                col[i] = last
        last = np.nan
        for i in range(n_times - 1, -1, -1):
            if not np.isnan(col[i]):
                last = col[i]
            elif not np.isnan(last):
                col[i] = last
        out[:, j] = col
    return out


def _tft_dimred(
    df: pl.DataFrame,
    rank: int,
    tft_encoder_length: int = 30,
    tft_prediction_length: int = 5,
    tft_max_epochs: int = 5,
    tft_training_cutoff: float = 0.8,
    tft_batch_size: int = 32,
    tft_mode: str = "univariate",
    random_state: int | None = 42,
    **kwargs,
) -> DimredResults:
    """Train a TFT and emit attention-vector embeddings (represent's dimred view)."""
    if rank is None or int(rank) < 1:
        raise ValueError("'rank' (hidden_size) must be a positive int for TFT dimred.")
    enc = int(tft_encoder_length)
    pred = int(tft_prediction_length)
    if df.height < enc + pred + 5:
        raise ValueError(
            f"TFT dimred needs len(df) >= encoder+prediction+5 "
            f"({enc}+{pred}+5); got {df.height}."
        )

    if tft_mode == "multivariate":
        training_set, validation_set = tft_mod.process_data_for_tft_multivariate(
            df,
            training_cutoff=tft_training_cutoff,
            min_encoder_length=enc,
            max_encoder_length=enc,
            min_prediction_length=pred,
            max_prediction_length=pred,
        )
    else:
        training_set, validation_set, _cats = tft_mod.process_data_for_tft_univariate(
            df,
            training_cutoff=tft_training_cutoff,
            min_encoder_length=enc,
            max_encoder_length=enc,
            min_prediction_length=pred,
            max_prediction_length=pred,
        )

    tft = tft_mod.train_tft(
        training_set,
        validation_set,
        rank=int(rank),
        max_epochs=int(tft_max_epochs),
        batch_size=int(tft_batch_size),
        enable_tensorboard=False,
        enable_mlflow=False,
        enable_model_summary=False,
        attention_head_size=kwargs.get("attention_head_size", 1),
        accelerator=kwargs.get("accelerator", "cpu"),
    )
    train_dl = training_set.to_dataloader(train=False, batch_size=int(tft_batch_size))
    val_dl = validation_set.to_dataloader(train=False, batch_size=int(tft_batch_size))
    attn = tft_mod.get_attention_vectors(tft, train_dl)
    attn_val = tft_mod.get_attention_vectors(tft, val_dl)
    for cat, by_t in attn_val.items():
        attn.setdefault(cat, {}).update(by_t)

    mat = _attention_to_matrix(attn, df.height)
    if mat.shape[1] != int(rank):
        n_comp = min(int(rank), mat.shape[0], mat.shape[1])
        pca = PCA(n_components=n_comp, random_state=random_state)
        projected = pca.fit_transform(mat)
        if projected.shape[1] < int(rank):
            pad = np.zeros((projected.shape[0], int(rank) - projected.shape[1]))
            projected = np.hstack([projected, pad])
        mat = projected
        model: dict = {"tft": tft, "pca": pca}
    else:
        model = {"tft": tft}

    return DimredResults(
        df=_as_frame(mat, df, "tft_"),
        dimred="tft",
        rank=mat.shape[1],
        rank_selection_method="user_specified",
        model=model,
    )


def _tft_generator(df: pl.DataFrame, params: dict):
    if "tft" not in params.get("algo", {}).get("dimred", []):
        return
    ranks = params["algo"].get("rank")
    if not ranks:
        raise ValueError("'rank' must be provided when using TFT dimred.")

    def _first(key, default):
        val = params["algo"].get(key, default)
        if isinstance(val, list):
            return val[0] if val else default
        return val

    for rank in ranks:
        for enc in params["algo"].get("tft_encoder_length", [30]):
            for pred in params["algo"].get("tft_prediction_length", [5]):
                for epochs in params["algo"].get("tft_max_epochs", [5]):
                    for mode in params["algo"].get("tft_mode", ["univariate"]):
                        yield _tft_dimred(
                            df,
                            rank=rank,
                            tft_encoder_length=enc,
                            tft_prediction_length=pred,
                            tft_max_epochs=epochs,
                            tft_training_cutoff=_first("tft_training_cutoff", 0.8),
                            tft_batch_size=_first("tft_batch_size", 32),
                            tft_mode=mode,
                        )


GENERATORS = {
    "bayesian_gmm": _bayesian_gmm_generator,
    "hmm": _hmm_generator,
    "kmeans": _kmeans_generator,
    "hdbscan": _hdbscan_generator,
    "optics": _optics_generator,
    "msar": _msar_generator,
    "wasserstein": _wasserstein_generator,
    "ruptures": _ruptures_generator,
    "tft": _tft_generator,
}
