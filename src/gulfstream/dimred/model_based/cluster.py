"""Clustering-based dimred: Bayesian GMM, k-means, HDBSCAN, OPTICS."""
from __future__ import annotations

import math

import polars as pl
from sklearn.cluster import KMeans
from sklearn.mixture import BayesianGaussianMixture

from gulfstream.common import frames
from gulfstream.common.options import DimredMethod
from gulfstream.common.results import DimredResults
from gulfstream.dimred import density as dens
from gulfstream.dimred.model_based._common import _as_frame, _require_regimes


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
    if DimredMethod.BAYESIAN_GMM not in params.get("algo", {}).get("dimred", []):
        return
    for regimes in params["algo"]["regimes"]:
        for reg_covar in params["algo"].get("reg_covar", [1e-5]):
            yield _bayesian_gmm_dimred(df, regimes=regimes, reg_covar=reg_covar)


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
    if DimredMethod.KMEANS not in params.get("algo", {}).get("dimred", []):
        return
    for regimes in params["algo"]["regimes"]:
        for random_state in params["algo"].get("random_state", [None]):
            yield _kmeans_dimred(df, regimes=regimes, random_state=random_state)


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
    if DimredMethod.HDBSCAN not in params.get("algo", {}).get("dimred", []):
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
    if DimredMethod.OPTICS not in params.get("algo", {}).get("dimred", []):
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
