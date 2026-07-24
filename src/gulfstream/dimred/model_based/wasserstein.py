"""Wasserstein model-based dimred (delegates to legacy.detectors.wasserstein)."""
from __future__ import annotations

import logging

import numpy as np
import polars as pl

from gulfstream.common.options import DimredMethod
from gulfstream.common.results import DimredResults
from gulfstream.dimred.model_based._common import _as_frame, _require_regimes
from gulfstream.legacy.detectors import wasserstein as wass_mod

logger = logging.getLogger(__name__)


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
    if DimredMethod.WASSERSTEIN not in params.get("algo", {}).get("dimred", []):
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
