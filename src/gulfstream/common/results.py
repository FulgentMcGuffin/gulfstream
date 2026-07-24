"""Result containers for the custom regime detection pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl


@dataclass
class AlgoResults:
    """Hard-label classical detector results (labels → breakpoints)."""

    bkpts: list[int]
    labels: list[int]
    params: dict | None = None


@dataclass
class SegmentResults:
    """Results from one run of the custom breakpoint algorithm."""

    bkpts: list[int]
    invalid_bkpts: list[int] = field(default_factory=list)
    stats: dict[int, Any] = field(default_factory=dict)
    hierarchy: dict[int, Any] = field(default_factory=dict)
    labels: list[int] | None = None
    params: dict | None = None
    persistence: dict[int, float] = field(default_factory=dict)
    low_confidence_bkpts: list[int] = field(default_factory=list)
    stability_score: float | None = None
    # Product: calibrated index bands ``bkpt -> (lo, hi)``
    bkpt_ci: dict[int, tuple[int, int]] = field(default_factory=dict)
    # Product: fraction of panel groups supporting each consensus break
    panel_support: dict[int, float] = field(default_factory=dict)


@dataclass
class DimredResults:
    """Results of a dimensionality-reduction step."""

    df: pl.DataFrame
    dimred: str
    rank: int | None = None
    rank_selection_method: str | None = None
    threshold: float | None = None
    eigvals: np.ndarray | None = None
    kernel_params: dict | None = None
    dmd_stride: int | None = None
    dmd_rolling_window: int | None = None
    tsne_perplexity: float | None = None
    tsne_n_iter: int | None = None
    umap_num_neighbors: float | None = None
    umap_min_dist: float | None = None
    umap_metric: str | None = None
    ica_max_iter: int | None = None
    ns_lambda: float | None = None
    factor_order: int | None = None
    fpca_smooth_window: int | None = None
    model: Any = None


@dataclass
class MappingResults:
    """Single kernel feature-map realization."""

    df: pl.DataFrame
    feature_map_approx_method: str
    num_features: int
    model: Any = None
    model_info: dict | None = None
    kernel_params: dict | None = None


@dataclass
class ManyMappingResults:
    """Multiple kernel feature-map realizations (e.g. several RFF seeds)."""

    dfs: list[pl.DataFrame]
    kernel_params: dict
    feature_map_approx_method: str
    num_features: int
    kernel_approx_error: float | None = None
    df_diffs: list[pl.DataFrame] | None = None
    diff_kernel_params: dict | None = None

    @property
    def num_mappings(self) -> int:
        return len(self.dfs)


@dataclass
class HyperparameterResults:
    """Selected lag / window / sample_size hyperparameters."""

    lag: dict | None = None
    window: dict | None = None
    sample_size: dict | None = None
