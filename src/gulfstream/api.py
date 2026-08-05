"""Public programmatic API for gulfstream.

Prefer these entry points over importing pipeline internals. All functions
accept either a typed :class:`~gulfstream.common.config.Config` or a plain
params dict (validated on the way in).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from gulfstream.common.config import Config, coerce_params
from gulfstream.common.results import AlgoResults, SegmentResults
from gulfstream.detection.time_index import get_regime_intervals
from gulfstream.metrics.regime_plots import plot_market_regimes
from gulfstream.pipelines.graph1 import run_graph1
from gulfstream.pipelines.graph2 import run_graph2, seed_regimes_from_results
from gulfstream.pipelines.hamilton.driver import load_features as _load_features
from gulfstream.pipelines.panel import run_panel_joint_segmentation
from gulfstream.pipelines.single_pass import run_single_segmentation
from gulfstream.pipelines.streaming import (
    IncrementalState,
    advance_incremental,
    run_streaming_graph1,
)


def load_features(
    source_yaml: str | Path,
    *,
    project_root: str | Path | None = None,
) -> pl.DataFrame:
    """Load a dated feature matrix from a source YAML."""
    df, _source_type = _load_features(source_yaml, project_root=project_root)
    return df


def detect_regimes(
    df: pl.DataFrame,
    config: Config | dict[str, Any],
) -> SegmentResults | None:
    """Run Graph 1 and return a segmentation result.

    Uses ``algo.detection_backend`` (``kernel_ruptures`` or ``classical``).
    When ``streaming.enabled`` or ``panel.enabled`` is set in config, routes to
    the corresponding product path. Side-effecting artifact writes still follow
    ``metrics.mode`` / ``metrics.dir``.
    """
    params = coerce_params(config)
    return run_graph1(df, params)


def detect_regimes_incremental(
    df: pl.DataFrame,
    config: Config | dict[str, Any],
    state: IncrementalState | None = None,
) -> tuple[SegmentResults, IncrementalState]:
    """Advance streaming Graph 1 by one step (or start a new stream).

    For a full expanding/rolling pass to the series end, prefer
    :func:`detect_regimes` with ``streaming.enabled: true``.
    """
    params = coerce_params(config)
    params.setdefault("streaming", {})["enabled"] = True
    return advance_incremental(df, params, state)


def detect_regimes_panel(
    df: pl.DataFrame,
    config: Config | dict[str, Any],
) -> SegmentResults:
    """Joint breakpoints across panel groups (``panel.enabled``)."""
    params = coerce_params(config)
    params.setdefault("panel", {})["enabled"] = True
    return run_panel_joint_segmentation(df, params)


def refine_regimes(
    df: pl.DataFrame,
    config: Config | dict[str, Any],
    seed: SegmentResults | pl.DataFrame | None = None,
) -> SegmentResults | None:
    """Run Graph 2, optionally seeded from a Graph 1 ``SegmentResults``."""
    params = coerce_params(config)
    if isinstance(seed, SegmentResults):
        params.setdefault("retrain", {})
        params["retrain"]["regimes_df"] = seed_regimes_from_results(df, seed)
    elif seed is not None:
        params.setdefault("retrain", {})
        params["retrain"]["regimes_df"] = seed
    if "retrain" not in params:
        raise ValueError("refine_regimes requires a 'retrain' section in config.")
    return run_graph2(df, params)


def plot_regimes(
    df: pl.DataFrame,
    results: SegmentResults,
    *,
    variables: list[str] | None = None,
    title: str = "",
    mode: str = "display",
    img_dir: str | None = None,
    emit: bool = True,
):
    """Plot regime shading for selected features; return the plotnine ggplot."""
    dates = df["date"].to_list() if "date" in df.columns else list(range(df.height))
    hierarchy = results.hierarchy or {b: 1 for b in results.bkpts}
    regimes_df = get_regime_intervals(hierarchy, dates)
    return plot_market_regimes(
        df,
        regimes_df,
        title=title,
        variables=variables,
        valid_bkpts=results.bkpts,
        invalid_bkpts=results.invalid_bkpts,
        low_confidence_bkpts=list(results.low_confidence_bkpts or []),
        bkpt_ci=dict(results.bkpt_ci or {}),
        plot_ci_ribbons=True,
        mode=mode,
        img_dir=img_dir,
        emit=emit,
    )


def regime_intervals(
    results: SegmentResults,
    dates: list,
) -> pl.DataFrame:
    """Build a Start/End/Regime table from breakpoint hierarchy."""
    hierarchy = results.hierarchy or {b: 1 for b in results.bkpts}
    return get_regime_intervals(hierarchy, dates)


__all__ = [
    "load_features",
    "detect_regimes",
    "detect_regimes_incremental",
    "detect_regimes_panel",
    "refine_regimes",
    "plot_regimes",
    "regime_intervals",
    "run_single_segmentation",
    "run_streaming_graph1",
    "advance_incremental",
    "IncrementalState",
    "seed_regimes_from_results",
    "Config",
    "SegmentResults",
    "AlgoResults",
]
