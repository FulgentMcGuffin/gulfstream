"""Gulfstream — regime detection pipelines (Graph 1 / Graph 2)."""

from gulfstream.api import (
    Config,
    IncrementalState,
    advance_incremental,
    detect_regimes,
    detect_regimes_incremental,
    detect_regimes_panel,
    load_features,
    plot_regimes,
    refine_regimes,
    regime_intervals,
    run_single_segmentation,
    run_streaming_graph1,
    seed_regimes_from_results,
)
from gulfstream.common.results import AlgoResults, SegmentResults

__all__ = [
    "Config",
    "SegmentResults",
    "AlgoResults",
    "IncrementalState",
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
    "seed_regimes_from_results",
]
