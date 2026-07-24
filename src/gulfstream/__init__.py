"""Gulfstream — regime detection pipelines (Graph 1 / Graph 2 / legacy)."""

from gulfstream.api import (
    Config,
    detect_regimes,
    load_features,
    plot_regimes,
    refine_regimes,
    regime_intervals,
    run_legacy_detector,
    run_single_segmentation,
    seed_regimes_from_results,
)
from gulfstream.common.results import AlgoResults, SegmentResults

__all__ = [
    "Config",
    "SegmentResults",
    "AlgoResults",
    "load_features",
    "detect_regimes",
    "refine_regimes",
    "run_legacy_detector",
    "plot_regimes",
    "regime_intervals",
    "run_single_segmentation",
    "seed_regimes_from_results",
]
