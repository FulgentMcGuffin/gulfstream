"""Single-run segmentation helper shared by robustness/stability / Graph 2.

The linear core is an Apache Hamilton DAG
(``gulfstream.pipelines.hamilton.segmentation``); this module is the thin
public API used by metrics and Graph 2.
"""
from __future__ import annotations

import logging

import polars as pl

from gulfstream.common.results import SegmentResults
from gulfstream.pipelines.hamilton.driver import run_segmentation_pair as _hamilton_pair

logger = logging.getLogger(__name__)


def run_single_segmentation(df: pl.DataFrame, params: dict) -> SegmentResults:
    """Run one full→RFF→ruptures→postprocess pass and return processed results."""
    _, processed = run_single_segmentation_pair(df, params)
    return processed


def run_single_segmentation_pair(
    df: pl.DataFrame, params: dict
) -> tuple[SegmentResults, SegmentResults]:
    """Return ``(unprocessed_local, processed_local)`` for one Graph 1 pass."""
    return _hamilton_pair(df, params)
