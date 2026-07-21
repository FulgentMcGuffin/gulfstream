"""Drivers that execute Apache Hamilton dataflows for gulfstream."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import polars as pl

from gulfstream.common.results import SegmentResults

logger = logging.getLogger(__name__)


def _builder(*modules, config: dict | None = None):
    from hamilton import registry

    registry.disable_autoload()
    from hamilton import driver

    return (
        driver.Builder()
        .with_config(config or {})
        .with_modules(*modules)
        .build()
    )


def run_segmentation_pair(
    features_df: pl.DataFrame, params: dict
) -> tuple[SegmentResults, SegmentResults]:
    """Execute the single-pass segmentation Hamilton DAG."""
    import gulfstream.pipelines.hamilton.segmentation as seg_mod

    dr = _builder(seg_mod)
    out = dr.execute(
        ["segmentation_pair"],
        inputs={"features_df": features_df, "params": params},
    )
    return out["segmentation_pair"]


def load_features(
    source_config: str | Path,
    *,
    project_root: str | Path | None = None,
) -> tuple[pl.DataFrame, str]:
    """Execute the feature-load Hamilton DAG (source YAML → dated frame)."""
    import gulfstream.pipelines.hamilton.load_features as load_mod

    dr = _builder(load_mod)
    out = dr.execute(
        ["features_df", "source_type"],
        inputs={
            "source_config": source_config,
            "project_root": project_root,
        },
    )
    return out["features_df"], out["source_type"]


def visualize_segmentation_dag(
    path: str | Path,
    *,
    features_df: pl.DataFrame | None = None,
    params: dict | None = None,
) -> Path | None:
    """Write a PNG of the segmentation DAG when graphviz extras are available."""
    import gulfstream.pipelines.hamilton.segmentation as seg_mod

    path = Path(path)
    dr = _builder(seg_mod)
    try:
        inputs: dict[str, Any] = {}
        if features_df is not None:
            inputs["features_df"] = features_df
        if params is not None:
            inputs["params"] = params
        dr.visualize_execution(
            ["segmentation_pair"],
            str(path.with_suffix("")),
            {"format": "png"},
            inputs=inputs or None,
        )
        png = path if path.suffix.lower() == ".png" else path.with_suffix(".png")
        logger.info("Wrote Hamilton segmentation DAG to %s", png)
        return png
    except Exception:
        logger.debug("Hamilton visualize_execution unavailable", exc_info=True)
        return None
