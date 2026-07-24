"""Helpers shared by Graph 1, Graph 2, and legacy pipelines."""
from __future__ import annotations

import datetime as dt
import logging
import os

import polars as pl

from gulfstream.common import frames
from gulfstream.common.results import SegmentResults
from gulfstream.detection import trees as bkpt_trees
from gulfstream.metrics import explainability as explainability_tools
from gulfstream.metrics import insights as post_information_visualization
from gulfstream.metrics import regime_plots as regime_visualization_tools
from gulfstream.metrics import robustness
from gulfstream.metrics import stability
from gulfstream.metrics import transitions as transition_matrices
from gulfstream.metrics import writers as results_writers

logger = logging.getLogger(__name__)


def _log_missing_columns(df: pl.DataFrame, features: list | None) -> None:
    if features is None:
        return
    cols = set(frames.feature_columns(df))
    missing = [f for f in features if f not in cols]
    if missing:
        logger.warning(
            "Missing the following requested features for explainability: %s",
            ", ".join(str(f) for f in missing),
        )


def _initialize_results_writer_and_dir(params: dict):
    import pandas as pd

    save = params["metrics"].get("mode", "write") in ["write", "display_and_write"]
    if save:
        if not params["metrics"].get("dir"):
            raise ValueError("Must provide metrics.dir for saving results.")
        os.makedirs(params["metrics"]["dir"], exist_ok=True)
        current_time = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        image_dir = os.path.join(params["metrics"]["dir"], f"bkpt_tests_{current_time}")
        os.makedirs(image_dir, exist_ok=True)
        results_sheet = os.path.join(image_dir, "results.xlsx")
        results_writer = pd.ExcelWriter(results_sheet, engine="openpyxl")
        template = results_writers.write_results_header(params, results_writer)
    else:
        image_dir = None
        results_writer = None
        template = None
    return image_dir, results_writer, template


def _initialize_test_sub_dir(params: dict, file_dir: str | None) -> str | None:
    save = params["metrics"].get("mode", "write") in ["write", "display_and_write"]
    if save and file_dir:
        sub_dir = os.path.join(file_dir, f"{params['test_num']}_{params['test']['choice']}")
        os.makedirs(sub_dir, exist_ok=True)
        return sub_dir
    return None


def produce_all_metrics(df: pl.DataFrame, res: SegmentResults, params: dict) -> None:
    if not params["metrics"].get("plot"):
        return
    res = robustness.evaluate_robustness(df, params, res)
    res = stability.evaluate_stability(df, params, res)

    regime_visualization_tools.produce_all_regime_visualization_tools(df, params, res)
    post_information_visualization.produce_all_post_information_visualization_tools(
        df, params, res
    )
    bkpt_trees.build_and_plot_bkpt_trees(df, params, res, unproc_res=None)
    explainability_tools.produce_all_explainability_tools(df, params, res)
    transition_matrices.produce_transition_matrices(df, params)
