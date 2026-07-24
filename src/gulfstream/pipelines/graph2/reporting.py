"""Prep dirs, heatmaps, and finalize reporting for Graph 2."""
from __future__ import annotations

import copy
import datetime as dt
import logging
import os

import numpy as np
import polars as pl

from gulfstream.common import utils
from gulfstream.common.results import SegmentResults
from gulfstream.detection import time_index as bkpt_timeindexing_conversions
from gulfstream.metrics import evaluation as evaluation_tools
from gulfstream.metrics import insights as post_information_visualization
from gulfstream.metrics import writers as results_writers
from gulfstream.pipelines._shared import produce_all_metrics

logger = logging.getLogger(__name__)


def _retrain_prep_dir(params: dict) -> tuple[str, str]:
    """Create ``metrics.dir/bkpt_tests_{ts}/0/`` for heatmaps and gallery."""
    metrics_dir = params.get("metrics", {}).get("dir")
    if not metrics_dir:
        raise ValueError("You must provide a path to a directory for saving test results.")
    os.makedirs(metrics_dir, exist_ok=True)
    current_time = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    image_dir = os.path.join(metrics_dir, f"bkpt_tests_{current_time}")
    os.makedirs(image_dir, exist_ok=True)
    relative_dir = os.path.relpath(image_dir, start=os.getcwd())
    sub_dir = os.path.join(image_dir, "0")
    os.makedirs(sub_dir, exist_ok=True)
    return relative_dir, sub_dir


def _heatmap_helper(
    df: pl.DataFrame,
    bkpts: list[int],
    iters: int,
    image_dir: str,
) -> tuple[np.ndarray, tuple[int, int]]:
    logger.info("Generating L2 error matrix.")
    loss_matrix = evaluation_tools.avg_features_loss(df, bkpts)
    name = f"retrain_iteration_{iters}_avg_feature_L2"
    post_information_visualization.draw_error_heatmaps(
        loss_matrix,
        df,
        bkpts,
        title=f"Retrain {iters}: Average daily L2 loss per feature in each regime (in bps)",
        cbar_label="average L2 dist to mean (measured in bps)",
        mode="write",
        img_dir=image_dir,
        gallery_filename=name,
    )
    logger.info("Saved error matrix to %s.", image_dir + utils.img_gallery_filename(name))
    return loss_matrix, tuple(np.unravel_index(np.argmax(loss_matrix), loss_matrix.shape))


def _finalize_results(
    df_full: pl.DataFrame,
    params: dict,
    *,
    unprocessed_bkpts: list[int],
    processed_bkpts: list[int],
    invalid_bkpts: list[int],
    unproc_bkpt_stats: dict,
    bkpt_stats: dict,
    bkpt_index_dict: dict,
    relative_dir: str,
    image_dir: str,
) -> SegmentResults:
    mode = params["metrics"].get("mode", "write")
    plot = params["metrics"].get("plot", False)
    params_copy = copy.deepcopy(params)
    params_copy["metrics"] = dict(params["metrics"])
    params_copy["row"] = 0
    params_copy["test_num"] = 0

    if mode in ("write", "display_and_write"):
        params_copy["metrics"]["image_dir"] = image_dir
        results_sheet_name = os.path.join(image_dir, "results.xlsx")
        import pandas as pd
        params_copy["results_writer"] = pd.ExcelWriter(results_sheet_name, engine="openpyxl")
        params_copy["template"] = results_writers.write_results_header(
            params_copy, params_copy["results_writer"]
        )
    else:
        params_copy["metrics"]["image_dir"] = None
        params_copy["results_writer"] = None
        params_copy["template"] = None

    unproc_res = SegmentResults(
        bkpts=sorted(unprocessed_bkpts),
        invalid_bkpts=sorted(invalid_bkpts),
        stats=unproc_bkpt_stats,
        hierarchy={},
        labels=bkpt_timeindexing_conversions.bkpts_to_labels(
            sorted(unprocessed_bkpts), df_full.height
        ),
        params=params_copy,
    )
    # Use merged processed_bkpts (not the seed list alone).
    proc_res = SegmentResults(
        bkpts=sorted(processed_bkpts),
        invalid_bkpts=sorted(invalid_bkpts),
        stats=bkpt_stats,
        hierarchy=dict(bkpt_index_dict),
        labels=bkpt_timeindexing_conversions.bkpts_to_labels(
            sorted(processed_bkpts), df_full.height
        ),
        params=params_copy,
    )

    results_writers.report_regime_statistics(df_full, params_copy, proc_res, unproc_res)
    if params_copy["metrics"].get("developer_mode", False):
        results_writers.report_performance(df_full, params_copy, proc_res, unproc_res)

    if plot:
        logger.info("Producing all explainability and visualization tools.")
        # Ensure nested stages see robustness/stability toggles on params_copy.
        params_copy["robustness"] = params.get("robustness") or {}
        params_copy["stability"] = params.get("stability") or {}
        produce_all_metrics(df_full, proc_res, params_copy)

    if params_copy.get("results_writer") is not None:
        params_copy["results_writer"].close()

    if relative_dir is not None:
        utils.generate_gallery(relative_dir)

    return proc_res
