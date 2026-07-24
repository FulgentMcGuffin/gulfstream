"""Main frontends for Graph 1 core regime detection."""
from __future__ import annotations

import logging
import os
from typing import NoReturn

import polars as pl

from gulfstream.detection import algorithm as bkpt_algo
from gulfstream.detection import stat_tests as bkpt_stat_tests
from gulfstream.detection import time_index as bkpt_timeindexing_conversions
from gulfstream.dimred import dispatcher as dimension_reduction
from gulfstream.features import names as feature_name_resolution
from gulfstream.common import frames
from gulfstream.detection import hyperparams as custom_hyperparameter_selection
from gulfstream.common.config import coerce_params
from gulfstream.features import kernel_map as kernel_feature_mapping
from gulfstream.common import logging as logging_config
from gulfstream.detection import postprocess as output_postprocessing
from gulfstream.metrics import writers as results_writers
from gulfstream.common import utils
from gulfstream.common.results import SegmentResults
from gulfstream.pipelines._shared import (
    _initialize_results_writer_and_dir,
    _initialize_test_sub_dir,
    _log_missing_columns,
    produce_all_metrics,
)
from gulfstream.pipelines.single_pass import run_single_segmentation_pair

logger = logging.getLogger(__name__)


def _handle_full_feature_map_case(df: pl.DataFrame, params: dict):
    for res in kernel_feature_mapping.feature_map_generator(df, params):
        mapped_dfs = res.dfs
        mapped_df_diffs = res.df_diffs
        mapping_params = {
            "feature_map_kernel_params": res.kernel_params,
            "feature_map_approx_method": res.feature_map_approx_method,
            "num_features": res.num_features,
            "num_mappings": len(mapped_dfs),
        }
        if res.kernel_approx_error:
            mapping_params["kernel_approx_error"] = res.kernel_approx_error
        if res.diff_kernel_params:
            mapping_params["diff_kernel_params"] = res.diff_kernel_params
        yield mapping_params, [mapped_dfs], [mapped_df_diffs]


def _handle_pca_feature_map_case(df: pl.DataFrame, params: dict):
    for res_list in kernel_feature_mapping.pcafeature_map_generator(df, params):
        mapped_dfs = [res.dfs for res in res_list]
        mapped_df_diffs = [res.df_diffs for res in res_list]
        mapping_params = []
        for res in res_list:
            this_pc_params = {
                "feature_map_kernel_params": res.kernel_params,
                "feature_map_approx_method": res.feature_map_approx_method,
                "num_features": res.num_features,
                "num_mappings": res.num_mappings,
            }
            if res.kernel_approx_error:
                this_pc_params["kernel_approx_error"] = res.kernel_approx_error
            if res.diff_kernel_params:
                this_pc_params["diff_kernel_params"] = res.diff_kernel_params
            mapping_params.append(this_pc_params)
        mapping_params = output_postprocessing.combine_params(mapping_params)
        yield mapping_params, mapped_dfs, mapped_df_diffs


CASE_HANDLERS = {
    "full": _handle_full_feature_map_case,
    "iterative_pca": _handle_pca_feature_map_case,
}


def _need_case(case: str, params: dict) -> bool:
    return case in params["algo"].get("recursive_method", [])


def get_dfs_for_case(df: pl.DataFrame, case: str) -> list:
    if case == "full":
        return [df]
    if case == "iterative_pca":
        return [frames.select_features(df, [col]) for col in frames.feature_columns(df)]
    raise ValueError(f"Unknown case {case}.")


def _process_cases(
    case_name: str,
    df: pl.DataFrame,
    df_dimred: pl.DataFrame,
    df_pca: pl.DataFrame | None,
    params: dict,
    misc_params: dict,
    algo_params: dict,
):
    row = misc_params["row"]
    test_num = misc_params["test_num"]
    handler = CASE_HANDLERS.get(case_name)
    if handler is None:
        raise ValueError(f"Unknown case: {case_name}.")
    if not _need_case(case_name, params):
        logger.info("Case %s not needed. Skipping.", case_name)
        return row, test_num

    dates = bkpt_timeindexing_conversions.get_strs_from_df_index(df)
    dfs = get_dfs_for_case(df_dimred, case_name)

    for items in handler(df_dimred, params):
        for ruptures_params in bkpt_algo.kernel_ruptures_generator(dfs, params):
            for test_params in bkpt_stat_tests.test_param_combos(params):
                for late_algo_params in bkpt_algo.late_algo_param_combos(params):
                    case_params = {
                        "test": test_params.copy(),
                        "algo": algo_params.copy(),
                    }
                    case_params["algo"].update(late_algo_params)
                    case_params["algo"].update(items[0])
                    case_params.update(misc_params)
                    case_params["row"] = row
                    case_params["test_num"] = test_num

                    results = []
                    df_grids = (items[1], items[2])
                    logger.info("Starting to test breakpoints (test_num=%s).", test_num)
                    for i, (mapped_dfs, mapped_df_diffs) in enumerate(zip(*df_grids)):
                        case_params["algo"]["ruptures_kernel_params"] = ruptures_params[i]
                        raw_res = bkpt_algo.find_and_test_bkpts(
                            dfs[i],
                            mapped_dfs,
                            case_params,
                            dates,
                            mapped_df_diffs=mapped_df_diffs,
                            df_pca=df_pca,
                        )
                        results.append(raw_res)

                    combined = output_postprocessing.combine_results(df_dimred.height, results)
                    combined_converted = bkpt_timeindexing_conversions.convert_results(
                        combined, df.height
                    )
                    case_algo_params = case_params["algo"]
                    for processing_params in output_postprocessing.post_processing_params_generator(
                        case_params["test"]["choice"], params
                    ):
                        case_params["algo"] = case_algo_params.copy()
                        case_params["algo"].update(processing_params)
                        processed = output_postprocessing.post_process(
                            res=combined,
                            processing_params=processing_params,
                            params=case_params,
                            length=df.height,
                            df_dimred=df_dimred,
                            mapped_dfs=items[1][0],
                            mapped_df_diffs=items[2][0],
                            df_pca=df_pca,
                        )
                        case_params["metrics"]["image_dir"] = _initialize_test_sub_dir(
                            case_params, misc_params["image_dir"]
                        )
                        results_writers.report_regime_statistics(
                            df, case_params, processed, combined_converted
                        )
                        if case_params["metrics"].get("developer_mode", False):
                            row = results_writers.report_performance(
                                df, case_params, processed, combined_converted
                            )
                        test_num += 1
                        case_params["row"] = row
                        case_params["test_num"] = test_num
                        case_params["robustness"] = params.get("robustness") or {}
                        case_params["stability"] = params.get("stability") or {}
                        case_params["_pipeline_params"] = params
                        produce_all_metrics(df, processed, case_params)
    return row, test_num


def _main_driver(df: pl.DataFrame, params: dict, misc_params: dict):
    if custom_hyperparameter_selection.asked_for_acf_lag_selection(params):
        df_pca = custom_hyperparameter_selection.calculate_pca_for_lag_selection(df)
    else:
        df_pca = None

    test_num = misc_params["test_num"]
    row = misc_params["row"]
    for res in dimension_reduction.dimred_generator(df, params):
        df_dimred = res.df
        algo_params = dimension_reduction.get_dimred_param_dict(res)
        for case_name in CASE_HANDLERS:
            algo_params["recursive_method"] = case_name
            try:
                row, test_num = _process_cases(
                    case_name, df, df_dimred, df_pca, params, misc_params, algo_params
                )
                misc_params["row"] = row
                misc_params["test_num"] = test_num
            except Exception:
                logger.exception("Failed processing case %s.", case_name)
    return test_num, row


def run_graph1(df: pl.DataFrame, params: dict) -> SegmentResults | None:
    """Run Graph 1 on a user-supplied feature DF; return the last segmentation.

    When the algo grid collapses to a single combo, the Hamilton single-pass
    path is preferred for the returned result; the full grid driver still
    writes artifacts for every combo.
    """
    try:
        params = coerce_params(params)
    except Exception:
        logger.exception("Invalid parameters; aborting.")
        return None
    last_result: SegmentResults | None = None
    with logging_config.LoggingContext(params["log"]["dir"], log_level=params["log"]["level"]):
        try:
            image_dir, results_writer, template = _initialize_results_writer_and_dir(params)
        except Exception:
            logger.exception("Failed to initialize results writer and directory.")
            return None

        feat_cols = set(frames.feature_columns(df))
        if isinstance(params["metrics"].get("explainability_features"), dict):
            exp_features = feature_name_resolution.get_column_names(
                params["metrics"].get("explainability_features")
            )
        else:
            exp_features = params["metrics"].get("explainability_features")
        if exp_features:
            missing = [f for f in exp_features if f not in feat_cols]
            if missing:
                logger.warning("Missing explainability features: %s", ", ".join(missing))

        # Prefer Hamilton single-pass for the returned result (deduped core).
        try:
            _unproc, last_result = run_single_segmentation_pair(df, params)
        except Exception:
            logger.exception("Hamilton single-pass failed; continuing with grid driver.")

        misc_params = {
            "test_num": 0,
            "row": 0,
            "results_writer": results_writer,
            "image_dir": image_dir,
            "template": template,
            "metrics": params["metrics"],
            "data_params": {},
        }
        _main_driver(df, params, misc_params)
        if results_writer is not None:
            results_writer.close()
        if image_dir is not None:
            utils.generate_gallery(image_dir)
        logger.info("Done. Outputs in %s", image_dir)
    return last_result
