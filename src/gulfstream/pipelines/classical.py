"""Classical hard-label detection path for Graph 1 / Graph 2.

Activated when ``algo.detection_backend`` includes ``classical``. Runs
dimred → optional feature map → classical detector → classical post-process,
returning ``SegmentResults`` so Graph 2 can seed/refine like kernel_ruptures.
"""
from __future__ import annotations

import logging
import os
from typing import Iterator

import polars as pl

from gulfstream.common import frames, utils
from gulfstream.common.options import DetectionBackend
from gulfstream.common.results import AlgoResults, SegmentResults
from gulfstream.detection import postprocess as output_postprocessing
from gulfstream.detection import time_index as bkpt_timeindexing_conversions
from gulfstream.detectors import CLASSICAL_METHODS, CLASSICAL_PARAM_GENERATORS
from gulfstream.dimred import dispatcher as dimension_reduction
from gulfstream.features import kernel_map as kernel_feature_mapping
from gulfstream.metrics import explainability as explainability_tools
from gulfstream.metrics import regime_plots
from gulfstream.metrics import writers as results_writers

logger = logging.getLogger(__name__)


def uses_classical_backend(params: dict) -> bool:
    backends = params.get("algo", {}).get("detection_backend") or [
        DetectionBackend.KERNEL_RUPTURES
    ]
    return any(str(b) == DetectionBackend.CLASSICAL for b in backends)


def run_classical_algo(method: str, df: pl.DataFrame, params: dict) -> AlgoResults:
    handler = CLASSICAL_METHODS.get(method)
    if not handler:
        raise ValueError(f"Unknown regime detection algorithm {method}.")
    res = handler(df, **params["algo"])
    res.params = params
    return res


def classical_method_params_generator(params: dict) -> Iterator[dict]:
    for gen in CLASSICAL_PARAM_GENERATORS.values():
        yield from gen(params)


def handle_classical_feature_map(df: pl.DataFrame, params: dict):
    for res in kernel_feature_mapping.classical_feature_map_generator(df, params):
        mapped_df = res.dfs[0]
        mapping_params = {
            "feature_map_kernel_params": res.kernel_params,
            "feature_map_approx_method": res.feature_map_approx_method,
            "num_features": res.num_features,
        }
        if res.kernel_approx_error:
            mapping_params["kernel_approx_error"] = res.kernel_approx_error
        yield mapping_params, mapped_df


def run_classical_segmentation_pair(
    df: pl.DataFrame, params: dict
) -> tuple[SegmentResults, SegmentResults]:
    """Single classical pass: first dimred × method × postprocess combo."""
    dimred_res = next(dimension_reduction.dimred_generator(df, params))
    df_dimred = dimred_res.df
    algo_params = dimension_reduction.get_dimred_param_dict(dimred_res)
    mapping_params, df_mapped = next(handle_classical_feature_map(df_dimred, params))
    method_params = next(classical_method_params_generator(params))
    method = method_params["regime_detection_algorithm"]

    case_params: dict = {"algo": {**algo_params, **mapping_params, **method_params}}
    case_params["metrics"] = params.get("metrics", {})
    case_params["test"] = params.get("test", {})

    raw = run_classical_algo(method, df_mapped, case_params)
    unproc = output_postprocessing.algo_results_to_segment(
        bkpt_timeindexing_conversions.convert_results(raw, df.height)
    )
    processing_params = next(
        output_postprocessing.classical_post_processing_params_generator(params)
    )
    case_params["algo"].update(processing_params)
    processed = output_postprocessing.classical_post_process(
        res=raw,
        processing_params=processing_params,
        params=case_params,
        length=df.height,
        df_dimred=df_dimred,
    )
    return unproc, processed


def _initialize_classical_test_sub_dir(params: dict, file_dir: str | None) -> str | None:
    save = params["metrics"].get("mode", "write") in ["write", "display_and_write"]
    if save and file_dir:
        sub_dir = os.path.join(
            file_dir,
            f"{params['test_num']}_{params['algo']['regime_detection_algorithm']}",
        )
        os.makedirs(sub_dir, exist_ok=True)
        return sub_dir
    return None


def _classical_regime_plot(
    df: pl.DataFrame,
    variables: list,
    invalid_bkpts: list,
    valid_bkpts: list,
    labels: list,
    bkpt_hierarchy: dict,
    mode: str = "display",
    img_dir: str | None = None,
    **kwargs,
) -> None:
    date_index = frames.dates_series(df).to_list()
    regimes_df = bkpt_timeindexing_conversions.get_regime_intervals_from_labels(
        labels, date_index
    )
    regime_plots.plot_market_regimes(
        df,
        regimes_df,
        title="Regime plot",
        variables=variables,
        invalid_bkpts=invalid_bkpts or [],
        valid_bkpts=valid_bkpts or [],
        bkpt_hierarchy=bkpt_hierarchy or {},
        mode=mode,
        img_dir=img_dir,
    )


def classical_produce_all_metrics(
    df: pl.DataFrame, params: dict, res: SegmentResults
) -> None:
    """Thin metrics wrapper for classical runs (label-based regime plot)."""
    if not params["metrics"].get("plot"):
        return
    df_explainable = explainability_tools._get_explainability_df(
        df, params["metrics"].get("explainability_features")
    )
    _classical_regime_plot(
        df=df_explainable,
        valid_bkpts=res.bkpts,
        labels=res.labels or [],
        mode=params["metrics"]["mode"],
        img_dir=params["metrics"].get("image_dir"),
        variables=params["metrics"].get("features_to_plot"),
        invalid_bkpts=[],
        bkpt_hierarchy=res.hierarchy or {},
    )


def classical_process_cases(
    df: pl.DataFrame,
    df_dimred: pl.DataFrame,
    df_mapped: pl.DataFrame,
    params: dict,
    misc_params: dict,
    algo_params: dict,
) -> tuple[int, int, SegmentResults | None]:
    row = misc_params["row"]
    test_num = misc_params["test_num"]
    last: SegmentResults | None = None
    for method_params in classical_method_params_generator(params):
        method = method_params["regime_detection_algorithm"]
        if method not in CLASSICAL_METHODS:
            raise ValueError(f"Unknown regime detection algorithm {method}.")
        case_params: dict = {"algo": algo_params.copy()}
        case_params["algo"].update(method_params)
        case_params.update(misc_params)
        case_params["row"] = row
        case_params["test_num"] = test_num
        case_params.setdefault("test", params.get("test", {}))

        logger.info("Running classical detector %s.", method)
        raw_res = run_classical_algo(method, df_mapped, case_params)
        unproc = output_postprocessing.algo_results_to_segment(
            bkpt_timeindexing_conversions.convert_results(raw_res, df.height)
        )

        case_algo_params = case_params["algo"]
        for processing_params in output_postprocessing.classical_post_processing_params_generator(
            params
        ):
            case_params["algo"] = case_algo_params.copy()
            case_params["algo"].update(processing_params)
            processed = output_postprocessing.classical_post_process(
                res=raw_res,
                processing_params=processing_params,
                params=case_params,
                length=df.height,
                df_dimred=df_dimred,
            )
            last = processed
            case_params["metrics"]["image_dir"] = _initialize_classical_test_sub_dir(
                case_params, misc_params["image_dir"]
            )
            try:
                results_writers.report_regime_statistics(
                    df, case_params, processed, unproc
                )
            except Exception:
                logger.exception("Failed classical regime statistics report")
            test_num += 1
            case_params["row"] = row
            case_params["test_num"] = test_num
            case_params["robustness"] = params.get("robustness") or {}
            case_params["stability"] = params.get("stability") or {}
            case_params["_pipeline_params"] = params
            try:
                from gulfstream.pipelines._shared import produce_all_metrics

                produce_all_metrics(df, processed, case_params)
            except Exception:
                logger.exception("Full metrics failed; falling back to classical plot")
                classical_produce_all_metrics(df, params, processed)
    return row, test_num, last


def classical_driver(
    df: pl.DataFrame, params: dict, misc_params: dict
) -> tuple[int, int, SegmentResults | None]:
    test_num = misc_params["test_num"]
    row = misc_params["row"]
    last: SegmentResults | None = None
    for res in dimension_reduction.dimred_generator(df, params):
        df_dimred = res.df
        algo_params = dimension_reduction.get_dimred_param_dict(res)
        for mapping_params, df_mapped in handle_classical_feature_map(df_dimred, params):
            algo_params_copy = algo_params.copy()
            algo_params_copy.update(mapping_params)
            try:
                row, test_num, last = classical_process_cases(
                    df,
                    df_dimred,
                    df_mapped,
                    params,
                    misc_params,
                    algo_params_copy,
                )
                misc_params["row"] = row
                misc_params["test_num"] = test_num
            except Exception:
                logger.exception("Failed classical processing case.")
    return test_num, row, last
