"""Orchestrator for legacy (non-interval) regime detection methods.

Bayesian Gaussian mixture, HMM, k-means, HDBSCAN, OPTICS, MSAR,
ruptures (non-binseg), and Wasserstein clustering.

Only the user-specified DataFrame path is supported (no S3 market-data entry).
"""
from __future__ import annotations

import logging
import os
from typing import Iterator

import polars as pl

from gulfstream.detection import time_index as bkpt_timeindexing_conversions
from gulfstream.dimred import dispatcher as dimension_reduction
from gulfstream.metrics import explainability as explainability_tools
from gulfstream.metrics import regime_plots
from gulfstream.features import names as feature_name_resolution
from gulfstream.common import frames
from gulfstream.features import kernel_map as kernel_feature_mapping
from gulfstream.common import logging as logging_config
from gulfstream.detection import postprocess as output_postprocessing
from gulfstream.common import utils
from gulfstream.common.results import AlgoResults
from gulfstream.validation import legacy
from gulfstream.pipelines._shared import (
    _initialize_results_writer_and_dir,
    _log_missing_columns,
)
from gulfstream.legacy.detectors import (
    bayesian_gmm,
    hdbscan,
    hmm,
    kmeans,
    msar,
    optics,
    ruptures_methods,
    wasserstein,
)

logger = logging.getLogger(__name__)

LEGACY_METHODS = {
    "bayesian_gmm": bayesian_gmm.bayesian_gmm_predict_regimes,
    "hmm": hmm.hmm_predict_regimes,
    "kmeans": kmeans.kmeans_predict_regimes,
    "hdbscan": hdbscan.hdbscan_predict_regimes,
    "optics": optics.optics_predict_regimes,
    "msar": msar.msar_predict_regimes,
    "ruptures": ruptures_methods.ruptures_predict_regimes,
    "wasserstein": wasserstein.wasserstein_clustering_predict_regimes,
}

LEGACY_PARAM_GENERATORS = {
    "bayesian_gmm": bayesian_gmm.bayesian_gmm_param_generator,
    "hmm": hmm.hmm_param_generator,
    "kmeans": kmeans.kmeans_param_generator,
    "hdbscan": hdbscan.hdbscan_param_generator,
    "optics": optics.optics_param_generator,
    "msar": msar.msar_params_generator,
    "ruptures": ruptures_methods.ruptures_param_generator,
    "wasserstein": wasserstein.wass_clustering_param_generator,
}

LEGACY_PARAM_FORMATTERS = {
    "bayesian_gmm": bayesian_gmm.bayesian_gmm_params_printout,
    "hmm": hmm.hmm_params_printout,
    "kmeans": kmeans.kmeans_params_printout,
    "hdbscan": hdbscan.hdbscan_params_printout,
    "optics": optics.optics_params_printout,
    "msar": msar.msar_params_printout,
    "ruptures": ruptures_methods.ruptures_params_printout,
    "wasserstein": wasserstein.wass_params_printout,
}


def legacy_evaluate_regimes_with_user_specified_df(df: pl.DataFrame, params: dict) -> None:
    """Run legacy regime detection on ``df`` over the parameter grid in ``params``."""
    if not legacy._valid_legacy_params_with_user_specified_df(params):
        return
    with logging_config.LoggingContext(params["log"]["dir"], log_level=params["log"]["level"]):
        try:
            image_dir, results_writer, template = (
                _initialize_results_writer_and_dir(params)
            )
        except Exception:
            logger.exception("Failed to initialize results writer and directory.")
            return

        if isinstance(params["metrics"].get("explainability_features"), dict):
            exp_features = feature_name_resolution.get_column_names(
                params["metrics"].get("explainability_features")
            )
        else:
            exp_features = params["metrics"].get("explainability_features")
        _log_missing_columns(df, exp_features)

        misc_params = {
            "test_num": 0,
            "row": 0,
            "results_writer": results_writer,
            "image_dir": image_dir,
            "template": template,
            "metrics": params["metrics"],
            "data_params": {},
        }
        legacy_driver(df, params, misc_params)
        if results_writer is not None:
            results_writer.close()
        if image_dir is not None:
            utils._generate_gallery(image_dir)


def legacy_driver(df: pl.DataFrame, params: dict, misc_params: dict) -> tuple[int, int]:
    test_num = misc_params["test_num"]
    row = misc_params["row"]
    for res in dimension_reduction._dimred_generator(df, params):
        df_dimred = res.df
        algo_params = dimension_reduction._get_dimred_param_dict(res)
        for mapping_params, df_mapped in handle_feature_map(df_dimred, params):
            algo_params_copy = algo_params.copy()
            algo_params_copy.update(mapping_params)
            try:
                row, test_num = legacy_process_cases(
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
                logger.exception("Failed processing case.")
    return test_num, row


def legacy_process_cases(
    df: pl.DataFrame,
    df_dimred: pl.DataFrame,
    df_mapped: pl.DataFrame,
    params: dict,
    misc_params: dict,
    algo_params: dict,
) -> tuple[int, int]:
    row = misc_params["row"]
    test_num = misc_params["test_num"]
    for method_params in _legacy_method_params_generator(params):
        method = method_params["regime_detection_algorithm"]
        if method not in LEGACY_METHODS:
            raise ValueError(f"Unknown regime detection algorithm {method}.")
        case_params: dict = {"algo": algo_params.copy()}
        case_params["algo"].update(method_params)
        case_params.update(misc_params)
        case_params["row"] = row
        case_params["test_num"] = test_num

        logger.info("Running %s regime detection algorithm.", method)
        raw_res = run_legacy_algo(method, df_mapped, case_params)
        res = bkpt_timeindexing_conversions._convert_results(raw_res, df.height)

        case_algo_params = case_params["algo"]
        for processing_params in output_postprocessing._legacy_post_processing_params_generator(
            params
        ):
            case_params["algo"] = case_algo_params.copy()
            case_params["algo"].update(processing_params)
            processed_results = output_postprocessing._legacy_post_process(
                res=raw_res,
                processing_params=processing_params,
                params=case_params,
                length=df.height,
                df_dimred=df_dimred,
            )
            case_params["metrics"]["image_dir"] = _initialize_test_sub_dir(
                case_params, misc_params["image_dir"]
            )
            row = _legacy_report_performance(df, case_params, processed_results, res)
            test_num += 1
            case_params["row"] = row
            case_params["test_num"] = test_num
            legacy_produce_all_metrics(df, params, processed_results)
    return row, test_num


def _legacy_method_params_generator(params: dict) -> Iterator[dict]:
    for gen in LEGACY_PARAM_GENERATORS.values():
        yield from gen(params)


def _initialize_test_sub_dir(params: dict, file_dir: str | None) -> str | None:
    save = params["metrics"].get("mode", "write") in ["write", "display_and_write"]
    if save and file_dir:
        sub_dir = os.path.join(
            file_dir,
            f"{params['test_num']}_{params['algo']['regime_detection_algorithm']}",
        )
        os.makedirs(sub_dir, exist_ok=True)
        return sub_dir
    return None


def run_legacy_algo(method: str, df: pl.DataFrame, params: dict) -> AlgoResults:
    handler = LEGACY_METHODS.get(method)
    if not handler:
        raise ValueError(f"Unknown regime detection algorithm {method}.")
    res = handler(df, **params["algo"])
    res.params = params
    return res


def handle_feature_map(df: pl.DataFrame, params: dict):
    for res in kernel_feature_mapping._legacy_feature_map_generator(df, params):
        mapped_df = res.dfs[0]
        mapping_params = {
            "feature_map_kernel_params": res.kernel_params,
            "feature_map_approx_method": res.feature_map_approx_method,
            "num_features": res.num_features,
        }
        if res.kernel_approx_error:
            mapping_params["kernel_approx_error"] = res.kernel_approx_error
        yield mapping_params, mapped_df


def _legacy_report_performance(
    df: pl.DataFrame,
    params: dict,
    proc_res: AlgoResults,
    unproc_res: AlgoResults,
) -> int:
    logger.info(
        "Legacy results: processed bkpts=%s unprocessed bkpts=%s algo=%s",
        len(proc_res.bkpts),
        len(unproc_res.bkpts),
        params["algo"].get("regime_detection_algorithm"),
    )
    row = int(params.get("row", 0))
    writer = params.get("results_writer")
    if writer is None:
        return row + 1
    template = params.get("template") or {}
    new_row = dict(template)
    new_row.update(
        {
            "test_num": params.get("test_num"),
            "algo": params["algo"].get("regime_detection_algorithm"),
            "dimred": params["algo"].get("dimred"),
            "n_bkpts_proc": len(proc_res.bkpts),
            "n_bkpts_raw": len(unproc_res.bkpts),
        }
    )
    try:
        sheet = "Results"
        existing = None
        if sheet in writer.book.sheetnames:
            import pandas as pd
            existing = pd.read_excel(writer, sheet_name=sheet)
        import pandas as pd
        out = (
            pd.concat([existing, pd.DataFrame([new_row])], ignore_index=True)
            if existing is not None
            else pd.DataFrame([new_row])
        )
        out.to_excel(writer, sheet_name=sheet, index=False)
    except Exception:
        logger.exception("Failed to write legacy performance row")
    return row + 1


def _regime_plot(
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
    regimes_df = bkpt_timeindexing_conversions.get_regime_intervals_legacy(
        labels, date_index
    )
    regime_plots._visualize_market_regimes(
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


def legacy_produce_all_metrics(df: pl.DataFrame, params: dict, res: AlgoResults) -> None:
    """Thin metrics wrapper: regime plot only (labels need not be intervals)."""
    if not params["metrics"].get("plot"):
        return
    df_explainable = explainability_tools._get_explainability_df(
        df, params["metrics"].get("explainability_features")
    )
    _regime_plot(
        df=df_explainable,
        valid_bkpts=res.bkpts,
        labels=res.labels or [],
        mode=params["metrics"]["mode"],
        img_dir=params["metrics"].get("image_dir"),
        variables=params["metrics"].get("features_to_plot"),
        invalid_bkpts=[],
        bkpt_hierarchy={},
    )
