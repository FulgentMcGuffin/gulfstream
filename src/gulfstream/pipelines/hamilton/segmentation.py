"""Apache Hamilton nodes for one Graph 1 / Graph 2 segmentation pass.

See https://hamilton.apache.org/ — each function is a DAG node; parameter
names declare dependencies. Imperative grid loops stay outside Hamilton;
this module expresses the linear core: prepare → dimred → map → ruptures
→ postprocess.
"""
from __future__ import annotations

import copy
from typing import Any

import polars as pl

from gulfstream.common.results import DimredResults, SegmentResults
from gulfstream.detection import algorithm as bkpt_algo
from gulfstream.detection import hyperparams as custom_hyperparameter_selection
from gulfstream.detection import postprocess as output_postprocessing
from gulfstream.detection import stat_tests as bkpt_stat_tests
from gulfstream.detection import time_index as bkpt_timeindexing_conversions
from gulfstream.dimred import dispatcher as dimension_reduction
from gulfstream.features import kernel_map as kernel_feature_mapping


def prepared_params(params: dict) -> dict:
    """Deep-copy params and normalize recursive_method / metrics for one pass."""
    out = copy.deepcopy(params)
    metrics = out.get("metrics") or {}
    out["metrics"] = {
        k: v
        for k, v in metrics.items()
        if k not in {"results_writer", "template", "image_dir"}
    }
    algo = out.setdefault("algo", {})
    if "recursive_method" not in algo:
        algo["recursive_method"] = ["full"]
    elif isinstance(algo["recursive_method"], str):
        algo["recursive_method"] = [algo["recursive_method"]]
    else:
        algo["recursive_method"] = [algo["recursive_method"][0]]
    return out


def lag_pca_df(features_df: pl.DataFrame, prepared_params: dict) -> pl.DataFrame | None:
    """Optional PCA frame used when ACF lag selection is requested."""
    if custom_hyperparameter_selection.asked_for_acf_lag_selection(prepared_params):
        return custom_hyperparameter_selection.calculate_pca_for_lag_selection(
            features_df
        )
    return None


def dimred_result(features_df: pl.DataFrame, prepared_params: dict) -> DimredResults:
    """First dimred embedding from the configured generator grid."""
    return next(dimension_reduction.dimred_generator(features_df, prepared_params))


def df_dimred(dimred_result: DimredResults) -> pl.DataFrame:
    return dimred_result.df


def dimred_algo_params(dimred_result: DimredResults) -> dict:
    algo_params = dimension_reduction.get_dimred_param_dict(dimred_result)
    algo_params["recursive_method"] = "full"
    return algo_params


def date_strings(features_df: pl.DataFrame) -> list[str]:
    return bkpt_timeindexing_conversions.get_strs_from_df_index(features_df)


def feature_map_bundle(
    df_dimred: pl.DataFrame, prepared_params: dict
) -> tuple[dict, list, list]:
    """Return ``(mapping_params, mapped_dfs_list, diff_dfs_list)`` for ``full``."""
    for res in kernel_feature_mapping.feature_map_generator(df_dimred, prepared_params):
        mapping_params = {
            "feature_map_kernel_params": res.kernel_params,
            "feature_map_approx_method": res.feature_map_approx_method,
            "num_features": res.num_features,
            "num_mappings": len(res.dfs),
        }
        return mapping_params, [res.dfs], [res.df_diffs]
    raise RuntimeError("feature_map_generator produced no mappings")


def case_params(
    prepared_params: dict,
    dimred_algo_params: dict,
    feature_map_bundle: tuple[dict, list, list],
    df_dimred: pl.DataFrame,
) -> dict:
    mapping_params, _mapped_lists, _diff_lists = feature_map_bundle
    ruptures_params = next(bkpt_algo.kernel_ruptures_generator([df_dimred], prepared_params))
    test_params = next(bkpt_stat_tests.test_param_combos(prepared_params))
    late = next(bkpt_algo.late_algo_param_combos(prepared_params))
    out = {
        "test": test_params.copy(),
        "algo": {**dimred_algo_params, **late, **mapping_params},
        "metrics": prepared_params.get("metrics", {}),
    }
    out["algo"]["ruptures_kernel_params"] = ruptures_params[0]
    return out


def raw_segmentation(
    df_dimred: pl.DataFrame,
    feature_map_bundle: tuple[dict, list, list],
    case_params: dict,
    date_strings: list[str],
    lag_pca_df: pl.DataFrame | None,
) -> SegmentResults:
    _mapping_params, mapped_lists, diff_lists = feature_map_bundle
    mapped_dfs = mapped_lists[0]
    mapped_df_diffs = diff_lists[0]
    return bkpt_algo.find_and_test_bkpts(
        df_dimred,
        mapped_dfs,
        case_params,
        date_strings,
        mapped_df_diffs=mapped_df_diffs,
        df_pca=lag_pca_df,
    )


def combined_raw(
    raw_segmentation: SegmentResults, df_dimred: pl.DataFrame
) -> SegmentResults:
    return output_postprocessing.combine_results(df_dimred.height, [raw_segmentation])


def unprocessed_result(
    combined_raw: SegmentResults, features_df: pl.DataFrame
) -> SegmentResults:
    return bkpt_timeindexing_conversions.convert_results(combined_raw, features_df.height)


def processed_result(
    combined_raw: SegmentResults,
    case_params: dict,
    prepared_params: dict,
    features_df: pl.DataFrame,
    df_dimred: pl.DataFrame,
    feature_map_bundle: tuple[dict, list, list],
    lag_pca_df: pl.DataFrame | None,
) -> SegmentResults:
    _mapping_params, mapped_lists, diff_lists = feature_map_bundle
    mapped_dfs = mapped_lists[0]
    mapped_df_diffs = diff_lists[0]
    processing_params = next(
        output_postprocessing.post_processing_params_generator(
            case_params["test"]["choice"], prepared_params
        )
    )
    case_params = copy.deepcopy(case_params)
    case_params["algo"].update(processing_params)
    return output_postprocessing.post_process(
        res=combined_raw,
        processing_params=processing_params,
        params=case_params,
        length=features_df.height,
        df_dimred=df_dimred,
        mapped_dfs=mapped_dfs,
        mapped_df_diffs=mapped_df_diffs,
        df_pca=lag_pca_df,
    )


def segmentation_pair(
    unprocessed_result: SegmentResults, processed_result: SegmentResults
) -> tuple[SegmentResults, SegmentResults]:
    """Final node: ``(unprocessed, processed)`` for callers."""
    return unprocessed_result, processed_result
