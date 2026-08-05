"""Graph 2 targeted retrain loop around a single Graph 1 segmentation."""
from __future__ import annotations

import copy
import json
import logging

import numpy as np
import polars as pl

from gulfstream.common import frames
from gulfstream.common import logging as logging_config
from gulfstream.common.results import SegmentResults
from gulfstream.detection import stat_tests as bkpt_stat_tests
from gulfstream.detection import time_index as bkpt_timeindexing_conversions
from gulfstream.detection import trees as bkpt_trees
from gulfstream.pipelines.graph2.interactive import (
    _get_user_requested_features,
    _get_user_requested_regime,
)
from gulfstream.pipelines.graph2.reporting import (
    _finalize_results,
    _heatmap_helper,
    _retrain_prep_dir,
)
from gulfstream.pipelines.graph2.seeding import (
    _resolve_regimes_df,
    _resolve_retrain_features,
    regimes_df_to_bkpts,
)
from gulfstream.pipelines.single_pass import run_single_segmentation_pair

logger = logging.getLogger(__name__)


def _test_choice(params: dict) -> str:
    choice = params.get("test", {}).get("choice")
    if isinstance(choice, list):
        if not choice:
            raise ValueError("params['test']['choice'] is empty.")
        return str(choice[0])
    return str(choice)


def _include_last_regime_handler(params: dict, training_last_regime: bool) -> None:
    """Force include_last_regime=[True] when the slice is not the series tail."""
    if training_last_regime:
        return
    algo = params.setdefault("algo", {})
    if algo.get("include_last_regime") is not None:
        algo["include_last_regime"] = [True]


def _shift_bkpts_stats(res: SegmentResults, start: int) -> SegmentResults:
    """Shift bkpts/stats/invalid to global indices; leave hierarchy empty for attach."""
    return SegmentResults(
        bkpts=[x + start for x in res.bkpts],
        invalid_bkpts=[x + start for x in res.invalid_bkpts],
        stats={k + start: v for k, v in res.stats.items()},
        hierarchy={},
        labels=res.labels,
        params=res.params,
    )


def _targeted_retrain_once(
    df_full: pl.DataFrame,
    params: dict,
    bkpts: list[int],
    regime: int,
    features: list[str],
) -> tuple[SegmentResults, SegmentResults, int, int, dict[int, int]]:
    """Detect on one regime slice; return shifted bkpts + local hierarchy for attach."""
    cols = [col for col in frames.feature_columns(df_full) if col in features]
    aug_bkpts = [0] + list(bkpts) + [df_full.height]
    start = int(aug_bkpts[regime])
    end = int(aug_bkpts[regime + 1])
    df_slice = frames.select_features(frames.slice_rows(df_full, start, end), cols)

    run_params = copy.deepcopy(params)
    # Drop write-side keys that should not nest into slice runs.
    metrics = run_params.get("metrics") or {}
    run_params["metrics"] = {
        k: v
        for k, v in metrics.items()
        if k not in {"results_writer", "template", "image_dir"}
    }
    # Slice runs should not recurse into robustness/stability.
    if "robustness" in run_params:
        run_params["robustness"] = {**(run_params["robustness"] or {}), "enabled": False}
    if "stability" in run_params:
        run_params["stability"] = {**(run_params["stability"] or {}), "enabled": False}

    _include_last_regime_handler(run_params, end == df_full.height)

    unproc_local, proc_local = run_single_segmentation_pair(df_slice, run_params)
    local_hierarchy = {int(k): int(v) for k, v in (proc_local.hierarchy or {}).items()}

    unproc_shifted = _shift_bkpts_stats(unproc_local, start)
    proc_shifted = _shift_bkpts_stats(proc_local, start)
    return unproc_shifted, proc_shifted, start, end, local_hierarchy


def _targeted_retrain(df_full: pl.DataFrame, params: dict) -> SegmentResults | None:
    with logging_config.LoggingContext(params["log"]["dir"], log_level=params["log"]["level"]):
        interactive = bool(params["retrain"]["interactive"])
        regimes_df = _resolve_regimes_df(params["retrain"].get("regimes_df"))
        bkpts, bkpt_index_dict = regimes_df_to_bkpts(df_full, regimes_df)
        date_list = bkpt_timeindexing_conversions.get_strs_from_df_index(df_full)
        bkpt_hierarchy = {
            date_list[b]: int(lvl) for b, lvl in sorted(bkpt_index_dict.items())
        }
        print("Current breakpoint hierarchy: ")
        print(json.dumps(bkpt_hierarchy, indent=4))

        unprocessed_bkpts = list(bkpts)
        invalid_bkpts: list[int] = []
        processed_bkpts = list(bkpts)

        test = _test_choice(params)
        if test not in bkpt_stat_tests.DEFAULT_STATS:
            raise ValueError(f"Unknown statistical test {test}.")
        default_stat = bkpt_stat_tests.DEFAULT_STATS[test]
        unproc_bkpt_stats = {b: default_stat for b in unprocessed_bkpts}
        bkpt_stats = {b: default_stat for b in processed_bkpts}

        try:
            relative_dir, image_dir = _retrain_prep_dir(params)
        except Exception:
            logger.exception("Failed to initialize directory for saving images.")
            return None

        features = _resolve_retrain_features(df_full, params["retrain"]["features"])
        if not features:
            logger.error("No valid features were specified. Quitting.")
            return None
        df_filtered = frames.select_features(df_full, features)

        score_method = str(
            (params.get("retrain") or {}).get("score_method", "mse_to_mean")
        )
        score_kwargs = dict((params.get("retrain") or {}).get("score") or {})

        iters = 0
        collected_plots: dict[str, object] = {}
        loss_matrix, worst_index, heatmap = _heatmap_helper(
            df_filtered,
            processed_bkpts,
            iters,
            image_dir,
            score_method=score_method,
            score_kwargs=score_kwargs,
        )
        collected_plots[f"retrain_iteration_{iters}"] = heatmap
        worst_loss = float(loss_matrix[worst_index])
        num_worst_features = min(
            int(params["retrain"].get("num_worst_features", 5)), len(features)
        )

        if not interactive:
            threshold = float(params["retrain"]["threshold"])
            max_iter = int(params["retrain"]["max_iter"])
            while worst_loss > threshold and iters < max_iter:
                regime = int(worst_index[1])
                k = min(num_worst_features, loss_matrix.shape[0])
                worst_features_idx = np.argpartition(loss_matrix[:, regime], -k)[-k:]
                target_features = [frames.feature_columns(df_filtered)[i] for i in worst_features_idx]

                print(
                    f"Retraining in regime {regime} with features "
                    f"{', '.join(target_features)}."
                )
                (
                    unproc_res,
                    proc_res,
                    start,
                    end,
                    local_hierarchy,
                ) = _targeted_retrain_once(
                    df_filtered, params, processed_bkpts, regime, target_features
                )
                print(f"Done retraining in regime {regime}.")

                if len(proc_res.bkpts) == 0:
                    print(
                        "Failed to find a breakpoint in the highest error regime. Quitting."
                    )
                    break

                unprocessed_bkpts.extend(unproc_res.bkpts)
                invalid_bkpts.extend(unproc_res.invalid_bkpts)
                processed_bkpts.extend(proc_res.bkpts)
                processed_bkpts = sorted(set(processed_bkpts))
                unproc_bkpt_stats.update(unproc_res.stats)
                bkpt_stats.update(proc_res.stats)
                # Attach unshifted local hierarchy (avoid double-offset).
                bkpt_index_dict = bkpt_trees.attach_retrain_tree(
                    bkpt_index_dict, local_hierarchy, start, end
                )
                bkpt_hierarchy = {
                    date_list[b]: int(lvl)
                    for b, lvl in sorted(bkpt_index_dict.items())
                }
                print("Current breakpoint hierarchy: ")
                print(json.dumps(bkpt_hierarchy, indent=4))

                iters += 1
                loss_matrix, worst_index, heatmap = _heatmap_helper(
                    df_filtered,
                    processed_bkpts,
                    iters,
                    image_dir,
                    score_method=score_method,
                    score_kwargs=score_kwargs,
                )
                collected_plots[f"retrain_iteration_{iters}"] = heatmap
                worst_loss = float(loss_matrix[worst_index])
        else:
            quits = ["q", "quit", "exit", "stop"]
            regime_raw = _get_user_requested_regime(processed_bkpts, int(worst_index[1]))
            regime: int | str = int(regime_raw) if regime_raw.isdigit() else regime_raw
            while regime not in quits:
                assert isinstance(regime, int)
                target_features = _get_user_requested_features(
                    df_filtered, loss_matrix, regime, num_worst_features
                )
                print(
                    f"Retraining in regime {regime} with features "
                    f"{', '.join(target_features)}."
                )
                (
                    unproc_res,
                    proc_res,
                    start,
                    end,
                    local_hierarchy,
                ) = _targeted_retrain_once(
                    df_filtered, params, processed_bkpts, regime, target_features
                )
                print(f"Done retraining in regime {regime}.")

                if len(proc_res.bkpts) == 0:
                    print(f"Failed to find a breakpoint in the requested regime {regime}.")

                unprocessed_bkpts.extend(unproc_res.bkpts)
                invalid_bkpts.extend(unproc_res.invalid_bkpts)
                processed_bkpts.extend(proc_res.bkpts)
                processed_bkpts = sorted(set(processed_bkpts))
                unproc_bkpt_stats.update(unproc_res.stats)
                bkpt_stats.update(proc_res.stats)
                bkpt_index_dict = bkpt_trees.attach_retrain_tree(
                    bkpt_index_dict, local_hierarchy, start, end
                )
                bkpt_hierarchy = {
                    date_list[b]: int(lvl)
                    for b, lvl in sorted(bkpt_index_dict.items())
                }
                print("Current breakpoint hierarchy: ")
                print(json.dumps(bkpt_hierarchy, indent=4))

                iters += 1
                loss_matrix, worst_index, heatmap = _heatmap_helper(
                    df_filtered,
                    processed_bkpts,
                    iters,
                    image_dir,
                    score_method=score_method,
                    score_kwargs=score_kwargs,
                )
                collected_plots[f"retrain_iteration_{iters}"] = heatmap
                regime_raw = _get_user_requested_regime(
                    processed_bkpts, int(worst_index[1])
                )
                regime = int(regime_raw) if regime_raw.isdigit() else regime_raw

        proc_res = _finalize_results(
            df_full,
            params,
            unprocessed_bkpts=unprocessed_bkpts,
            processed_bkpts=processed_bkpts,
            invalid_bkpts=invalid_bkpts,
            unproc_bkpt_stats=unproc_bkpt_stats,
            bkpt_stats=bkpt_stats,
            bkpt_index_dict=bkpt_index_dict,
            relative_dir=relative_dir,
            image_dir=image_dir,
            collected_plots=collected_plots,
        )
        logger.info("Targeted retrain done. Outputs in %s", relative_dir)
        return proc_res


def run_graph2(df: pl.DataFrame, params: dict) -> SegmentResults | None:
    """Public Graph 2 entry: run targeted retrain on a user-supplied feature DF."""
    from gulfstream.common.config import coerce_params

    try:
        params = coerce_params(params)
    except Exception:
        logger.exception("Invalid parameters; aborting.")
        return None
    if "retrain" not in params or params.get("retrain") is None:
        logger.error("'retrain' section is required for Graph 2.")
        return None
    return _targeted_retrain(df, params)
