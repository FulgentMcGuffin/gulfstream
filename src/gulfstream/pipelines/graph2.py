"""Graph 2 targeted retrain loop around a single Graph 1 segmentation."""
from __future__ import annotations

import copy
import datetime as dt
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from gulfstream.detection import stat_tests as bkpt_stat_tests
from gulfstream.detection import time_index as bkpt_timeindexing_conversions
from gulfstream.detection import trees as bkpt_trees
from gulfstream.metrics import evaluation as evaluation_tools
from gulfstream.features import names as feature_name_resolution
from gulfstream.common import frames
from gulfstream.common import logging as logging_config
from gulfstream.metrics import insights as post_information_visualization
from gulfstream.metrics import writers as results_writers
from gulfstream.common import utils
from gulfstream.common.results import SegmentResults
from gulfstream.pipelines.single_pass import run_single_segmentation_pair
from gulfstream.pipelines._shared import _produce_all_metrics

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


def _resolve_regimes_df(spec: Any, project_root: Path | None = None) -> pl.DataFrame | None:
    """Load seed regimes from null / path / CSV / records / DataFrame."""
    if spec is None:
        return None
    if isinstance(spec, pl.DataFrame):
        return spec
    if isinstance(spec, dict) and "path" in spec:
        path = Path(spec["path"])
        if not path.is_absolute() and project_root is not None:
            path = project_root / path
        return pl.read_csv(path)
    if isinstance(spec, (str, Path)):
        path = Path(spec)
        if not path.is_absolute() and project_root is not None:
            path = project_root / path
        return pl.read_csv(path)
    if isinstance(spec, list):
        return pl.DataFrame(spec)
    if isinstance(spec, dict):
        # Inline single-record dict without path — treat as empty unless columns present.
        if {"End", "Hierarchy Level of End"} <= set(spec.keys()):
            return pl.DataFrame([spec])
        return None
    raise TypeError(f"Unsupported regimes_df spec type: {type(spec)}")


def _resolve_retrain_features(df: pl.DataFrame, features_spec: Any) -> list[str]:
    if features_spec is None:
        raise TypeError("params['retrain']['features'] must be provided.")
    if isinstance(features_spec, list):
        if features_spec == ["__auto__"] or (
            len(features_spec) == 1 and features_spec[0] == "__auto__"
        ):
            return frames.feature_columns(df)
        names = feature_name_resolution.get_column_names(features_spec)
    elif isinstance(features_spec, dict):
        names = feature_name_resolution.get_column_names(features_spec)
    elif features_spec == "__auto__":
        return frames.feature_columns(df)
    else:
        raise TypeError("params['retrain']['features'] must be type dict or list[str].")
    feat_cols = set(frames.feature_columns(df))
    extra = [c for c in names if c not in feat_cols]
    if extra:
        logger.warning(
            "The following features are not present in df and will be ignored: %s.",
            ", ".join(extra),
        )
    return [c for c in names if c in feat_cols]


def _test_choice(params: dict) -> str:
    choice = params.get("test", {}).get("choice")
    if isinstance(choice, list):
        if not choice:
            raise ValueError("params['test']['choice'] is empty.")
        return str(choice[0])
    return str(choice)


def _heatmap_helper(
    df: pl.DataFrame,
    bkpts: list[int],
    iters: int,
    image_dir: str,
) -> tuple[np.ndarray, tuple[int, int]]:
    logger.info("Generating L2 error matrix.")
    loss_matrix = evaluation_tools._avg_features_loss(df, bkpts)
    name = f"retrain_iteration_{iters}_avg_feature_L2"
    post_information_visualization._draw_error_heatmaps(
        loss_matrix,
        df,
        bkpts,
        title=f"Retrain {iters}: Average daily L2 loss per feature in each regime (in bps)",
        cbar_label="average L2 dist to mean (measured in bps)",
        mode="write",
        img_dir=image_dir,
        gallery_filename=name,
    )
    logger.info("Saved error matrix to %s.", image_dir + utils._img_gallery_filename(name))
    return loss_matrix, tuple(np.unravel_index(np.argmax(loss_matrix), loss_matrix.shape))


def _get_user_requested_regime(bkpts: list[int], suggested: int) -> str:
    quits = ["q", "quit", "exit", "stop"]
    regime = input(
        f"Enter a regime number to break. Valid regime numbers are between 0 and "
        f"{len(bkpts)} inclusive. Suggested: regime {suggested}. "
        "Or enter 'q' to quit. "
    )
    not_digit_or_quit = not regime.isdigit() and regime.lower() not in quits
    digit_out_of_range = regime.isdigit() and (int(regime) < 0 or int(regime) > len(bkpts))
    while not_digit_or_quit or digit_out_of_range:
        if not_digit_or_quit:
            regime = input(
                "Invalid input. Enter a regime number to break. "
                f"Valid regime numbers are between 0 and {len(bkpts)} inclusive. "
                f"Suggested: regime {suggested}. Or enter 'q' to quit. "
            )
        else:
            regime = input(
                "Invalid regime number. Regime number must be between 0 and "
                f"{len(bkpts)} inclusive. Enter a regime number to break. "
                f"Suggested: regime {suggested}. Or enter 'q' to quit. "
            )
        not_digit_or_quit = not regime.isdigit() and regime.lower() not in quits
        digit_out_of_range = regime.isdigit() and (int(regime) < 0 or int(regime) > len(bkpts))
    return regime.lower()


def _get_user_requested_features(
    df: pl.DataFrame,
    loss_matrix: np.ndarray,
    regime: int,
    num_worst_features: int,
) -> list[str]:
    k = min(num_worst_features, loss_matrix.shape[0])
    worst_features_idx = np.argpartition(loss_matrix[:, regime], -k)[-k:]
    while True:
        feat_cols = frames.feature_columns(df)
        allowed_features = ",\n".join(feat_cols)
        print(f"Valid features to retrain on:\n{allowed_features}\n")
        highest_error_features = ",\n".join([feat_cols[i] for i in worst_features_idx])
        print(f"Highest error features in regime {regime} are:\n{highest_error_features}")
        features = input(
            "Enter a list of features, separated by a comma and space, to "
            "retrain on. Or press enter to use highest error features "
            "listed above. "
        )
        if not features.strip():
            return [feat_cols[i] for i in worst_features_idx]
        valid_format = all(word.strip() for word in features.split(", "))
        if valid_format:
            requested_columns = list({word.strip() for word in features.split(", ")})
            invalid_columns = [col for col in requested_columns if col not in feat_cols]
            if requested_columns and not invalid_columns:
                return requested_columns
            if not requested_columns:
                print("No valid columns provided.")
            if invalid_columns:
                print(
                    "The following requested features are not valid: "
                    + ", ".join(invalid_columns)
                )


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
) -> None:
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
        params_copy["template"] = results_writers._write_results_header(
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
        labels=bkpt_timeindexing_conversions._bkpts_to_labels(
            sorted(unprocessed_bkpts), df_full.height
        ),
        params=params_copy,
    )
    # Use merged processed_bkpts (represent incorrectly used the seed list).
    proc_res = SegmentResults(
        bkpts=sorted(processed_bkpts),
        invalid_bkpts=sorted(invalid_bkpts),
        stats=bkpt_stats,
        hierarchy=dict(bkpt_index_dict),
        labels=bkpt_timeindexing_conversions._bkpts_to_labels(
            sorted(processed_bkpts), df_full.height
        ),
        params=params_copy,
    )

    results_writers._report_regime_statistics(df_full, params_copy, proc_res, unproc_res)
    if params_copy["metrics"].get("developer_mode", False):
        results_writers._report_performance(df_full, params_copy, proc_res, unproc_res)

    if plot:
        logger.info("Producing all explainability and visualization tools.")
        # Ensure nested stages see robustness/stability toggles on params_copy.
        params_copy["robustness"] = params.get("robustness") or {}
        params_copy["stability"] = params.get("stability") or {}
        _produce_all_metrics(df_full, proc_res, params_copy)

    if params_copy.get("results_writer") is not None:
        params_copy["results_writer"].close()

    if relative_dir is not None:
        utils._generate_gallery(relative_dir)


def _targeted_retrain(df_full: pl.DataFrame, params: dict) -> None:
    with logging_config.LoggingContext(params["log"]["dir"], log_level=params["log"]["level"]):
        interactive = bool(params["retrain"]["interactive"])
        regimes_df = _resolve_regimes_df(params["retrain"].get("regimes_df"))
        bkpts, bkpt_index_dict = bkpt_timeindexing_conversions.regimes_df_to_bkpts(
            df_full, regimes_df
        )
        date_list = bkpt_timeindexing_conversions._get_strs_from_df_index(df_full)
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
            return

        features = _resolve_retrain_features(df_full, params["retrain"]["features"])
        if not features:
            logger.error("No valid features were specified. Quitting.")
            return
        df_filtered = frames.select_features(df_full, features)

        iters = 0
        loss_matrix, worst_index = _heatmap_helper(
            df_filtered, processed_bkpts, iters, image_dir
        )
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
                bkpt_index_dict = bkpt_trees._attach_retrain_tree(
                    bkpt_index_dict, local_hierarchy, start, end
                )
                bkpt_hierarchy = {
                    date_list[b]: int(lvl)
                    for b, lvl in sorted(bkpt_index_dict.items())
                }
                print("Current breakpoint hierarchy: ")
                print(json.dumps(bkpt_hierarchy, indent=4))

                iters += 1
                loss_matrix, worst_index = _heatmap_helper(
                    df_filtered, processed_bkpts, iters, image_dir
                )
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
                bkpt_index_dict = bkpt_trees._attach_retrain_tree(
                    bkpt_index_dict, local_hierarchy, start, end
                )
                bkpt_hierarchy = {
                    date_list[b]: int(lvl)
                    for b, lvl in sorted(bkpt_index_dict.items())
                }
                print("Current breakpoint hierarchy: ")
                print(json.dumps(bkpt_hierarchy, indent=4))

                iters += 1
                loss_matrix, worst_index = _heatmap_helper(
                    df_filtered, processed_bkpts, iters, image_dir
                )
                regime_raw = _get_user_requested_regime(
                    processed_bkpts, int(worst_index[1])
                )
                regime = int(regime_raw) if regime_raw.isdigit() else regime_raw

        _finalize_results(
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
        )
        logger.info("Targeted retrain done. Outputs in %s", relative_dir)


def targeted_retrain_with_user_specified_df(df: pl.DataFrame, params: dict) -> None:
    """Public Graph 2 entry: run targeted retrain on a user-supplied feature DF."""
    from gulfstream import validation as input_validation

    if not input_validation._valid_params_for_user_specified_df(params):
        logger.error("Invalid parameters; aborting.")
        return
    if "retrain" not in params:
        logger.error("'retrain' section is required for Graph 2.")
        return
    _targeted_retrain(df, params)
