"""Robustness / persistence of breakpoints under hyperparameter perturbations."""
from __future__ import annotations

import logging
import os

import polars as pl

from gulfstream.common import plotting, utils
from gulfstream.common.results import SegmentResults
from gulfstream.metrics import evaluation as evaluation_tools
from gulfstream.pipelines import single_pass as single_run

logger = logging.getLogger(__name__)


def _one_at_a_time_perturbations(base_params: dict, perturbations: dict) -> list[tuple[str, dict]]:
    """Yield (label, params) for one-at-a-time perturbations around the baseline."""
    import copy as _copy

    slim = {
        "algo": _copy.deepcopy(base_params.get("algo", {})),
        "test": _copy.deepcopy(base_params.get("test", {})),
        "metrics": {
            "mode": "write",
            "plot": False,
            "dir": (base_params.get("metrics") or {}).get("dir", "."),
        },
        "log": {"dir": (base_params.get("log") or {}).get("dir", "."), "level": "WARNING"},
    }

    out = []
    for key, values in (perturbations or {}).items():
        for val in values:
            p = _copy.deepcopy(slim)
            if key == "depth":
                p["algo"]["depth"] = [int(val)]
            elif key == "significance_level":
                p["test"]["significance_level"] = [float(val)]
            elif key == "min_regime_length":
                p["algo"]["min_regime_length"] = [int(val)]
            elif key == "num_features":
                p["algo"]["num_features"] = [int(val)]
            else:
                logger.warning("Unknown robustness perturbation key %s; skipping.", key)
                continue
            out.append((f"{key}={val}", p))
    return out


def annotate_persistence(
    baseline: SegmentResults,
    scores: dict[int, float],
    *,
    threshold: float,
) -> SegmentResults:
    low = [b for b, s in scores.items() if s < threshold]
    return SegmentResults(
        bkpts=list(baseline.bkpts),
        invalid_bkpts=list(baseline.invalid_bkpts),
        stats=dict(baseline.stats),
        hierarchy=dict(baseline.hierarchy),
        labels=baseline.labels,
        params=baseline.params,
        persistence=dict(scores),
        low_confidence_bkpts=low,
        stability_score=baseline.stability_score,
    )


def _plot_persistence(
    scores: dict[int, float],
    *,
    threshold: float,
    img_dir: str | None,
    mode: str,
) -> None:
    if not scores:
        return
    bkpts = sorted(scores.keys())
    vals = [scores[b] for b in bkpts]
    plot = plotting.ggplot_threshold_bars(
        [str(b) for b in bkpts],
        vals,
        threshold=threshold,
        title="Breakpoint persistence under HP perturbations",
        ylab="Persistence",
        xlab="Breakpoint index",
        above_color="#2ca02c",
        below_color="#d62728",
    )
    path = None
    if img_dir:
        os.makedirs(img_dir, exist_ok=True)
        path = os.path.join(
            img_dir, utils._img_gallery_filename("persistence_summary").lstrip("/")
        )
        pl.DataFrame({"bkpt": bkpts, "persistence": vals}).write_csv(
            os.path.join(img_dir, "persistence_report.csv")
        )
    plotting.emit_ggplot(
        plot, path=path, mode=mode, width=8, height=4, log_label="persistence summary"
    )


def evaluate_robustness(
    df: pl.DataFrame,
    params: dict,
    baseline: SegmentResults,
) -> SegmentResults:
    """Perturb hyperparameters, score bkpt persistence, annotate baseline results."""
    cfg = params.get("robustness") or {}
    if not cfg.get("enabled", False):
        return baseline

    tolerance = int(cfg.get("match_tolerance", 5))
    threshold = float(cfg.get("low_persistence_threshold", 0.5))
    perturbations = cfg.get("perturbations") or {
        "depth": [1, 2, 3],
        "significance_level": [0.01, 0.05, 0.1],
        "min_regime_length": [10, 20, 40],
    }

    pipeline_params = params.get("_pipeline_params") or params
    runs = _one_at_a_time_perturbations(pipeline_params, perturbations)
    if not runs:
        return baseline

    recoveries = {b: 0 for b in baseline.bkpts}
    n_ok = 0
    for label, p in runs:
        try:
            res = single_run.run_single_segmentation(df, p)
            n_ok += 1
            rate_map = evaluation_tools._match(baseline.bkpts, res.bkpts)
            for b, (matched, dist) in rate_map.items():
                if matched is not None and dist <= tolerance:
                    recoveries[b] = recoveries.get(b, 0) + 1
            logger.info(
                "Robustness run %s → bkpts=%s recovery=%.2f",
                label,
                res.bkpts,
                evaluation_tools.recovery_rate(baseline.bkpts, res.bkpts, tolerance=tolerance),
            )
        except Exception:
            logger.exception("Robustness perturbation %s failed", label)

    denom = max(n_ok, 1)
    scores = {b: recoveries.get(b, 0) / denom for b in baseline.bkpts}
    annotated = annotate_persistence(baseline, scores, threshold=threshold)

    metrics = params.get("metrics", {})
    _plot_persistence(
        scores,
        threshold=threshold,
        img_dir=metrics.get("image_dir") or metrics.get("dir"),
        mode=metrics.get("mode", "write"),
    )
    logger.info(
        "Persistence scores=%s low_confidence=%s",
        scores,
        annotated.low_confidence_bkpts,
    )
    return annotated
