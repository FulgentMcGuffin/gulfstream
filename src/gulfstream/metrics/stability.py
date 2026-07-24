"""Stability of breakpoints across sources, tenors, and time windows."""
from __future__ import annotations

import logging
import os
import re

import polars as pl

from gulfstream.common import frames as frame_helpers
from gulfstream.common import plotting, utils
from gulfstream.common.results import SegmentResults
from gulfstream.metrics import evaluation as evaluation_tools
from gulfstream.pipelines import single_pass as single_run

logger = logging.getLogger(__name__)

_RATE_COL = re.compile(r"^([A-Z]{3})_(Y\d+p\d+)$")


def _parse_groups(columns) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    by_source: dict[str, list[str]] = {}
    by_tenor: dict[str, list[str]] = {}
    for c in columns:
        m = _RATE_COL.match(str(c))
        if not m:
            continue
        source, tenor = m.group(1), m.group(2)
        by_source.setdefault(source, []).append(c)
        by_tenor.setdefault(tenor, []).append(c)
    return by_source, by_tenor


def _ablation_frames(df: pl.DataFrame, cfg: dict) -> list[tuple[str, pl.DataFrame]]:
    frames: list[tuple[str, pl.DataFrame]] = []
    feat_cols = frame_helpers.feature_columns(df)
    by_source, by_tenor = _parse_groups(feat_cols)

    if cfg.get("leave_one_source_out", True) and len(by_source) > 1:
        for source, cols in by_source.items():
            keep = [c for c in feat_cols if c not in cols]
            if len(keep) < 2:
                continue
            frames.append((f"drop_source_{source}", frame_helpers.select_features(df, keep)))

    if cfg.get("leave_one_tenor_out", True) and len(by_tenor) > 1:
        for tenor, cols in by_tenor.items():
            keep = [c for c in feat_cols if c not in cols]
            if len(keep) < 2:
                continue
            frames.append((f"drop_tenor_{tenor}", frame_helpers.select_features(df, keep)))

    tw = cfg.get("time_windows") or {}
    mode = tw.get("mode", "expanding")
    min_frac = float(tw.get("min_frac", 0.5))
    n = df.height
    if mode == "expanding" and n > 50:
        for frac in (min_frac, 0.75, 1.0):
            end = max(int(n * frac), 30)
            if end >= n and frac < 1.0:
                continue
            frames.append((f"window_frac_{frac:.2f}", frame_helpers.slice_rows(df, 0, end)))
    return frames


def _plot_stability(
    rows: list[dict],
    *,
    img_dir: str | None,
    mode: str,
    floor: float,
) -> None:
    if not rows:
        return
    report = pl.DataFrame(rows)
    plot = plotting.ggplot_threshold_bars(
        report["ablation"].to_list(),
        report["recovery"].to_list(),
        threshold=floor,
        title="Stability across ablations",
        ylab="Bkpt recovery rate",
        xlab="Ablation",
        above_color="#1f77b4",
        below_color="#d62728",
    )
    path = None
    if img_dir:
        os.makedirs(img_dir, exist_ok=True)
        path = os.path.join(
            img_dir, utils.img_gallery_filename("stability_summary").lstrip("/")
        )
        report.write_csv(os.path.join(img_dir, "stability_report.csv"))
    plotting.emit_ggplot(
        plot, path=path, mode=mode, width=max(6, len(rows) * 0.5), height=4, log_label="stability summary"
    )


def evaluate_stability(
    df: pl.DataFrame,
    params: dict,
    baseline: SegmentResults,
) -> SegmentResults:
    """Ablate sources/tenors/time windows and score baseline breakpoint stability."""
    cfg = params.get("stability") or {}
    if not cfg.get("enabled", False):
        return baseline

    tolerance = int(cfg.get("match_tolerance", 5))
    floor = float(cfg.get("stability_floor", 0.5))
    ablations = _ablation_frames(df, cfg)
    if not ablations:
        # Synthetic frames without SOURCE_TENOR columns: use time windows only.
        n = df.height
        min_frac = float((cfg.get("time_windows") or {}).get("min_frac", 0.5))
        ablations = [
            (f"window_frac_{frac:.2f}", frame_helpers.slice_rows(df, 0, max(int(n * frac), 30)))
            for frac in (min_frac, 0.75)
            if int(n * frac) < n
        ]

    rows = []
    recoveries = []
    for label, sub in ablations:
        if frame_helpers.n_features(sub) < 1 or sub.height < 40:
            continue
        try:
            pipeline_params = params.get("_pipeline_params") or params
            # Slim copy without ExcelWriter.
            run_params = {
                "algo": pipeline_params.get("algo", {}),
                "test": pipeline_params.get("test", {}),
                "metrics": {"mode": "write", "plot": False, "dir": "."},
                "log": {"dir": (pipeline_params.get("log") or {}).get("dir", "."), "level": "WARNING"},
            }
            res = single_run.run_single_segmentation(sub, run_params)
            # Only compare bkpts that fall inside the sub-series length.
            base_in = [b for b in baseline.bkpts if b < sub.height]
            rate = evaluation_tools.recovery_rate(
                base_in, res.bkpts, tolerance=tolerance
            )
            haus = evaluation_tools._directed_hausdorff_wrapper(
                base_in, res.bkpts, sub.height
            )
            rows.append(
                {
                    "ablation": label,
                    "recovery": rate,
                    "hausdorff": haus,
                    "n_bkpts": len(res.bkpts),
                    "n_rows": sub.height,
                    "n_cols": frame_helpers.n_features(sub),
                }
            )
            recoveries.append(rate)
            logger.info("Stability %s recovery=%.2f hausdorff=%.1f", label, rate, haus)
        except Exception:
            logger.exception("Stability ablation %s failed", label)
            rows.append(
                {
                    "ablation": label,
                    "recovery": 0.0,
                    "hausdorff": float("nan"),
                    "n_bkpts": 0,
                    "n_rows": sub.height,
                    "n_cols": frame_helpers.n_features(sub),
                }
            )
            recoveries.append(0.0)

    score = float(sum(recoveries) / len(recoveries)) if recoveries else 1.0
    metrics = params.get("metrics", {})
    _plot_stability(
        rows,
        img_dir=metrics.get("image_dir") or metrics.get("dir"),
        mode=metrics.get("mode", "write"),
        floor=floor,
    )

    annotated = SegmentResults(
        bkpts=list(baseline.bkpts),
        invalid_bkpts=list(baseline.invalid_bkpts),
        stats=dict(baseline.stats),
        hierarchy=dict(baseline.hierarchy),
        labels=baseline.labels,
        params=baseline.params,
        persistence=dict(baseline.persistence or {}),
        low_confidence_bkpts=list(baseline.low_confidence_bkpts or []),
        stability_score=score,
        bkpt_ci=dict(baseline.bkpt_ci or {}),
        panel_support=dict(baseline.panel_support or {}),
    )
    if score < floor:
        logger.warning(
            "Segmentation stability_score=%.2f below floor=%.2f", score, floor
        )
    else:
        logger.info("Segmentation stability_score=%.2f", score)
    return annotated
