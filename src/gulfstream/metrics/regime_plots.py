"""Regime visualization tools for the Graph 1 core path (plotnine)."""
from __future__ import annotations

import logging
import os
from typing import Literal

import numpy as np
import pandas as pd
import polars as pl
from plotnine import (
    aes,
    element_text,
    facet_wrap,
    geom_line,
    geom_rect,
    geom_text,
    geom_vline,
    ggplot,
    labs,
    theme,
    theme_bw,
)

from gulfstream.common import frames, plotting, utils
from gulfstream.common.results import SegmentResults
from gulfstream.detection import time_index as bkpt_timeindexing_conversions

logger = logging.getLogger(__name__)


def plot_market_regimes(
    df: pl.DataFrame,
    regimes_df: pl.DataFrame,
    title: str = "",
    variables: list[str] | None = None,
    n_columns: int = 2,
    invalid_bkpts: list[int] | None = None,
    valid_bkpts: list[int] | None = None,
    low_confidence_bkpts: list[int] | None = None,
    bkpt_hierarchy: dict | None = None,
    bkpt_ranks: dict | None = None,
    bkpt_ci: dict[int, tuple[int, int]] | None = None,
    plot_ci_ribbons: bool = True,
    mode: Literal["display", "write", "display_and_write"] = "display",
    img_dir: str | None = None,
    name: str = "regime_plot",
) -> None:
    invalid_bkpts = invalid_bkpts or []
    valid_bkpts = valid_bkpts or []
    low_confidence_bkpts = low_confidence_bkpts or []
    bkpt_hierarchy = bkpt_hierarchy or {}
    bkpt_ci = bkpt_ci or {}

    feat_cols = frames.feature_columns(df)
    if variables:
        variables = [c for c in variables if c in feat_cols]
    if not variables:
        variables = feat_cols[:n_columns]

    dates = frames.dates_series(df).to_list()
    # Numeric x for reliable spans/vlines across datetime/string dates.
    x = np.arange(len(dates), dtype=float)

    series_rows: list[dict] = []
    for var in variables:
        vals = frames.to_numpy(frames.select_features(df, [var])).ravel()
        for xi, yi in zip(x, vals):
            series_rows.append({"x": float(xi), "y": float(yi), "variable": var})
    series = pd.DataFrame(series_rows)

    plot = ggplot(series, aes(x="x", y="y")) + theme_bw()

    if regimes_df.height > 0 and {"Start", "End", "Regime"}.issubset(set(regimes_df.columns)):
        # Map Start/End date-like values to indices when possible.
        date_to_idx = {d: i for i, d in enumerate(dates)}
        rect_rows = []
        for row in regimes_df.iter_rows(named=True):
            start, end = row["Start"], row["End"]
            if start in date_to_idx and end in date_to_idx:
                x0, x1 = float(date_to_idx[start]), float(date_to_idx[end])
            else:
                # Already numeric indices.
                x0, x1 = float(start), float(end)
            rect_rows.append(
                {
                    "xmin": x0,
                    "xmax": max(x0 + 1e-6, x1),
                    "ymin": -np.inf,
                    "ymax": np.inf,
                    "Regime": str(int(row["Regime"])),
                }
            )
        if rect_rows:
            rects = pd.DataFrame(rect_rows)
            plot = plot + geom_rect(
                rects,
                aes(xmin="xmin", xmax="xmax", ymin="ymin", ymax="ymax", fill="Regime"),
                alpha=0.25,
                inherit_aes=False,
            )

    # Uncertainty CI ribbons (index bands around confirmed breakpoints).
    if plot_ci_ribbons and bkpt_ci:
        ci_rows = []
        n = len(dates)
        for b, band in bkpt_ci.items():
            try:
                lo, hi = int(band[0]), int(band[1])
            except (TypeError, ValueError, IndexError):
                continue
            lo = max(0, min(lo, n - 1))
            hi = max(0, min(hi, n - 1))
            if hi < lo:
                lo, hi = hi, lo
            ci_rows.append(
                {
                    "xmin": float(lo),
                    "xmax": float(max(lo + 1e-6, hi)),
                    "ymin": -np.inf,
                    "ymax": np.inf,
                    "bkpt": int(b),
                }
            )
        if ci_rows:
            ci_df = pd.DataFrame(ci_rows)
            plot = plot + geom_rect(
                ci_df,
                aes(xmin="xmin", xmax="xmax", ymin="ymin", ymax="ymax"),
                fill="#4C78A8",
                alpha=0.22,
                inherit_aes=False,
                show_legend=False,
            )

    plot = plot + geom_line(color="black", size=0.4)

    def _vline_df(indices: list[int], color: str) -> pd.DataFrame | None:
        xs = [float(i) for i in indices if 0 <= i < len(dates)]
        if not xs:
            return None
        return pd.DataFrame({"xintercept": xs, "color": color})

    for color, idxs in (
        ("magenta", invalid_bkpts),
        ("orange", low_confidence_bkpts),
        ("red", valid_bkpts),
    ):
        vdf = _vline_df(idxs, color)
        if vdf is not None:
            plot = plot + geom_vline(
                vdf,
                aes(xintercept="xintercept"),
                color=color,
                size=0.6,
                inherit_aes=False,
            )

    if valid_bkpts and bkpt_hierarchy:
        label_rows = []
        y_min = float(series["y"].min()) if len(series) else 0.0
        for b in valid_bkpts:
            if 0 <= b < len(dates) and b in bkpt_hierarchy:
                label_rows.append(
                    {"x": float(b), "y": y_min, "label": str(bkpt_hierarchy[b])}
                )
        if label_rows:
            labels = pd.DataFrame(label_rows)
            plot = plot + geom_text(
                labels,
                aes(x="x", y="y", label="label"),
                size=7,
                va="bottom",
                inherit_aes=False,
            )

    plot = (
        plot
        + facet_wrap("~variable", ncol=1, scales="free_y")
        + labs(title=title or "Regime plot", x="time index", y="")
        + theme(
            figure_size=(14, max(3, 2.8 * len(variables))),
            axis_text_x=element_text(rotation=0),
            legend_position="right",
        )
    )

    path = None
    if mode in ("write", "display_and_write"):
        if not img_dir:
            raise ValueError("img_dir required when writing plots")
        os.makedirs(img_dir, exist_ok=True)
        path = os.path.join(img_dir, utils.img_gallery_filename(name).lstrip("/"))
    plotting.emit_ggplot(
        plot,
        path=path,
        mode=mode,
        width=14,
        height=max(3, 2.8 * len(variables)),
        log_label="regime plot",
    )


def produce_all_regime_visualization_tools(
    df: pl.DataFrame,
    params: dict,
    res: SegmentResults,
) -> None:
    """Core-path regime plot."""
    hierarchy = res.hierarchy or {b: 1 for b in res.bkpts}
    date_index = frames.dates_series(df).to_list()
    regimes_df = bkpt_timeindexing_conversions.get_regime_intervals(hierarchy, date_index)
    features = params.get("metrics", {}).get("features_to_plot") or []
    if not features:
        feat_cols = frames.feature_columns(df)
        features = feat_cols[: min(3, len(feat_cols))]
    mode = params.get("metrics", {}).get("mode", "display")
    img_dir = params.get("metrics", {}).get("image_dir") or params.get("metrics", {}).get("dir")
    plot_ci = bool(params.get("metrics", {}).get("plot_ci_ribbons", True))
    plot_market_regimes(
        df,
        regimes_df,
        title=f"test_{params.get('test_num', 0)}_{params.get('test', {}).get('choice', '')}",
        variables=features,
        invalid_bkpts=res.invalid_bkpts,
        valid_bkpts=res.bkpts,
        low_confidence_bkpts=list(res.low_confidence_bkpts or []),
        bkpt_hierarchy={b: hierarchy.get(b, 1) for b in res.bkpts},
        bkpt_ci=dict(res.bkpt_ci or {}),
        plot_ci_ribbons=plot_ci,
        mode=mode,
        img_dir=img_dir,
    )
