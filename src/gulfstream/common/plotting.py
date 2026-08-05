"""Plotnine helpers for gulfstream gallery plots.

Prefer plotnine (ggplot) for statistical graphics. Callers that need
sklearn ``plot_tree`` or networkx drawings still use matplotlib.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Sequence

import numpy as np
import pandas as pd
from plotnine import (
    aes,
    element_text,
    geom_hline,
    geom_point,
    geom_col,
    geom_text,
    geom_tile,
    ggplot,
    labs,
    scale_fill_cmap,
    scale_fill_manual,
    theme,
    theme_bw,
    theme_minimal,
)

logger = logging.getLogger(__name__)

# plotnine rejects saves when either dimension exceeds 25 inches (default limitsize).
_PLOTNINE_MAX_INCHES = 25.0


def cap_figure_inches(width: float, height: float) -> tuple[float, float]:
    """Clamp ggplot save dimensions to plotnine's default 25-inch limit."""
    return (
        min(float(width), _PLOTNINE_MAX_INCHES),
        min(float(height), _PLOTNINE_MAX_INCHES),
    )


def _cap_figure_inches(width: float, height: float) -> tuple[float, float]:
    return cap_figure_inches(width, height)


def matrix_to_long(
    matrix: np.ndarray,
    *,
    x_labels: Sequence[str] | None = None,
    y_labels: Sequence[str] | None = None,
    x_name: str = "x",
    y_name: str = "y",
    value_name: str = "value",
) -> pd.DataFrame:
    """Flatten a 2-d array into a long DataFrame for ``geom_tile``."""
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2:
        raise ValueError("matrix must be 2-d")
    n_y, n_x = arr.shape
    x_labels = list(x_labels) if x_labels is not None else [str(i) for i in range(n_x)]
    y_labels = list(y_labels) if y_labels is not None else [str(i) for i in range(n_y)]
    rows: list[dict[str, Any]] = []
    for i in range(n_y):
        for j in range(n_x):
            rows.append(
                {
                    x_name: x_labels[j],
                    y_name: y_labels[i],
                    value_name: float(arr[i, j]),
                    "_xi": j,
                    "_yi": i,
                }
            )
    return pd.DataFrame(rows)


def emit_ggplot(
    plot: ggplot,
    *,
    path: str | None,
    mode: str = "write",
    width: float = 8,
    height: float = 5,
    dpi: int = 120,
    log_label: str = "plot",
) -> None:
    """Save and/or display a plotnine ggplot according to metrics ``mode``."""
    width, height = cap_figure_inches(width, height)
    plot = plot + theme(figure_size=(width, height))
    if mode in ("write", "display_and_write") and path:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        plot.save(
            path,
            width=width,
            height=height,
            dpi=dpi,
            verbose=False,
            limitsize=False,
        )
        logger.info("Saved %s to %s", log_label, path)
    if mode in ("display", "display_and_write"):
        try:
            print(plot)
        except Exception:
            pass


def ggplot_heatmap(
    matrix: np.ndarray,
    *,
    title: str,
    x_labels: Sequence[str] | None = None,
    y_labels: Sequence[str] | None = None,
    fill_label: str = "value",
    cmap: str = "viridis",
    annotate: bool = False,
) -> ggplot:
    """Build a tile heatmap (replaces seaborn ``heatmap``)."""
    long = matrix_to_long(matrix, x_labels=x_labels, y_labels=y_labels)
    long["x"] = pd.Categorical(
        long["x"], categories=list(dict.fromkeys(long["x"])), ordered=True
    )
    y_order = list(dict.fromkeys(long["y"]))
    long["y"] = pd.Categorical(long["y"], categories=list(reversed(y_order)), ordered=True)
    if annotate and matrix.size and max(matrix.shape) <= 12:
        long["label"] = long["value"].map(lambda v: f"{v:.2f}")
    p = (
        ggplot(long, aes(x="x", y="y", fill="value"))
        + geom_tile(color="white", size=0.1)
        + scale_fill_cmap(cmap_name=cmap, name=fill_label)
        + labs(title=title, x="", y="")
        + theme_minimal()
        + theme(
            axis_text_x=element_text(rotation=45, ha="right"),
            figure_size=_cap_figure_inches(
                max(4, matrix.shape[1] * 0.55 + 2),
                max(3, matrix.shape[0] * 0.35 + 1.5),
            ),
        )
    )
    if "label" in long.columns:
        p = p + geom_text(aes(label="label"), size=7, color="black")
    return p


def ggplot_clusters_2d(
    xy: np.ndarray,
    labels: np.ndarray,
    *,
    title: str,
) -> ggplot:
    """Scatter of regime means colored by cluster label."""
    df = pd.DataFrame(
        {
            "x": xy[:, 0],
            "y": xy[:, 1],
            "cluster": [str(int(c)) for c in labels],
            "regime": [f"R{i}" for i in range(len(labels))],
        }
    )
    return (
        ggplot(df, aes(x="x", y="y", color="cluster"))
        + geom_point(size=4)
        + geom_text(aes(label="regime"), nudge_y=0.02, size=8, show_legend=False)
        + labs(title=title, x="dim 1", y="dim 2", color="cluster")
        + theme_bw()
        + theme(figure_size=(6, 5))
    )


def ggplot_threshold_bars(
    categories: Sequence[str],
    values: Sequence[float],
    *,
    threshold: float,
    title: str,
    ylab: str,
    xlab: str = "",
    below_color: str = "#d62728",
    above_color: str = "#2ca02c",
) -> ggplot:
    """Bar chart with a horizontal threshold line (persistence / stability)."""
    df = pd.DataFrame(
        {
            "category": list(categories),
            "value": list(values),
            "flag": ["below" if v < threshold else "above" for v in values],
        }
    )
    df["category"] = pd.Categorical(df["category"], categories=list(categories), ordered=True)
    return (
        ggplot(df, aes(x="category", y="value", fill="flag"))
        + geom_col()
        + geom_hline(yintercept=threshold, linetype="dashed", color="black")
        + scale_fill_manual(
            values={"below": below_color, "above": above_color},
            breaks=["below", "above"],
            labels=[f"< {threshold}", f"≥ {threshold}"],
            name="",
        )
        + labs(title=title, x=xlab, y=ylab)
        + theme_bw()
        + theme(
            axis_text_x=element_text(rotation=45, ha="right"),
            figure_size=(max(6, len(categories) * 0.45), 4),
        )
    )
