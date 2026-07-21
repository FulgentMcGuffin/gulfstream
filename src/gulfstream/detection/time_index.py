"""Breakpoint index / date conversions and regime interval helpers."""
from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from gulfstream.common import frames
from gulfstream.common.results import AlgoResults, SegmentResults


def _get_strs_from_df_index(df: pl.DataFrame) -> list[str]:
    """Return string labels for each row date."""
    return frames.date_strs(df)


def _get_dates_from_df_index(df_slice: pl.DataFrame) -> list:
    """Return plottable x-coordinates from a dataframe date column."""
    if df_slice.height == 0:
        return []
    return frames.dates_series(df_slice).to_list()


def _get_date_labels_for_plots(
    df: pl.DataFrame,
    bkpts: list[int],
    newlines: bool = False,
) -> list[str]:
    """Human-readable regime date labels for legends."""
    dates = _get_strs_from_df_index(df)
    n = len(dates)
    edges = [0] + sorted(b for b in bkpts if 0 < b < n) + [n]
    sep = "\n" if newlines else " → "
    labels = []
    for i in range(len(edges) - 1):
        start = dates[edges[i]] if edges[i] < n else dates[-1]
        end_idx = edges[i + 1] - 1
        end = dates[end_idx] if 0 <= end_idx < n else dates[-1]
        labels.append(f"{start}{sep}{end}")
    return labels


def _get_regime_intervals(bkpt_hierarchy: dict, index) -> pl.DataFrame:
    """Build Start/End/Regime dataframe from breakpoint hierarchy keys."""
    n = len(index)
    bkpts = sorted(int(b) for b in bkpt_hierarchy.keys() if 0 < int(b) < n)
    edges = [0] + bkpts + [n]
    rows = []
    for i in range(len(edges) - 1):
        start_i, end_i = edges[i], edges[i + 1]
        if end_i <= start_i:
            continue
        rows.append(
            {
                "Start": index[start_i],
                "End": index[min(end_i - 1, n - 1)],
                "Regime": i,
            }
        )
    return pl.DataFrame(rows) if rows else pl.DataFrame(
        schema={"Start": pl.Object, "End": pl.Object, "Regime": pl.Int64}
    )


def get_regime_intervals_legacy(labels: list[int], index) -> pl.DataFrame:
    """Build intervals from a contiguous label sequence."""
    if not labels:
        return pl.DataFrame(schema={"Start": pl.Object, "End": pl.Object, "Regime": pl.Int64})
    rows = []
    start = 0
    current = labels[0]
    for i in range(1, len(labels)):
        if labels[i] != current:
            rows.append({"Start": index[start], "End": index[i - 1], "Regime": current})
            start = i
            current = labels[i]
    rows.append({"Start": index[start], "End": index[len(labels) - 1], "Regime": current})
    return pl.DataFrame(rows)


def _convert_results(
    res: SegmentResults | AlgoResults,
    length: int,
) -> SegmentResults | AlgoResults:
    """Identity conversion when dimred length equals original length.

    For DMD-style index remapping, callers should set algo params; core path
    (PCA/raw) uses 1:1 indexing so this is a shallow copy with clamped bkpts.
    """
    def _clamp(bkpts: list[int]) -> list[int]:
        return [b for b in bkpts if 0 < b < length - 1]

    if isinstance(res, SegmentResults):
        new_bkpts = _clamp(list(res.bkpts))
        new_invalid = _clamp(list(res.invalid_bkpts))
        new_stats = {b: res.stats[b] for b in new_bkpts if b in res.stats}
        new_hier = {b: res.hierarchy[b] for b in new_bkpts if b in res.hierarchy}
        labels = res.labels
        if labels is not None and len(labels) != length:
            labels = _bkpts_to_labels(new_bkpts, length)
        elif labels is None:
            labels = _bkpts_to_labels(new_bkpts, length)
        return SegmentResults(
            bkpts=new_bkpts,
            invalid_bkpts=new_invalid,
            stats=new_stats,
            hierarchy=new_hier,
            labels=labels,
            params=res.params,
        )
    new_bkpts = _clamp(list(res.bkpts))
    return AlgoResults(
        bkpts=new_bkpts,
        labels=_bkpts_to_labels(new_bkpts, length),
        params=res.params,
    )


def _shift_algo_results(res: SegmentResults, start: int) -> SegmentResults:
    """Add start offset to all breakpoint indices (targeted retrain)."""
    def shift_list(xs: list[int]) -> list[int]:
        return [x + start for x in xs]

    return SegmentResults(
        bkpts=shift_list(res.bkpts),
        invalid_bkpts=shift_list(res.invalid_bkpts),
        stats={k + start: v for k, v in res.stats.items()},
        hierarchy={k + start: v for k, v in res.hierarchy.items()},
        labels=res.labels,
        params=res.params,
    )


def convert_dimred_index_to_original_index(bkpt: int, params: dict) -> int:
    """Map a dimred-index breakpoint to original series index (identity for PCA)."""
    algo = params.get("algo", {})
    if algo.get("dimred") == "dmd":
        stride = int(algo.get("dmd_stride", 1) or 1)
        window = int(algo.get("dmd_rolling_window", 1) or 1)
        return int(bkpt * stride + window // 2)
    return int(bkpt)


def convert_original_index_to_dimred_index(idx: int, params: dict) -> int:
    """Map an original-series index to dimred units (identity for PCA)."""
    algo = params.get("algo", {})
    if algo.get("dimred") == "dmd":
        stride = int(algo.get("dmd_stride", 1) or 1)
        window = int(algo.get("dmd_rolling_window", 1) or 1)
        return max(0, int((idx - window // 2) // stride))
    return int(idx)


def _bkpts_to_labels(bkpts: list[int], length: int) -> list[int]:
    labels = np.zeros(length, dtype=int)
    edges = [0] + sorted(b for b in bkpts if 0 < b < length) + [length]
    for i in range(len(edges) - 1):
        labels[edges[i] : edges[i + 1]] = i
    return labels.tolist()


def regimes_df_to_bkpts(
    df: pl.DataFrame,
    regimes_df: pl.DataFrame | None,
) -> tuple[list[int], dict[int, int]]:
    """Convert a seed regimes_df into breakpoint indices and hierarchy dict.

    Empty / None regimes_df → no breakpoints.
    Uses ``End`` of all but the last row as breakpoint dates (represent semantics).
    """
    empty = (
        regimes_df is None
        or not isinstance(regimes_df, pl.DataFrame)
        or regimes_df.height == 0
        or regimes_df.height < 2
    )
    if empty:
        return [], {}

    required = {"End", "Hierarchy Level of End"}
    missing = required - set(regimes_df.columns)
    if missing:
        raise KeyError(f"regimes_df missing columns: {sorted(missing)}")

    bkpt_index_dict: dict[int, int] = {}
    # All but the last row.
    for row in regimes_df.slice(0, regimes_df.height - 1).iter_rows(named=True):
        end_val = row["End"]
        try:
            loc = frames.row_index_for_date(df, end_val)
        except KeyError:
            continue
        if 0 < loc < df.height:
            bkpt_index_dict[loc] = int(row["Hierarchy Level of End"])
    bkpts = sorted(bkpt_index_dict.keys())
    return bkpts, bkpt_index_dict
