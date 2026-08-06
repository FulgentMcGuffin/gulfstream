"""Polars DataFrame conventions and helpers for gulfstream.

All feature matrices are ``pl.DataFrame`` with a required ``date`` column
(Datetime or Utf8/Date). Remaining columns are numeric features.

Third-party libraries that need pandas/numpy (sklearn, ruptures, plotnine,
openpyxl, pytorch-forecasting) receive conversions only at the call boundary
via ``to_numpy`` / ``to_pandas``.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Sequence

import numpy as np
import polars as pl

DATE_COL = "date"


def ensure_date_column(df: pl.DataFrame, *, name: str = DATE_COL) -> pl.DataFrame:
    """Ensure ``name`` exists; if missing, synthesize a business-day range."""
    if name in df.columns:
        col = df[name]
        if col.dtype in (pl.Datetime, pl.Date):
            if col.dtype == pl.Date:
                return df.with_columns(pl.col(name).cast(pl.Datetime(time_unit="us")))
            return df
        # String / other → parse as datetime.
        return df.with_columns(
            pl.col(name).cast(pl.Utf8).str.to_datetime(strict=False).alias(name)
        )
    n = df.height
    start = datetime(2000, 1, 1)
    dates = [start + timedelta(days=i) for i in range(n)]
    return df.with_columns(pl.Series(name, dates)).select([name, *df.columns])


def from_numpy(
    values: np.ndarray,
    *,
    dates: Sequence[Any] | pl.Series | None = None,
    columns: Sequence[str] | None = None,
    date_col: str = DATE_COL,
) -> pl.DataFrame:
    """Build a dated feature frame from a 2d numpy array."""
    arr = np.asarray(values)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    n, k = arr.shape
    if columns is None:
        columns = [f"c{i}" for i in range(k)]
    if dates is None:
        start = datetime(2000, 1, 1)
        dates = [start + timedelta(days=i) for i in range(n)]
    elif isinstance(dates, pl.Series):
        dates = dates.to_list()
    data = {date_col: list(dates)}
    for i, c in enumerate(columns):
        data[str(c)] = arr[:, i]
    return pl.DataFrame(data)


def feature_columns(df: pl.DataFrame, *, date_col: str = DATE_COL) -> list[str]:
    return [c for c in df.columns if c != date_col]


def n_features(df: pl.DataFrame, *, date_col: str = DATE_COL) -> int:
    return len(feature_columns(df, date_col=date_col))


def to_numpy(df: pl.DataFrame, *, date_col: str = DATE_COL) -> np.ndarray:
    cols = feature_columns(df, date_col=date_col)
    if not cols:
        return np.empty((df.height, 0), dtype=float)
    return df.select(cols).to_numpy().astype(float, copy=False)


def dates_series(df: pl.DataFrame, *, date_col: str = DATE_COL) -> pl.Series:
    if date_col not in df.columns:
        raise KeyError(f"Missing required '{date_col}' column")
    return df[date_col]


def date_strs(df: pl.DataFrame, *, date_col: str = DATE_COL) -> list[str]:
    s = dates_series(df, date_col=date_col)
    out: list[str] = []
    for v in s.to_list():
        if hasattr(v, "strftime"):
            out.append(v.strftime("%Y-%m-%d"))
        else:
            out.append(str(v)[:10])
    return out


def slice_rows(df: pl.DataFrame, start: int, end: int | None = None) -> pl.DataFrame:
    """Half-open row slice ``[start, end)``."""
    if end is None:
        end = df.height
    start = max(0, int(start))
    end = min(df.height, int(end))
    if end <= start:
        return df.clear()
    return df.slice(start, end - start)


def select_features(
    df: pl.DataFrame,
    columns: Sequence[str],
    *,
    date_col: str = DATE_COL,
) -> pl.DataFrame:
    """Keep date + requested feature columns (order preserved)."""
    cols = [c for c in columns if c != date_col and c in df.columns]
    return df.select([date_col, *cols]) if date_col in df.columns else df.select(cols)


def with_same_dates(
    values: np.ndarray,
    template: pl.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    date_col: str = DATE_COL,
) -> pl.DataFrame:
    """Attach ``template``'s dates to a new feature matrix."""
    return from_numpy(
        values,
        dates=dates_series(template, date_col=date_col),
        columns=columns,
        date_col=date_col,
    )


def row_index_for_date(
    df: pl.DataFrame,
    value: Any,
    *,
    date_col: str = DATE_COL,
) -> int:
    """Locate row index for a date-like value (first match)."""
    s = dates_series(df, date_col=date_col)
    # Normalize search value.
    if isinstance(value, str):
        target = value[:10]
        for i, v in enumerate(s.to_list()):
            vs = v.strftime("%Y-%m-%d") if hasattr(v, "strftime") else str(v)[:10]
            if vs == target:
                return i
    else:
        for i, v in enumerate(s.to_list()):
            if v == value:
                return i
            if hasattr(v, "date") and hasattr(value, "date") and v.date() == value.date():
                return i
    raise KeyError(f"Date {value!r} not found in '{date_col}'")


def to_pandas(df: pl.DataFrame, *, date_col: str = DATE_COL):
    """Boundary helper: polars → pandas with DatetimeIndex named like classical code."""
    import pandas as pd

    pdf = df.to_pandas()
    if date_col in pdf.columns:
        pdf[date_col] = pd.to_datetime(pdf[date_col])
        pdf = pdf.set_index(date_col)
        pdf.index.name = None
    return pdf


def from_pandas(pdf, *, date_col: str = DATE_COL) -> pl.DataFrame:
    """Boundary helper: pandas (index or column) → dated polars frame."""
    import pandas as pd

    if not isinstance(pdf, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(pdf)}")
    if date_col in pdf.columns:
        out = pl.from_pandas(pdf)
        return ensure_date_column(out, name=date_col)
    # Index holds dates.
    reset = pdf.reset_index()
    first = reset.columns[0]
    reset = reset.rename(columns={first: date_col})
    return ensure_date_column(pl.from_pandas(reset), name=date_col)


def clone(df: pl.DataFrame) -> pl.DataFrame:
    return df.clone()


def height(df: pl.DataFrame) -> int:
    return df.height


def shape_features(df: pl.DataFrame, *, date_col: str = DATE_COL) -> tuple[int, int]:
    return df.height, n_features(df, date_col=date_col)
