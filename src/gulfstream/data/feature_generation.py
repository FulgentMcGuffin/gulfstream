"""Feature generation from outright yield / FX dataframes."""
from __future__ import annotations

import itertools
import logging
import re

import numpy as np
import polars as pl

from gulfstream.common import frames

logger = logging.getLogger(__name__)


_TENOR_RE = re.compile(r"Y(\d+)p(\d+)")


def _tenor_years(col_suffix: str) -> float | None:
    m = _TENOR_RE.fullmatch(col_suffix)
    if not m:
        return None
    return int(m.group(1)) + int(m.group(2)) / 10.0


def _parse_rate_column(name: str) -> tuple[str, str] | None:
    """Parse ``USA_Y010p0`` → (``USA``, ``Y010p0``)."""
    parts = name.split("_", 1)
    if len(parts) != 2:
        return None
    source, tenor = parts
    if _tenor_years(tenor) is None:
        return None
    return source, tenor


def generate_yield_features(
    df: pl.DataFrame,
    *,
    vol_window: int = 60,
    corr_window: int = 60,
    include_levels: bool = True,
) -> pl.DataFrame:
    """Build interpretable features from a wide rates(+FX) DataFrame.

    Features include:
    - outright levels (optional)
    - intra-curve spreads and butterflies
    - cross-country spreads on matching tenors
    - rolling volatility and pairwise correlations of key series
    """
    df = frames.ensure_date_column(df)
    feat_cols = frames.feature_columns(df)

    rate_cols: dict[str, list[tuple[str, float]]] = {}
    for col in feat_cols:
        parsed = _parse_rate_column(col)
        if parsed is None:
            continue
        source, tenor = parsed
        years = _tenor_years(tenor)
        assert years is not None
        rate_cols.setdefault(source, []).append((col, years))

    for source in rate_cols:
        rate_cols[source].sort(key=lambda x: x[1])

    exprs: list[pl.Expr] = [pl.col(frames.DATE_COL)]

    if include_levels:
        for source, cols in rate_cols.items():
            for col, _ in cols:
                exprs.append(pl.col(col))

    for source, cols in rate_cols.items():
        names = [c for c, _ in cols]
        for a, b in itertools.combinations(names, 2):
            exprs.append((pl.col(a) - pl.col(b)).alias(f"{a}_minus_{b}"))
        if len(names) >= 3:
            for i in range(len(names) - 2):
                left, mid, right = names[i], names[i + 1], names[i + 2]
                exprs.append(
                    (2 * pl.col(mid) - pl.col(left) - pl.col(right)).alias(
                        f"fly_{left}_{mid}_{right}"
                    )
                )

    sources = list(rate_cols.keys())
    for s1, s2 in itertools.combinations(sources, 2):
        tenors1 = {c.split("_", 1)[1]: c for c, _ in rate_cols[s1]}
        tenors2 = {c.split("_", 1)[1]: c for c, _ in rate_cols[s2]}
        for tenor in set(tenors1) & set(tenors2):
            c1, c2 = tenors1[tenor], tenors2[tenor]
            exprs.append((pl.col(c1) - pl.col(c2)).alias(f"{s1}_{s2}_{tenor}_spd"))

    fx_cols = [c for c in feat_cols if _parse_rate_column(c) is None]
    for c in fx_cols:
        exprs.append(pl.col(c))

    key_series = []
    for source, cols in rate_cols.items():
        prefer = [c for c, y in cols if abs(y - 10.0) < 0.1]
        key_series.append(prefer[0] if prefer else cols[-1][0])
    key_series = key_series[:4]
    for col in key_series:
        if col in feat_cols:
            exprs.append(pl.col(col).rolling_std(vol_window).alias(f"{col}_vol"))
    for a, b in itertools.combinations(key_series, 2):
        if a in feat_cols and b in feat_cols:
            exprs.append(
                pl.rolling_corr(pl.col(a), pl.col(b), window_size=corr_window).alias(
                    f"{a}_{b}_corr"
                )
            )

    out = df.select(exprs)
    # Drop non-finite / null rows.
    feat = frames.feature_columns(out)
    out = out.filter(
        pl.all_horizontal([pl.col(c).is_not_null() & pl.col(c).is_finite() for c in feat])
    )
    logger.info("Generated %s features over %s rows", len(feat), out.height)
    return out


def generate_all_features(df_in: pl.DataFrame, **kwargs) -> pl.DataFrame:
    return generate_yield_features(df_in, **kwargs)
