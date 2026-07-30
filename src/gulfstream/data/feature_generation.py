"""Feature generation from outright yield / FX dataframes."""
from __future__ import annotations

import itertools
import logging
import re
import tempfile
from collections.abc import Sequence
from pathlib import Path

import polars as pl
import yaml

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


def identity_features(df: pl.DataFrame, **_kwargs) -> pl.DataFrame:
    """No-op feature generator — returns the dated raw frame unchanged."""
    return frames.ensure_date_column(df)


def _wide_curve_frame_to_long(df: pl.DataFrame) -> pl.DataFrame:
    """Convert gulfstream wide ``SOURCE_TENOR`` columns to panelyzer long rates.

    Output columns: ``date``, ``source``, plus tenor columns such as ``Y010p0``.
    Non-rate columns (e.g. FX) are dropped.
    """
    df = frames.ensure_date_column(df)
    by_source: dict[str, dict[str, str]] = {}
    for col in frames.feature_columns(df):
        parsed = _parse_rate_column(col)
        if parsed is None:
            continue
        source, tenor = parsed
        by_source.setdefault(source, {})[tenor] = col
    if not by_source:
        raise ValueError(
            "generate_panelyzer_features expected wide rate columns like "
            "'USA_Y010p0'; found none"
        )

    pieces: list[pl.DataFrame] = []
    for source, tenor_map in sorted(by_source.items()):
        rename = {wide: tenor for tenor, wide in tenor_map.items()}
        # rename maps wide→tenor; build select
        selected = df.select(
            [
                pl.col(frames.DATE_COL),
                *[pl.col(wide).alias(tenor) for tenor, wide in sorted(tenor_map.items())],
            ]
        ).with_columns(pl.lit(source).alias("source"))
        pieces.append(selected)

    # Outer-align tenor columns across sources (null-fill missing tenors).
    all_tenors = sorted({t for m in by_source.values() for t in m})
    aligned: list[pl.DataFrame] = []
    for piece in pieces:
        for tenor in all_tenors:
            if tenor not in piece.columns:
                piece = piece.with_columns(pl.lit(None).cast(pl.Float64).alias(tenor))
        aligned.append(piece.select([frames.DATE_COL, "source", *all_tenors]))
    return pl.concat(aligned, how="vertical").sort([frames.DATE_COL, "source"])


def generate_panelyzer_features(
    df: pl.DataFrame,
    *,
    config: str | Path,
    project_root: str | Path | None = None,
    use_input_frame: bool = True,
    drop_nulls: bool = True,
) -> pl.DataFrame:
    """Build features by delegating to panelyzer's ``feature_builder.create_features``.

    Panelyzer framecache is **always disabled** here — YAML need not (and should
    not) specify ``cache`` / backend settings; any ``cache`` block in the YAML is
    overwritten with ``enabled: false``.

    Parameters
    ----------
    df
        Dated wide feature frame from the gulfstream source loader (e.g. rates).
    config
        Path to a panelyzer feature-builder YAML (expressions / optional sources).
        Relative paths resolve against ``project_root`` or the process cwd.
    use_input_frame
        If true (default), convert ``df`` to a long rates panel and inject it as
        the primary parquet source (YAML ``sources`` are replaced). If false,
        call panelyzer with the YAML sources as written.
    drop_nulls
        Drop rows with any null/non-finite feature after evaluation.
    """
    from feature_builder import create_features

    root = Path(project_root) if project_root is not None else Path.cwd()
    cfg_path = Path(config)
    if not cfg_path.is_absolute():
        cfg_path = (root / cfg_path).resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"panelyzer feature config not found: {cfg_path}")

    df = frames.ensure_date_column(df)
    dates = frames.dates_series(df)
    dmin = dates.min()
    dmax = dates.max()
    if dmin is None or dmax is None:
        raise ValueError("Input frame has no dates for panelyzer evaluation")

    def _as_date_str(value: object) -> str:
        if hasattr(value, "date") and callable(getattr(value, "date", None)):
            try:
                return value.date().isoformat()  # datetime
            except Exception:
                pass
        if hasattr(value, "isoformat"):
            return value.isoformat()
        return str(value)[:10]

    from_s = _as_date_str(dmin)
    to_s = _as_date_str(dmax)

    raw_cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw_cfg, dict):
        raise ValueError(f"panelyzer config must be a mapping: {cfg_path}")
    raw_cfg = dict(raw_cfg)
    # Always disable panelyzer caching (no YAML cache/backend required).
    raw_cfg["cache"] = {"enabled": False, "db_path": ":memory:"}

    if use_input_frame:
        long_df = _wide_curve_frame_to_long(df)
        tenors = [c for c in long_df.columns if c not in {frames.DATE_COL, "source"}]
        default_feature = "Y010p0" if "Y010p0" in tenors else tenors[0]

        with tempfile.TemporaryDirectory(prefix="gulfstream_panelyzer_") as tmp:
            parquet_path = Path(tmp) / "rates_long.parquet"
            long_df.write_parquet(parquet_path)

            raw_cfg["date_range"] = {"from": from_s, "to": to_s}
            raw_cfg.setdefault("output", {})
            if isinstance(raw_cfg["output"], dict):
                raw_cfg["output"] = {
                    **raw_cfg["output"],
                    "mode": raw_cfg["output"].get("mode", "expressions_only"),
                }
            raw_cfg.setdefault("panel", {})
            if isinstance(raw_cfg["panel"], dict):
                raw_cfg["panel"] = {
                    **raw_cfg["panel"],
                    "date_column_name": frames.DATE_COL,
                    "min_date": from_s,
                    "max_date": to_s,
                }
            raw_cfg["sources"] = [
                {
                    "type": "parquet",
                    "path": str(parquet_path),
                    "entity_col": "source",
                    "default_feature": default_feature,
                    "date_col": frames.DATE_COL,
                    "date_col_type": "datetime",
                    "primary": True,
                }
            ]
            out = create_features(raw_cfg, config_dir=cfg_path.parent)
    else:
        raw_cfg.setdefault("date_range", {"from": from_s, "to": to_s})
        out = create_features(raw_cfg, config_dir=cfg_path.parent)

    # Normalize date column name for gulfstream.
    if frames.DATE_COL not in out.columns:
        for cand in ("datetime", "Date", "DATE"):
            if cand in out.columns:
                out = out.rename({cand: frames.DATE_COL})
                break
    out = frames.ensure_date_column(out)
    if drop_nulls:
        feat = frames.feature_columns(out)
        if feat:
            out = out.filter(
                pl.all_horizontal(
                    [pl.col(c).is_not_null() & pl.col(c).is_finite() for c in feat]
                )
            )
    logger.info(
        "panelyzer features from %s: %s cols over %s rows (cache off)",
        cfg_path.name,
        frames.n_features(out),
        out.height,
    )
    return out


def generate_ewma_features(
    df: pl.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    span: float = 20.0,
    adjust: bool = False,
    min_periods: int | None = None,
    keep_levels: bool = True,
    drop_nulls: bool = True,
) -> pl.DataFrame:
    """Append causal EWMA columns for a subset of input features.

    Uses Polars ``ewm_mean``, which is a trailing (backward-looking) smoother —
    each row depends only on the current and past observations (no lookahead).

    Parameters
    ----------
    columns
        Feature columns to smooth. Defaults to the first five feature columns.
    span
        EWMA span (``alpha = 2 / (span + 1)`` when ``adjust`` follows Polars).
    adjust
        Passed to ``ewm_mean`` (default ``False`` for a simple recursive EWMA).
    min_periods
        Optional minimum observations before emitting a value.
    keep_levels
        If true, retain the original feature columns alongside ``*_ewma``.
    drop_nulls
        Drop rows with any null/non-finite feature after smoothing.
    """
    df = frames.ensure_date_column(df)
    feat_cols = frames.feature_columns(df)
    if columns is None:
        columns = feat_cols[:5]
    missing = [c for c in columns if c not in feat_cols]
    if missing:
        raise KeyError(f"EWMA columns not in frame: {missing}")
    if len(columns) == 0:
        raise ValueError("generate_ewma_features requires at least one column")

    ewm_kwargs: dict[str, object] = {"span": float(span), "adjust": bool(adjust)}
    if min_periods is not None:
        ewm_kwargs["min_periods"] = int(min_periods)

    exprs: list[pl.Expr] = [pl.col(frames.DATE_COL)]
    if keep_levels:
        exprs.extend(pl.col(c) for c in feat_cols)
    for c in columns:
        exprs.append(pl.col(c).ewm_mean(**ewm_kwargs).alias(f"{c}_ewma"))

    out = df.select(exprs)
    if drop_nulls:
        feat = frames.feature_columns(out)
        out = out.filter(
            pl.all_horizontal(
                [pl.col(c).is_not_null() & pl.col(c).is_finite() for c in feat]
            )
        )
    logger.info(
        "EWMA features: %s input cols → %s total features over %s rows (span=%s)",
        len(columns),
        frames.n_features(out),
        out.height,
        span,
    )
    return out
