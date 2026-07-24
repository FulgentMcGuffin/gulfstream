"""YAML-driven data source loading for the gulfstream CLI."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import polars as pl
import yaml
from mcp_data.backends.duckdb_backend import DuckDBSource
from mcp_data.backends.sqlite_backend import SQLiteSource

from gulfstream.common import frames
from gulfstream.common.options import SourceType, values as option_values
from gulfstream.data import feature_generation, synth

logger = logging.getLogger(__name__)

SUPPORTED_TYPES = tuple(option_values(SourceType))


def load_source_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Source config must be a mapping, got {type(cfg)}")
    return cfg


def load_features_from_source_yaml(
    config_path: str | Path,
    *,
    project_root: Path | None = None,
) -> tuple[pl.DataFrame, str]:
    """Load a feature DataFrame from a source YAML.

    Returns ``(df, source_type)`` where ``df`` is a dated polars frame.
    """
    cfg = load_source_config(config_path)
    source_type = str(cfg.get("type", "")).lower().strip()
    if source_type not in SUPPORTED_TYPES:
        raise ValueError(
            f"Unknown source type {source_type!r}. Supported: {', '.join(SUPPORTED_TYPES)}"
        )
    root = project_root or Path.cwd()
    handlers = {
        "synthetic": _load_synthetic,
        "faker": _load_faker,
        "parquet": _load_parquet,
        "duckdb": _load_db,
        "sqlite": _load_db,
        "csv": _load_csv,
    }
    df = handlers[source_type](cfg, root=root)
    logger.info(
        "Loaded source type=%s shape=%s",
        source_type,
        frames.shape_features(df),
    )
    return df, source_type


def _resolve_path(path: str | Path, root: Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = root / p
    return p


def _maybe_features(raw: pl.DataFrame, cfg: dict) -> pl.DataFrame:
    """Optionally run yield-feature generation on raw rate/FX frames."""
    raw = frames.ensure_date_column(raw)
    generate = cfg.get("generate_features")
    if generate is None:
        feat = frames.feature_columns(raw)
        if any(str(c).startswith("f") and str(c)[1:].isdigit() for c in feat):
            return raw
        generate = True
    if generate:
        return feature_generation.generate_yield_features(
            raw,
            vol_window=int(cfg.get("vol_window", 60)),
            corr_window=int(cfg.get("corr_window", 60)),
            include_levels=bool(cfg.get("include_levels", True)),
        )
    return raw


def _load_synthetic(cfg: dict, *, root: Path) -> pl.DataFrame:
    test_params = {
        "case": cfg.get("case", "jump"),
        "regimes": int(cfg.get("regimes", 3)),
        "features": int(cfg.get("features", 4)),
        "durations": cfg.get("durations", "uniform"),
        "regime_params": cfg.get("regime_params", "random"),
    }
    if "seed" in cfg:
        test_params["seed"] = int(cfg["seed"])
    if isinstance(cfg.get("params"), dict):
        test_params.update(cfg["params"])
    df, *_ = synth.generate_synth_data(test_params)
    return df


def _load_faker(cfg: dict, *, root: Path) -> pl.DataFrame:
    raw = synth.generate_faker_yield_frame(
        n_days=int(cfg.get("n_days", 500)),
        sources=cfg.get("sources"),
        tenors=cfg.get("tenors"),
        fx_pairs=cfg.get("fx_pairs"),
        seed=int(cfg.get("seed", 42)),
        n_regimes=int(cfg.get("n_regimes", 3)),
    )
    return _maybe_features(raw, cfg)


def _load_parquet(cfg: dict, *, root: Path) -> pl.DataFrame:
    path = cfg.get("path") or cfg.get("file")
    if not path:
        raise KeyError("parquet source requires 'path'")
    path = _resolve_path(path, root)
    if not path.exists():
        if cfg.get("create_if_missing", False):
            kind = cfg.get("create_kind", "faker_yields")
            synth.save_synthetic_parquet(
                path,
                kind=kind,
                seed=int(cfg.get("seed", 42)),
                n_days=int(cfg.get("n_days", 500)),
            )
        else:
            raise FileNotFoundError(f"Parquet file not found: {path}")
    raw = synth.load_parquet(path)
    return _maybe_features(raw, cfg)


def _load_csv(cfg: dict, *, root: Path) -> pl.DataFrame:
    path = cfg.get("path") or cfg.get("file")
    if not path:
        raise KeyError("csv source requires 'path'")
    path = _resolve_path(path, root)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    read_kwargs: dict[str, Any] = {}
    if cfg.get("separator") is not None:
        read_kwargs["separator"] = cfg["separator"]
    elif cfg.get("sep") is not None:
        read_kwargs["separator"] = cfg["sep"]
    if cfg.get("encoding") is not None:
        read_kwargs["encoding"] = cfg["encoding"]

    raw = pl.read_csv(path, try_parse_dates=True, **read_kwargs)
    date_col = cfg.get("date_column") or cfg.get("index_column") or frames.DATE_COL
    if date_col in raw.columns and date_col != frames.DATE_COL:
        raw = raw.rename({date_col: frames.DATE_COL})
    raw = frames.ensure_date_column(raw)

    start = cfg.get("start_date")
    end = cfg.get("end_date")
    if start is not None:
        raw = raw.filter(pl.col(frames.DATE_COL) >= pl.lit(start).str.to_datetime())
    if end is not None:
        raw = raw.filter(pl.col(frames.DATE_COL) <= pl.lit(end).str.to_datetime())

    columns = cfg.get("columns")
    if columns:
        missing = [c for c in columns if c not in frames.feature_columns(raw)]
        if missing:
            raise KeyError(f"CSV missing requested columns: {missing}")
        raw = frames.select_features(raw, list(columns))

    return _maybe_features(raw, cfg)


def _load_db(cfg: dict, *, root: Path) -> pl.DataFrame:
    source_type = str(cfg.get("type", "duckdb")).lower()
    db_path = cfg.get("db_path") or cfg.get("path")
    if not db_path:
        raise KeyError(f"{source_type} source requires 'db_path'")
    db_path = _resolve_path(db_path, root)

    raw = load_rates_and_fx(
        db_path,
        backend=source_type,
        rate_table=cfg.get("rate_table", "zero_rates"),
        sources=cfg.get("sources", ["USA", "DEU", "ITA"]),
        tenors=cfg.get("tenors", ["Y002p0", "Y005p0", "Y010p0", "Y030p0"]),
        fx_pairs=cfg.get("fx_pairs", ["EURUSD", "GBPUSD"]),
        fx_table=cfg.get("fx_table", "spotfx"),
        start_date=cfg.get("start_date", "2015-01-01"),
        end_date=cfg.get("end_date", "2024-12-31"),
        sql=cfg.get("sql"),
    )
    return _maybe_features(
        raw, {**cfg, "generate_features": cfg.get("generate_features", True)}
    )


def load_rates_and_fx(
    db_path: str | Path,
    *,
    backend: str = "duckdb",
    rate_table: str = "zero_rates",
    sources: Sequence[str] = ("USA", "DEU", "ITA"),
    tenors: Sequence[str] = ("Y002p0", "Y005p0", "Y010p0", "Y030p0"),
    fx_pairs: Sequence[str] = ("EURUSD", "GBPUSD"),
    fx_table: str = "spotfx",
    start_date: str = "2015-01-01",
    end_date: str = "2024-12-31",
    sql: str | None = None,
) -> pl.DataFrame:
    """Return a dated wide polars DataFrame with rate and FX columns."""
    backend = backend.lower()
    if backend not in {"duckdb", "sqlite"}:
        raise ValueError(f"backend must be duckdb or sqlite, got {backend}")

    source_cls = DuckDBSource if backend == "duckdb" else SQLiteSource
    db_path = Path(db_path)

    with source_cls(db_path, read_only=True) as db:
        if sql:
            frame = pl.from_pandas(db.run_query(sql).to_pandas())
            if "date" not in frame.columns:
                raise ValueError("Custom SQL must return a 'date' column")
            if frames.DATE_COL != "date":
                frame = frame.rename({"date": frames.DATE_COL})
            return frames.ensure_date_column(frame).sort(frames.DATE_COL)

        if rate_table not in {"zero_rates", "par_rates"}:
            raise ValueError(
                f"rate_table must be zero_rates or par_rates, got {rate_table}"
            )

        tenor_cols = ", ".join(tenors)
        source_list = ", ".join(f"'{s}'" for s in sources)
        fx_cols = ", ".join(fx_pairs) if fx_pairs else None

        rates = pl.from_pandas(
            db.run_query(
                f"""
                SELECT date, source, {tenor_cols}
                FROM {rate_table}
                WHERE source IN ({source_list})
                  AND date >= '{start_date}'
                  AND date <= '{end_date}'
                ORDER BY date, source
                """
            ).to_pandas()
        )

        fx = None
        if fx_cols:
            fx = pl.from_pandas(
                db.run_query(
                    f"""
                    SELECT date, {fx_cols}
                    FROM {fx_table}
                    WHERE date >= '{start_date}'
                      AND date <= '{end_date}'
                    ORDER BY date
                    """
                ).to_pandas()
            )

    if rates.is_empty():
        raise LookupError(
            f"No rows in {rate_table} for sources={list(sources)} "
            f"between {start_date} and {end_date}"
        )

    rates = rates.with_columns(
        pl.col("date").cast(pl.Utf8).str.to_datetime(strict=False)
    )
    pieces: list[pl.DataFrame] = []
    for source in sources:
        sub = rates.filter(pl.col("source") == source).drop("source")
        rename = {t: f"{source}_{t}" for t in tenors if t in sub.columns}
        sub = sub.rename(rename)
        pieces.append(sub)

    wide = pieces[0]
    for p in pieces[1:]:
        wide = wide.join(p, on="date", how="full", coalesce=True)

    if fx is not None and not fx.is_empty():
        fx = fx.with_columns(
            pl.col("date").cast(pl.Utf8).str.to_datetime(strict=False)
        )
        fx = fx.unique(subset=["date"], keep="last")
        wide = wide.join(fx, on="date", how="left")

    if "date" in wide.columns and frames.DATE_COL != "date":
        wide = wide.rename({"date": frames.DATE_COL})
    elif "date" in wide.columns:
        pass
    wide = frames.ensure_date_column(wide).sort(frames.DATE_COL)
    feat = frames.feature_columns(wide)
    wide = wide.drop_nulls(subset=feat)
    logger.info(
        "Loaded %s rows x %s cols from %s (%s)",
        wide.height,
        len(feat),
        backend,
        rate_table,
    )
    return wide
