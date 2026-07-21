"""Synthetic regime data and Faker-based fake yield frames for benchmarks."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
from faker import Faker

from gulfstream.common import frames
from gulfstream.common import utils

logger = logging.getLogger(__name__)


def generate_synth_data(test_params: dict):
    """Generate synthetic multi-regime time series.

    Parameters
    ----------
    test_params : dict
        Keys: case, regimes, features, durations ('uniform'|'poisson'|list),
        regime_params ('random'|dict), optional seed.
    """
    case = test_params.get("case")
    if not case:
        raise KeyError("'case' must be specified.")
    seed = test_params.get("seed")
    if seed is not None:
        np.random.seed(int(seed))

    regimes = int(test_params["regimes"])
    features = int(test_params["features"])
    regime_params = test_params.get("regime_params", "random")

    if regime_params == "random":
        durations = _random_durations(regimes, test_params.get("durations", "uniform"))
        regime_params = _random_params(case, regimes, features)
    else:
        durations = test_params["durations"]
        if len(durations) != regimes:
            raise ValueError("'durations' length must equal 'regimes'.")

    handlers = {
        "jump": _gen_jump,
        "correlated": _gen_correlated,
        "piecewise": _gen_piecewise,
        "crisis": _gen_crisis,
    }
    handler = handlers.get(case)
    if handler is None:
        raise ValueError(
            f"Unknown case {case}. Supported for core path: {list(handlers)}"
        )

    df, x, labels = handler(regimes, features, durations, regime_params)
    bkpts = utils._convert_labels_to_bkpts(labels)
    return df, x, labels, bkpts, list(durations), regime_params


def _random_durations(n: int, distribution) -> list[int]:
    if isinstance(distribution, list):
        return list(distribution)
    if distribution == "uniform":
        return np.random.randint(100, 301, size=n).tolist()
    if distribution in ("poisson", "skewed"):
        return np.maximum(50, np.random.poisson(200, size=n)).tolist()
    raise ValueError(f"Unknown durations {distribution}")


def _random_params(case: str, regimes: int, features: int) -> dict:
    params = {}
    for i in range(regimes):
        if case in ("jump", "piecewise", "crisis"):
            params[i] = {
                "mean": np.random.uniform(1.0, 8.0, size=features),
                "vol": np.random.uniform(0.05, 0.4, size=features),
            }
        elif case == "correlated":
            a = np.random.uniform(0.1, 1.0, size=(features, features))
            params[i] = {
                "mean": np.random.uniform(1.0, 8.0, size=features),
                "cov": a.T @ a,
            }
        else:
            raise ValueError(case)
    if case == "crisis" and regimes >= 2:
        mid = regimes // 2
        params[mid]["vol"] = params[mid]["vol"] * 4.0
        params[mid]["mean"] = params[mid]["mean"] + 2.0
    return params


def _business_dates(n: int, start: str = "2000-01-01") -> list[datetime]:
    """Approximate business-day calendar (skip Sat/Sun)."""
    d = datetime.fromisoformat(start)
    out: list[datetime] = []
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def _assemble(segments: list[np.ndarray], labels_parts: list[np.ndarray]):
    x = np.vstack(segments)
    labels = np.concatenate(labels_parts).tolist()
    cols = [f"f{i}" for i in range(x.shape[1])]
    df = frames.from_numpy(x, dates=_business_dates(len(x)), columns=cols)
    return df, x, labels


def _gen_jump(regimes, features, durations, regime_params):
    segs, labs = [], []
    for i, dur in enumerate(durations):
        p = regime_params[i]
        mean, vol = np.asarray(p["mean"]), np.asarray(p["vol"])
        noise = np.random.normal(0, 1, size=(dur, features)) * vol
        jumps = (np.random.rand(dur, features) < 0.05) * np.random.normal(
            0, 1.5, size=(dur, features)
        )
        segs.append(mean + noise + jumps)
        labs.append(np.full(dur, i, dtype=int))
    return _assemble(segs, labs)


def _gen_correlated(regimes, features, durations, regime_params):
    segs, labs = [], []
    for i, dur in enumerate(durations):
        p = regime_params[i]
        mean, cov = np.asarray(p["mean"]), np.asarray(p["cov"])
        segs.append(np.random.multivariate_normal(mean, cov, size=dur))
        labs.append(np.full(dur, i, dtype=int))
    return _assemble(segs, labs)


def _gen_piecewise(regimes, features, durations, regime_params):
    segs, labs = [], []
    for i, dur in enumerate(durations):
        p = regime_params[i]
        mean, vol = np.asarray(p["mean"]), np.asarray(p["vol"])
        segs.append(mean + np.random.normal(0, 1, size=(dur, features)) * vol)
        labs.append(np.full(dur, i, dtype=int))
    return _assemble(segs, labs)


def _gen_crisis(regimes, features, durations, regime_params):
    return _gen_jump(regimes, features, durations, regime_params)


def generate_faker_yield_frame(
    *,
    n_days: int = 500,
    sources: list[str] | None = None,
    tenors: list[str] | None = None,
    fx_pairs: list[str] | None = None,
    seed: int = 42,
    n_regimes: int = 3,
) -> pl.DataFrame:
    """Create a fake yield/FX wide frame shaped like DuckDB loader output."""
    fake = Faker()
    Faker.seed(seed)
    np.random.seed(seed)

    sources = sources or ["USA", "DEU", "ITA"]
    tenors = tenors or ["Y002p0", "Y005p0", "Y010p0", "Y030p0"]
    fx_pairs = fx_pairs or ["EURUSD", "GBPUSD"]

    dates = _business_dates(n_days, start="2018-01-01")
    edges = np.linspace(0, n_days, n_regimes + 1, dtype=int)
    base_levels = {
        "USA": {t: 1.5 + 0.3 * i for i, t in enumerate(tenors)},
        "DEU": {t: 0.5 + 0.25 * i for i, t in enumerate(tenors)},
        "ITA": {t: 2.0 + 0.35 * i for i, t in enumerate(tenors)},
    }
    data: dict[str, np.ndarray] = {}
    for source in sources:
        for tenor in tenors:
            series = np.zeros(n_days)
            for r in range(n_regimes):
                a, b = edges[r], edges[r + 1]
                shift = r * 0.8 + (0.2 if source == "ITA" else 0.0)
                level = base_levels.get(source, base_levels["USA"])[tenor] + shift
                jitter = float(fake.pyfloat(min_value=-0.05, max_value=0.05))
                series[a:b] = level + jitter + np.cumsum(np.random.normal(0, 0.02, b - a))
            data[f"{source}_{tenor}"] = series

    for pair in fx_pairs:
        base = 1.1 if pair.startswith("EUR") else 1.3
        data[pair] = base + np.cumsum(np.random.normal(0, 0.002, n_days))

    cols = list(data.keys())
    mat = np.column_stack([data[c] for c in cols])
    df = frames.from_numpy(mat, dates=dates, columns=cols)
    logger.info("Faker yield frame: %s rows x %s cols (%s)", df.height, len(cols), fake.name())
    return df


def save_synthetic_parquet(
    path: str | Path,
    *,
    kind: Literal["faker_yields", "jump"] = "faker_yields",
    **kwargs,
) -> Path:
    """Generate and persist a synthetic dataframe for offline benchmarks."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if kind == "faker_yields":
        df = generate_faker_yield_frame(**kwargs)
    elif kind == "jump":
        df, *_ = generate_synth_data(
            {
                "case": "jump",
                "regimes": kwargs.get("n_regimes", 3),
                "features": kwargs.get("features", 4),
                "durations": "uniform",
                "regime_params": "random",
                "seed": kwargs.get("seed", 42),
            }
        )
    else:
        raise ValueError(kind)
    df.write_parquet(path)
    logger.info("Wrote synthetic data to %s", path)
    return path


def load_parquet(path: str | Path) -> pl.DataFrame:
    df = pl.read_parquet(path)
    if frames.DATE_COL in df.columns:
        return frames.ensure_date_column(df)
    # Legacy pandas-style parquet with date as index written as column on reset.
    for cand in ("index", "__index_level_0__", "Date", "DATE"):
        if cand in df.columns:
            df = df.rename({cand: frames.DATE_COL})
            return frames.ensure_date_column(df)
    return frames.ensure_date_column(df)
