"""Synthetic regime data and Faker-based fake yield frames for benchmarks."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
from faker import Faker
from hmmlearn.hmm import GaussianHMM

from gulfstream.common import frames
from gulfstream.common import utils

logger = logging.getLogger(__name__)

LABEL_COL = "true_regime"


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


@dataclass(frozen=True)
class HmmPanelResult:
    """Faker HMM panel plus ground-truth outer-regime labels."""

    df: pl.DataFrame
    labels: list[int]
    bkpts: list[int]
    regime_sequence: list[int]
    feature_names: list[str]


def _sanitize_feature_name(raw: str, idx: int) -> str:
    name = re.sub(r"[^0-9A-Za-z_]+", "_", str(raw).strip().lower()).strip("_")
    if not name or name[0].isdigit():
        name = f"feat_{name}" if name else f"feat_{idx}"
    return name[:48]


def _faker_feature_names(n_features: int, *, seed: int) -> list[str]:
    fake = Faker("en_US")
    Faker.seed(seed)
    names: list[str] = []
    seen: set[str] = set()
    for i in range(n_features):
        base = _sanitize_feature_name(fake.unique.word(), i)
        name = base
        k = 2
        while name in seen:
            name = f"{base}_{k}"
            k += 1
        seen.add(name)
        names.append(name)
    return names


def _spd_cov(rng: np.random.Generator, dim: int, scale: float) -> np.ndarray:
    a = rng.normal(0.0, 1.0, size=(dim, dim))
    cov = a @ a.T / dim
    cov = cov * (scale**2) + np.eye(dim) * (0.05 * scale**2)
    return cov


def _regime_hmm_params(
    regime_id: int,
    n_features: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Two MVN emission components + sticky 2-state transition for one outer regime."""
    # Separable outer-regime centres so detectors recover breaks with mixed accuracy.
    centres = {
        0: (-2.0, 0.6),
        1: (2.2, 0.7),
        2: (0.0, 1.8),
        3: (1.0, 1.1),
    }
    mean_shift, scale = centres.get(regime_id, (float(regime_id), 1.0))
    offset = rng.normal(0.0, 0.15, size=n_features)
    means = np.vstack(
        [
            np.full(n_features, mean_shift) + offset,
            np.full(n_features, mean_shift + 0.9 * scale) - 0.5 * offset,
        ]
    )
    covs = np.stack(
        [
            _spd_cov(rng, n_features, scale),
            _spd_cov(rng, n_features, 0.75 * scale),
        ]
    )
    # Sticky within-regime HMM (mix of the two MVNs).
    stay = 0.92
    transmat = np.array([[stay, 1.0 - stay], [1.0 - stay, stay]], dtype=float)
    startprob = np.array([0.55, 0.45], dtype=float)
    return means, covs, transmat, startprob


def _sample_hmm_segment(
    n: int,
    means: np.ndarray,
    covs: np.ndarray,
    transmat: np.ndarray,
    startprob: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    model = GaussianHMM(
        n_components=2,
        covariance_type="full",
        init_params="",
        params="",
        random_state=int(rng.integers(0, 2**31 - 1)),
    )
    model.startprob_ = startprob
    model.transmat_ = transmat
    model.means_ = means
    model.covars_ = covs
    x, _states = model.sample(n)
    return np.asarray(x, dtype=float)


def _outer_regime_sequence(
    n_regimes: int,
    *,
    n_repeating: int,
    rng: np.random.Generator,
) -> list[int]:
    """Build a segment label sequence with ``n_regimes`` types and repeats.

    Exactly ``n_repeating`` regime ids appear twice; the rest appear once.
    """
    if n_regimes < 1:
        raise ValueError("n_regimes must be >= 1")
    if not (0 <= n_repeating <= n_regimes):
        raise ValueError("n_repeating must be in [0, n_regimes]")
    base = list(range(n_regimes))
    repeat_ids = list(rng.choice(base, size=n_repeating, replace=False)) if n_repeating else []
    seq = base + repeat_ids
    rng.shuffle(seq)
    return [int(x) for x in seq]


def generate_faker_hmm_panel(
    *,
    n_years: float = 10.0,
    n_features: int = 10,
    n_regimes: int = 4,
    n_repeating: int = 2,
    seed: int = 42,
    start: str = "2014-01-01",
    n_days: int | None = None,
    include_labels: bool = False,
) -> HmmPanelResult:
    """Faker-named panel from outer regimes, each an HMM mix of two MVNs.

    Default layout: ~10 years of business days, 10 features, 4 outer regimes,
    with 2 regime ids repeating later in the sample. No yield-feature engineering
    is applied — returned columns are the simulated series themselves.
    """
    rng = np.random.default_rng(int(seed))
    np.random.seed(int(seed))
    if n_days is None:
        # ~252 business days / year
        n_days = int(round(float(n_years) * 252))
    n_days = max(int(n_days), n_regimes + n_repeating)

    feature_names = _faker_feature_names(n_features, seed=seed)
    regime_sequence = _outer_regime_sequence(
        n_regimes, n_repeating=n_repeating, rng=rng
    )
    n_segments = len(regime_sequence)
    # Uneven durations so break recovery difficulty varies by method.
    weights = rng.uniform(0.7, 1.4, size=n_segments)
    weights = weights / weights.sum()
    durations = np.maximum(40, np.floor(weights * n_days).astype(int))
    # Fix rounding so sum == n_days.
    durations[-1] += n_days - int(durations.sum())
    if durations[-1] < 40:
        donor = int(np.argmax(durations[:-1]))
        need = 40 - int(durations[-1])
        durations[donor] -= need
        durations[-1] += need

    segments: list[np.ndarray] = []
    labels_parts: list[np.ndarray] = []
    for regime_id, dur in zip(regime_sequence, durations, strict=True):
        means, covs, transmat, startprob = _regime_hmm_params(
            int(regime_id), n_features, rng
        )
        segments.append(
            _sample_hmm_segment(int(dur), means, covs, transmat, startprob, rng)
        )
        labels_parts.append(np.full(int(dur), int(regime_id), dtype=int))

    x = np.vstack(segments)
    labels = np.concatenate(labels_parts).astype(int).tolist()
    dates = _business_dates(len(labels), start=start)
    df = frames.from_numpy(x, dates=dates, columns=feature_names)
    if include_labels:
        df = df.with_columns(pl.Series(LABEL_COL, labels))
    bkpts = utils._convert_labels_to_bkpts(labels)
    logger.info(
        "Faker HMM panel: %s rows x %s feats, outer sequence=%s, bkpts=%s",
        df.height,
        n_features,
        regime_sequence,
        bkpts,
    )
    return HmmPanelResult(
        df=df,
        labels=labels,
        bkpts=bkpts,
        regime_sequence=regime_sequence,
        feature_names=feature_names,
    )


def drop_label_column(df: pl.DataFrame) -> pl.DataFrame:
    """Remove ground-truth label column if present."""
    if LABEL_COL in df.columns:
        return df.drop(LABEL_COL)
    return df


def save_synthetic_parquet(
    path: str | Path,
    *,
    kind: Literal["faker_yields", "jump", "hmm_panel"] = "faker_yields",
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
    elif kind == "hmm_panel":
        include_labels = bool(kwargs.pop("include_labels", True))
        result = generate_faker_hmm_panel(include_labels=include_labels, **kwargs)
        df = result.df
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
