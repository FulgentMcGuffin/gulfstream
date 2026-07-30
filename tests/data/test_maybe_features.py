"""Tests for pluggable feature generation in source loading."""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from gulfstream.common import frames
from gulfstream.data import feature_generation
from gulfstream.data.source_loader import _maybe_features, resolve_feature_generator


def _rates_frame(n: int = 120) -> pl.DataFrame:
    """Minimal USA/DEU curve + FX frame shaped like the DuckDB loader output."""
    dates = [datetime(2018, 1, 1) + timedelta(days=i) for i in range(n)]
    rng = np.random.default_rng(0)
    data = {frames.DATE_COL: dates}
    for source, base in (("USA", 2.0), ("DEU", 0.5)):
        for i, tenor in enumerate(("Y002p0", "Y005p0", "Y010p0", "Y030p0")):
            data[f"{source}_{tenor}"] = (
                base + 0.3 * i + np.cumsum(rng.normal(0, 0.02, n))
            ).tolist()
    data["EURUSD"] = (1.1 + np.cumsum(rng.normal(0, 0.001, n))).tolist()
    return pl.DataFrame(data)


def _panel_frame(n: int = 80, n_features: int = 8) -> pl.DataFrame:
    dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n)]
    rng = np.random.default_rng(1)
    data = {frames.DATE_COL: dates}
    for i in range(n_features):
        data[f"x{i}"] = rng.normal(0, 1, n).tolist()
    return pl.DataFrame(data)


def test_resolve_feature_generator_dotted_path():
    fn = resolve_feature_generator(
        "gulfstream.data.feature_generation.generate_yield_features"
    )
    assert fn is feature_generation.generate_yield_features


def test_maybe_features_requires_generator_when_enabled():
    raw = _rates_frame()
    with pytest.raises(ValueError, match="feature_generator"):
        _maybe_features(raw, {"generate_features": True})


def test_yield_features_via_abstracted_generator():
    """Current yield FE, invoked through the pluggable feature_generator hook."""
    raw = _rates_frame()
    cfg = {
        "generate_features": True,
        "feature_generator": "gulfstream.data.feature_generation.generate_yield_features",
        "feature_generator_kwargs": {
            "vol_window": 30,
            "corr_window": 30,
            "include_levels": True,
        },
    }
    out = _maybe_features(raw, cfg)
    assert frames.DATE_COL in out.columns
    feat = frames.feature_columns(out)
    # Spreads / flies / vols appear beyond the raw columns.
    assert len(feat) > len(frames.feature_columns(raw))
    assert any("_minus_" in c for c in feat)
    assert any(c.endswith("_vol") for c in feat)
    # Same result as calling the function directly.
    direct = feature_generation.generate_yield_features(
        raw, vol_window=30, corr_window=30, include_levels=True
    )
    assert out.columns == direct.columns
    assert out.height == direct.height


def test_identity_feature_generator_leaves_raw_unchanged():
    raw = _panel_frame()
    cfg = {
        "generate_features": True,
        "feature_generator": "gulfstream.data.feature_generation.identity_features",
    }
    out = _maybe_features(raw, cfg)
    assert out.columns == raw.columns
    assert out.height == raw.height
    assert out.equals(frames.ensure_date_column(raw))


def test_ewma_feature_generator_on_five_columns_no_lookahead():
    raw = _panel_frame(n=60, n_features=8)
    columns = frames.feature_columns(raw)[:5]
    span = 10.0
    cfg = {
        "generate_features": True,
        "feature_generator": "gulfstream.data.feature_generation.generate_ewma_features",
        "feature_generator_kwargs": {
            "columns": columns,
            "span": span,
            "adjust": False,
            "keep_levels": True,
            "drop_nulls": False,
        },
    }
    out = _maybe_features(raw, cfg)
    ewma_cols = [f"{c}_ewma" for c in columns]
    assert all(c in out.columns for c in ewma_cols)
    assert all(c in out.columns for c in frames.feature_columns(raw))

    # Causal check: truncating the series must not change past EWMA values.
    cut = 40
    prefix = raw.head(cut)
    out_full = feature_generation.generate_ewma_features(
        raw,
        columns=columns,
        span=span,
        adjust=False,
        keep_levels=True,
        drop_nulls=False,
    )
    out_prefix = feature_generation.generate_ewma_features(
        prefix,
        columns=columns,
        span=span,
        adjust=False,
        keep_levels=True,
        drop_nulls=False,
    )
    for c in ewma_cols:
        full_head = out_full[c].head(cut).to_list()
        pref = out_prefix[c].to_list()
        assert full_head == pytest.approx(pref, rel=1e-9, abs=1e-9)
