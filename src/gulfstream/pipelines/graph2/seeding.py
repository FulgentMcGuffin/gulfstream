"""Seed regimes resolution and Graph 1 → Graph 2 regimes_df conversion."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import polars as pl

from gulfstream.common import frames
from gulfstream.common.results import SegmentResults
from gulfstream.detection.time_index import regimes_df_to_bkpts as _regimes_df_to_bkpts
from gulfstream.features import names as feature_name_resolution

logger = logging.getLogger(__name__)


def regimes_df_to_bkpts(
    df: pl.DataFrame,
    regimes_df: pl.DataFrame | None,
) -> tuple[list[int], dict[int, int]]:
    """Thin wrapper around ``gulfstream.detection.time_index.regimes_df_to_bkpts``."""
    return _regimes_df_to_bkpts(df, regimes_df)


def seed_regimes_from_results(df: pl.DataFrame, res: SegmentResults) -> pl.DataFrame:
    """Convert SegmentResults breakpoints into a Graph 2 ``regimes_df``.

    Graph 2 reads ``End`` on all but the last row as *breakpoint dates*
    (see ``regimes_df_to_bkpts``), so ``End`` must be ``dates[bkpt]``, not the
    last observation of the preceding regime. The final row is the series end
    with ``Hierarchy Level of End`` of 0.
    """
    dates = frames.dates_series(df).to_list()
    n = len(dates)
    bkpts = sorted(int(b) for b in (res.bkpts or []) if 0 < int(b) < n)
    hierarchy = {
        int(k): int(v)
        for k, v in (res.hierarchy or {b: 1 for b in bkpts}).items()
    }
    rows = []
    for i, b in enumerate(bkpts):
        start_i = 0 if i == 0 else bkpts[i - 1]
        rows.append(
            {
                "Start": dates[start_i],
                "End": dates[b],
                "Regime": i,
                "Hierarchy Level of End": int(hierarchy.get(b, 1)),
            }
        )
    start_last = bkpts[-1] if bkpts else 0
    rows.append(
        {
            "Start": dates[start_last],
            "End": dates[n - 1],
            "Regime": len(bkpts),
            "Hierarchy Level of End": 0,
        }
    )
    return pl.DataFrame(rows)


def _resolve_regimes_df(spec: Any, project_root: Path | None = None) -> pl.DataFrame | None:
    """Load seed regimes from null / path / CSV / records / DataFrame."""
    if spec is None:
        return None
    if isinstance(spec, pl.DataFrame):
        return spec
    if isinstance(spec, dict) and "path" in spec:
        path = Path(spec["path"])
        if not path.is_absolute() and project_root is not None:
            path = project_root / path
        return pl.read_csv(path)
    if isinstance(spec, (str, Path)):
        path = Path(spec)
        if not path.is_absolute() and project_root is not None:
            path = project_root / path
        return pl.read_csv(path)
    if isinstance(spec, list):
        return pl.DataFrame(spec)
    if isinstance(spec, dict):
        # Inline single-record dict without path — treat as empty unless columns present.
        if {"End", "Hierarchy Level of End"} <= set(spec.keys()):
            return pl.DataFrame([spec])
        return None
    raise TypeError(f"Unsupported regimes_df spec type: {type(spec)}")


def _resolve_retrain_features(df: pl.DataFrame, features_spec: Any) -> list[str]:
    if features_spec is None:
        raise TypeError("params['retrain']['features'] must be provided.")
    if isinstance(features_spec, list):
        if features_spec == ["__auto__"] or (
            len(features_spec) == 1 and features_spec[0] == "__auto__"
        ):
            return frames.feature_columns(df)
        names = feature_name_resolution.get_column_names(features_spec)
    elif isinstance(features_spec, dict):
        names = feature_name_resolution.get_column_names(features_spec)
    elif features_spec == "__auto__":
        return frames.feature_columns(df)
    else:
        raise TypeError("params['retrain']['features'] must be type dict or list[str].")
    feat_cols = set(frames.feature_columns(df))
    extra = [c for c in names if c not in feat_cols]
    if extra:
        logger.warning(
            "The following features are not present in df and will be ignored: %s.",
            ", ".join(extra),
        )
    return [c for c in names if c in feat_cols]
