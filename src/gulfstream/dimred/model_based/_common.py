"""Shared helpers for model-based dimensionality-reduction generators."""
from __future__ import annotations

import numpy as np
import polars as pl

from gulfstream.common import frames


def _require_regimes(regimes: int | None) -> int:
    if regimes is None or not isinstance(regimes, int) or regimes < 1:
        raise ValueError("'regimes' must be a positive int for model-based dimred.")
    return regimes


def _as_frame(x: np.ndarray, template: pl.DataFrame, prefix: str) -> pl.DataFrame:
    cols = [f"{prefix}{i}" for i in range(x.shape[1])]
    return frames.with_same_dates(x, template, columns=cols)
