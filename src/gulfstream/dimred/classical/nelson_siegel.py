"""Nelson–Siegel factor dimred for curve-like panels.

Fits level / slope / curvature (β0, β1, β2) per date by OLS on a fixed λ,
treating feature columns as ordered maturities. Useful for yield / FX tenor
panels; for generic features, maturities default to 1..p.
"""
from __future__ import annotations

import logging
import re

import numpy as np
import polars as pl

from gulfstream.common import frames
from gulfstream.common.options import DimredMethod
from gulfstream.common.results import DimredResults
from gulfstream.dimred.classical import _common as common

logger = logging.getLogger(__name__)


def _parse_maturity_years(name: str) -> float | None:
    """Best-effort parse of tenor strings like ``Y002p0``, ``10Y``, ``2.5y``."""
    s = str(name).lower().replace("_", "")
    m = re.search(r"y0*(\d+)p(\d+)", s)
    if m:
        return float(m.group(1)) + float(m.group(2)) / 10.0
    m = re.search(r"(\d+(?:\.\d+)?)y", s)
    if m:
        return float(m.group(1))
    m = re.search(r"y(\d+)", s)
    if m:
        return float(m.group(1))
    return None


def _maturity_grid(df: pl.DataFrame, maturities: list[float] | None) -> np.ndarray:
    cols = frames.feature_columns(df)
    if maturities is not None and len(maturities) == len(cols):
        return np.asarray(maturities, dtype=float)
    parsed = [_parse_maturity_years(c) for c in cols]
    if all(v is not None and v > 0 for v in parsed):
        return np.asarray(parsed, dtype=float)
    # Fallback: evenly spaced 1..p years
    return np.arange(1, len(cols) + 1, dtype=float)


def _ns_loadings(tau: np.ndarray, lam: float) -> np.ndarray:
    """Return (n_tau, 3) design matrix for Nelson–Siegel factors."""
    tau = np.asarray(tau, dtype=float)
    lam = float(lam)
    with np.errstate(divide="ignore", invalid="ignore"):
        x = lam * tau
        # (1 - e^{-x}) / x
        factor = np.where(np.abs(x) < 1e-8, 1.0 - x / 2.0, (1.0 - np.exp(-x)) / x)
        slope = factor
        curve = factor - np.exp(-x)
    ones = np.ones_like(tau)
    return np.column_stack([ones, slope, curve])


def _nelson_siegel_dimred(
    df: pl.DataFrame,
    *,
    ns_lambda: float = 0.0609,
    ns_maturities: list[float] | None = None,
    **kwargs,
) -> DimredResults:
    X = frames.to_numpy(df)
    tau = _maturity_grid(df, ns_maturities)
    if X.shape[1] != len(tau):
        raise ValueError("Nelson–Siegel maturity grid length must match feature count.")
    if X.shape[1] < 3:
        raise ValueError("Nelson–Siegel needs at least 3 features (tenors).")

    G = _ns_loadings(tau, ns_lambda)  # (p, 3)
    # OLS per row: β = (G'G)^{-1} G' y
    gtg = G.T @ G
    try:
        gtgi = np.linalg.inv(gtg)
    except np.linalg.LinAlgError:
        gtgi = np.linalg.pinv(gtg)
    betas = (gtgi @ G.T @ X.T).T  # (n, 3)
    return DimredResults(
        df=frames.with_same_dates(betas, df),
        dimred=DimredMethod.NELSON_SIEGEL,
        rank=3,
        rank_selection_method="user_specified",
        ns_lambda=float(ns_lambda),
        model={"lambda": float(ns_lambda), "maturities": tau.tolist(), "loadings": G},
    )


def _nelson_siegel_generator(df: pl.DataFrame, params: dict):
    if not common._need_dimred(DimredMethod.NELSON_SIEGEL, params):
        return
    mats = params["algo"].get("ns_maturities")
    # Allow a single flat list (not grid) of maturities
    if mats and isinstance(mats[0], (int, float)):
        maturities = [float(x) for x in mats]
    else:
        maturities = None
    for lam in params["algo"].get("ns_lambda", [0.0609]):
        try:
            yield _nelson_siegel_dimred(
                df, ns_lambda=float(lam), ns_maturities=maturities
            )
        except ValueError as exc:
            logger.warning("Nelson–Siegel skipped: %s", exc)
            return
