"""Diebold–Li dimred: Nelson–Siegel with time-varying λ.

For each date, choose λ on a grid to minimise the OLS residual of the
Nelson–Siegel loadings, then emit (β0, β1, β2) — optionally appending λ.
"""
from __future__ import annotations

import logging

import numpy as np
import polars as pl

from gulfstream.common import frames
from gulfstream.common.options import DimredMethod
from gulfstream.common.results import DimredResults
from gulfstream.dimred.classical import _common as common
from gulfstream.dimred.classical.nelson_siegel import (
    _maturity_grid,
    _ns_loadings,
)

logger = logging.getLogger(__name__)

# Typical λ grid spanning short- to long-rate curvature peaks
_DEFAULT_LAMBDA_GRID = np.array(
    [0.01, 0.03, 0.0609, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0], dtype=float
)


def _ols_betas(y: np.ndarray, G: np.ndarray) -> tuple[np.ndarray, float]:
    """Return β and residual SSE for design G."""
    try:
        beta, residuals, *_ = np.linalg.lstsq(G, y, rcond=None)
        sse = float(residuals[0]) if len(residuals) else float(np.sum((y - G @ beta) ** 2))
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(G) @ y
        sse = float(np.sum((y - G @ beta) ** 2))
    return beta, sse


def _diebold_li_dimred(
    df: pl.DataFrame,
    *,
    dl_lambda_grid: list[float] | None = None,
    dl_include_lambda: bool = False,
    ns_maturities: list[float] | None = None,
    **kwargs,
) -> DimredResults:
    X = frames.to_numpy(df)
    tau = _maturity_grid(df, ns_maturities)
    if X.shape[1] != len(tau):
        raise ValueError("Diebold–Li maturity grid length must match feature count.")
    if X.shape[1] < 3:
        raise ValueError("Diebold–Li needs at least 3 features (tenors).")

    grid = (
        np.asarray(dl_lambda_grid, dtype=float)
        if dl_lambda_grid is not None
        else _DEFAULT_LAMBDA_GRID
    )
    grid = grid[grid > 0]
    if grid.size == 0:
        grid = _DEFAULT_LAMBDA_GRID

    # Precompute loadings for each λ
    loadings = {float(lam): _ns_loadings(tau, float(lam)) for lam in grid}

    n = X.shape[0]
    betas = np.zeros((n, 3), dtype=float)
    lams = np.zeros(n, dtype=float)
    for i in range(n):
        y = X[i]
        best_sse = float("inf")
        best_beta = np.zeros(3)
        best_lam = float(grid[0])
        for lam, G in loadings.items():
            beta, sse = _ols_betas(y, G)
            if sse < best_sse:
                best_sse = sse
                best_beta = beta
                best_lam = float(lam)
        betas[i] = best_beta
        lams[i] = best_lam

    if dl_include_lambda:
        out = np.column_stack([betas, lams])
        rank = 4
    else:
        out = betas
        rank = 3

    return DimredResults(
        df=frames.with_same_dates(out, df),
        dimred=DimredMethod.DIEBOLD_LI,
        rank=rank,
        rank_selection_method="user_specified",
        ns_lambda=float(np.median(lams)),
        model={"lambdas": lams.tolist(), "lambda_grid": grid.tolist()},
    )


def _diebold_li_generator(df: pl.DataFrame, params: dict):
    if not common._need_dimred(DimredMethod.DIEBOLD_LI, params):
        return
    mats = params["algo"].get("ns_maturities")
    if mats and isinstance(mats[0], (int, float)):
        maturities = [float(x) for x in mats]
    else:
        maturities = None
    grids = params["algo"].get("dl_lambda_grid") or [None]
    # Allow a single flat list of floats as one grid
    if grids and isinstance(grids[0], (int, float)):
        grids = [grids]
    include_flags = params["algo"].get("dl_include_lambda", [False])
    for grid in grids:
        for include in include_flags:
            try:
                yield _diebold_li_dimred(
                    df,
                    dl_lambda_grid=None if grid is None else [float(x) for x in grid],
                    dl_include_lambda=bool(include),
                    ns_maturities=maturities,
                )
            except ValueError as exc:
                logger.warning("Diebold–Li skipped: %s", exc)
                return
