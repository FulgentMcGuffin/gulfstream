"""Ruptures model-based dimred (delegates to detectors.ruptures_methods)."""
from __future__ import annotations

import numpy as np
import polars as pl

from gulfstream.common import frames
from gulfstream.common.options import DimredMethod
from gulfstream.common.results import DimredResults
from gulfstream.dimred.model_based._common import _as_frame
from gulfstream.detectors import ruptures_methods as ruptures_mod


def _ruptures_dimred(
    df: pl.DataFrame,
    ruptures_algorithm: str,
    ruptures_cost_params: dict | None = None,
    regimes: int | None = None,
    ruptures_penalty: float | None = None,
    ruptures_window: int | None = None,
    ruptures_rec_error: float | None = None,
    **kwargs,
) -> DimredResults:
    """Distances to ruptures-segment means as embedding columns."""
    cost = ruptures_cost_params or {"model": "l2"}
    res = ruptures_mod.ruptures_predict_regimes(
        df,
        ruptures_algorithm=ruptures_algorithm,
        ruptures_cost_params=cost,
        regimes=regimes,
        ruptures_penalty=ruptures_penalty,
        ruptures_window=ruptures_window,
        ruptures_rec_error=ruptures_rec_error,
    )
    labels = np.asarray(res.labels, dtype=int)
    X = frames.to_numpy(df)
    uniq = sorted(set(labels.tolist()))
    if not uniq:
        uniq = [0]
        labels = np.zeros(df.height, dtype=int)

    means = []
    for lab in uniq:
        mask = labels == lab
        means.append(X[mask].mean(axis=0))
    means = np.vstack(means)
    dists = np.linalg.norm(X[:, None, :] - means[None, :, :], axis=2)
    return DimredResults(
        df=_as_frame(dists, df, "ruptures_"),
        dimred="ruptures",
        rank=dists.shape[1],
        rank_selection_method="regimes",
        model={
            "bkpts": res.bkpts,
            "labels": labels.tolist(),
            "algorithm": ruptures_algorithm,
        },
    )


def _ruptures_generator(df: pl.DataFrame, params: dict):
    if DimredMethod.RUPTURES not in params.get("algo", {}).get("dimred", []):
        return
    algos = params["algo"].get("ruptures_algorithm", ["pelt"])
    costs = params["algo"].get("ruptures_cost_params", [{"model": "l2"}])
    for algo in algos:
        for cost in costs:
            if algo == "pelt":
                for pen in params["algo"].get("ruptures_penalty", [None]):
                    yield _ruptures_dimred(
                        df,
                        ruptures_algorithm=algo,
                        ruptures_cost_params=cost,
                        ruptures_penalty=pen,
                    )
            elif algo == "dynp":
                for regimes in params["algo"]["regimes"]:
                    yield _ruptures_dimred(
                        df,
                        ruptures_algorithm=algo,
                        ruptures_cost_params=cost,
                        regimes=regimes,
                    )
            elif algo == "window":
                for window in params["algo"]["ruptures_window"]:
                    if "regimes" in params["algo"]:
                        for regimes in params["algo"]["regimes"]:
                            yield _ruptures_dimred(
                                df,
                                ruptures_algorithm=algo,
                                ruptures_cost_params=cost,
                                ruptures_window=window,
                                regimes=regimes,
                            )
                    elif "ruptures_penalty" in params["algo"]:
                        for pen in params["algo"]["ruptures_penalty"]:
                            yield _ruptures_dimred(
                                df,
                                ruptures_algorithm=algo,
                                ruptures_cost_params=cost,
                                ruptures_window=window,
                                ruptures_penalty=pen,
                            )
                    else:
                        for eps in params["algo"]["ruptures_rec_error"]:
                            yield _ruptures_dimred(
                                df,
                                ruptures_algorithm=algo,
                                ruptures_cost_params=cost,
                                ruptures_window=window,
                                ruptures_rec_error=eps,
                            )
            else:
                raise ValueError(f"Unknown ruptures_algorithm for dimred: {algo}")
