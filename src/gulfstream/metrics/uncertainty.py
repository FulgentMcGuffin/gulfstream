"""Calibrated uncertainty bands on breakpoints from ensemble samples."""
from __future__ import annotations

import copy
import logging

import numpy as np
import polars as pl

from gulfstream.common import frames
from gulfstream.common.results import SegmentResults
from gulfstream.metrics import evaluation as evaluation_tools
from gulfstream.metrics import robustness as robustness_mod
from gulfstream.pipelines import single_pass as single_run

logger = logging.getLogger(__name__)


def _slim_params(params: dict) -> dict:
    out = copy.deepcopy(params)
    metrics = out.get("metrics") or {}
    out["metrics"] = {**metrics, "plot": False, "mode": metrics.get("mode", "write")}
    out["robustness"] = {"enabled": False}
    out["stability"] = {"enabled": False}
    out["streaming"] = {"enabled": False}
    out["panel"] = {"enabled": False}
    out["uncertainty"] = {"enabled": False}
    return out


def _block_bootstrap(
    df: pl.DataFrame,
    block: int,
    rng: np.random.Generator,
) -> pl.DataFrame:
    """Circular block bootstrap preserving the date column length."""
    n = df.height
    block = max(2, min(int(block), n))
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=n_blocks)
    idx: list[int] = []
    for s in starts:
        for j in range(block):
            idx.append(int((s + j) % n))
            if len(idx) >= n:
                break
        if len(idx) >= n:
            break
    idx = idx[:n]
    feat = frames.to_numpy(df)
    boot = feat[idx]
    return frames.with_same_dates(boot, df)


def _collect_robustness_samples(
    df: pl.DataFrame,
    params: dict,
) -> list[list[int]]:
    pipeline_params = params.get("_pipeline_params") or params
    cfg = params.get("robustness") or {}
    perturbations = cfg.get("perturbations") or {
        "depth": [1, 2],
        "significance_level": [0.01, 0.05],
        "min_regime_length": [10, 20],
    }
    runs = robustness_mod._one_at_a_time_perturbations(pipeline_params, perturbations)
    samples: list[list[int]] = []
    slim_base = _slim_params(pipeline_params)
    for label, p in runs:
        try:
            # merge slim flags onto perturbation params
            p = {**slim_base, **p, "algo": p.get("algo", slim_base["algo"]), "test": p.get("test", slim_base["test"])}
            p["metrics"] = slim_base["metrics"]
            p["robustness"] = {"enabled": False}
            p["stability"] = {"enabled": False}
            p["uncertainty"] = {"enabled": False}
            res = single_run.run_single_segmentation(df, p)
            samples.append(list(res.bkpts))
            logger.info("Uncertainty robustness sample %s → %s", label, res.bkpts)
        except Exception:
            logger.exception("Uncertainty robustness sample %s failed", label)
    return samples


def _collect_bootstrap_samples(
    df: pl.DataFrame,
    params: dict,
    *,
    n_bootstrap: int,
    bootstrap_block: int,
    seed: int = 42,
) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    slim = _slim_params(params)
    samples: list[list[int]] = []
    for i in range(max(0, int(n_bootstrap))):
        try:
            boot = _block_bootstrap(df, bootstrap_block, rng)
            res = single_run.run_single_segmentation(boot, slim)
            samples.append(list(res.bkpts))
            logger.info("Uncertainty bootstrap %s → %s", i, res.bkpts)
        except Exception:
            logger.exception("Uncertainty bootstrap %s failed", i)
    return samples


def _bands_from_samples(
    baseline_bkpts: list[int],
    samples: list[list[int]],
    *,
    level: float,
    tolerance: int,
    length: int,
) -> dict[int, tuple[int, int]]:
    if not baseline_bkpts:
        return {}
    alpha = (1.0 - float(level)) / 2.0
    ci: dict[int, tuple[int, int]] = {}
    for b in baseline_bkpts:
        locs: list[int] = []
        for alt in samples:
            if not alt:
                continue
            mapping = evaluation_tools.match_breakpoints([b], alt)
            matched, dist = mapping.get(b, (None, float("inf")))
            if matched is not None and dist <= tolerance:
                locs.append(int(matched))
            else:
                # nearest alt break as soft evidence
                nearest = min(alt, key=lambda x: abs(int(x) - b))
                if abs(int(nearest) - b) <= 2 * tolerance:
                    locs.append(int(nearest))
        if not locs:
            ci[b] = (max(1, b - tolerance), min(length - 2, b + tolerance))
        else:
            lo = int(np.quantile(locs, alpha))
            hi = int(np.quantile(locs, 1.0 - alpha))
            lo = max(0, min(lo, b))
            hi = min(length - 1, max(hi, b))
            ci[b] = (lo, hi)
    return ci


def evaluate_uncertainty(
    df: pl.DataFrame,
    params: dict,
    baseline: SegmentResults,
) -> SegmentResults:
    """Attach calibrated ``bkpt_ci`` bands from robustness / bootstrap ensembles."""
    cfg = params.get("uncertainty") or {}
    if not cfg.get("enabled", False):
        return baseline

    level = float(cfg.get("level", 0.9))
    tolerance = int(cfg.get("match_tolerance", 5))
    sources = [str(s).lower() for s in (cfg.get("sources") or ["robustness", "bootstrap"])]
    samples: list[list[int]] = []

    if "robustness" in sources:
        samples.extend(_collect_robustness_samples(df, params))
    if "bootstrap" in sources:
        samples.extend(
            _collect_bootstrap_samples(
                df,
                params,
                n_bootstrap=int(cfg.get("n_bootstrap", 8)),
                bootstrap_block=int(cfg.get("bootstrap_block", 20)),
                seed=int(cfg.get("random_state", 42)),
            )
        )

    if not samples:
        logger.warning("Uncertainty enabled but no ensemble samples collected.")
        return baseline

    ci = _bands_from_samples(
        list(baseline.bkpts),
        samples,
        level=level,
        tolerance=tolerance,
        length=df.height,
    )
    # Low confidence: wide bands relative to series length
    width_thresh = max(tolerance * 3, int(0.02 * df.height))
    low = [b for b, (lo, hi) in ci.items() if (hi - lo) > width_thresh]
    # Merge with existing low_confidence from persistence
    low = sorted(set(list(baseline.low_confidence_bkpts or []) + low))

    logger.info("Uncertainty bands (level=%.2f): %s", level, ci)
    return SegmentResults(
        bkpts=list(baseline.bkpts),
        invalid_bkpts=list(baseline.invalid_bkpts),
        stats={
            **dict(baseline.stats or {}),
            "_uncertainty": {
                "level": level,
                "n_samples": len(samples),
                "sources": sources,
            },
        },
        hierarchy=dict(baseline.hierarchy or {}),
        labels=baseline.labels,
        params=baseline.params,
        persistence=dict(baseline.persistence or {}),
        low_confidence_bkpts=low,
        stability_score=baseline.stability_score,
        bkpt_ci=ci,
        panel_support=dict(baseline.panel_support or {}),
    )
