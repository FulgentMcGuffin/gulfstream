"""Joint breakpoints across a panel of series groups."""
from __future__ import annotations

import copy
import logging
from collections import defaultdict

import numpy as np
import polars as pl

from gulfstream.common import frames, utils
from gulfstream.common.results import SegmentResults
from gulfstream.metrics.stability import _parse_groups
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


def resolve_panel_groups(df: pl.DataFrame, cfg: dict) -> dict[str, list[str]]:
    """Build named feature groups for panel joint segmentation."""
    feat = frames.feature_columns(df)
    explicit = cfg.get("groups") or []
    if explicit:
        return {f"g{i}": list(cols) for i, cols in enumerate(explicit) if cols}

    groupby = str(cfg.get("groupby", "source")).lower()
    by_source, by_tenor = _parse_groups(feat)
    if groupby == "source" and len(by_source) > 1:
        return by_source
    if groupby == "tenor" and len(by_tenor) > 1:
        return by_tenor

    # Fallback for synthetic / non-SOURCE_TENOR panels: split columns in half
    if len(feat) >= 2:
        mid = max(1, len(feat) // 2)
        left, right = feat[:mid], feat[mid:]
        if not right:
            right = feat[-1:]
            left = feat[:-1] or feat[:1]
        return {"panel_a": left, "panel_b": right}
    return {"all": feat}


def _consensus_bkpts(
    group_bkpts: dict[str, list[int]],
    *,
    combine: str,
    min_group_frac: float,
    tolerance: int,
    length: int,
) -> tuple[list[int], dict[int, float]]:
    """Cluster breakpoints across groups within ``tolerance``; apply combine rule."""
    n_groups = max(len(group_bkpts), 1)
    # Flatten all candidates
    all_pts = sorted({int(b) for pts in group_bkpts.values() for b in pts if 0 < int(b) < length})
    if not all_pts:
        return [], {}

    # Greedy cluster
    clusters: list[list[int]] = []
    for b in all_pts:
        if not clusters or b - np.median(clusters[-1]) > tolerance:
            clusters.append([b])
        else:
            clusters[-1].append(b)

    combine = str(combine).lower()
    kept: list[int] = []
    support: dict[int, float] = {}
    for cluster in clusters:
        center = int(round(float(np.median(cluster))))
        # Which groups have a break near the center?
        hit = 0
        for pts in group_bkpts.values():
            if any(abs(int(p) - center) <= tolerance for p in pts):
                hit += 1
        frac = hit / n_groups
        accept = False
        if combine == "union":
            accept = hit >= 1
        elif combine == "intersection":
            accept = hit == n_groups
        else:  # majority
            accept = frac >= float(min_group_frac)
        if accept and 0 < center < length:
            kept.append(center)
            support[center] = float(frac)

    return sorted(set(kept)), support


def run_panel_joint_segmentation(df: pl.DataFrame, params: dict) -> SegmentResults:
    """Segment each panel group, then form a consensus breakpoint set."""
    cfg = params.get("panel") or {}
    if not cfg.get("enabled", False):
        raise ValueError("run_panel_joint_segmentation requires panel.enabled=true")

    groups = resolve_panel_groups(df, cfg)
    if len(groups) < 1:
        raise ValueError("Panel joint segmentation found no feature groups.")

    slim = _slim_params(params)
    group_results: dict[str, SegmentResults] = {}
    group_bkpts: dict[str, list[int]] = {}
    for name, cols in groups.items():
        cols = [c for c in cols if c in frames.feature_columns(df)]
        if len(cols) < 1:
            continue
        sub = frames.select_features(df, cols)
        try:
            res = single_run.run_single_segmentation(sub, slim)
        except Exception:
            logger.exception("Panel group %s failed; skipping.", name)
            continue
        group_results[name] = res
        group_bkpts[name] = list(res.bkpts)
        logger.info("Panel group %s (%d feats) → bkpts=%s", name, len(cols), res.bkpts)

    if not group_bkpts:
        labels = [0] * df.height
        return SegmentResults(bkpts=[], labels=labels, params=params)

    consensus, support = _consensus_bkpts(
        group_bkpts,
        combine=str(cfg.get("combine", "majority")),
        min_group_frac=float(cfg.get("min_group_frac", 0.5)),
        tolerance=int(cfg.get("match_tolerance", 5)),
        length=df.height,
    )
    labels = utils.convert_bkpts_to_labels(consensus, df.height)
    return SegmentResults(
        bkpts=consensus,
        labels=labels,
        hierarchy={b: 1 for b in consensus},
        params=params,
        panel_support=support,
        stats={
            "_panel": {
                "groupby": cfg.get("groupby", "source"),
                "combine": cfg.get("combine", "majority"),
                "n_groups": len(group_bkpts),
                "group_bkpts": {k: list(v) for k, v in group_bkpts.items()},
            }
        },
    )
