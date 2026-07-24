"""Incremental / streaming Graph 1 (expanding or rolling windows)."""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field

import polars as pl

from gulfstream.common import frames, utils
from gulfstream.common.results import SegmentResults
from gulfstream.pipelines import single_pass as single_run

logger = logging.getLogger(__name__)


@dataclass
class IncrementalState:
    """Warm state for :func:`advance_incremental` / streaming Graph 1."""

    confirmed_bkpts: list[int] = field(default_factory=list)
    last_t: int = 0
    snapshots: list[dict] = field(default_factory=list)


def _slim_params(params: dict) -> dict:
    out = copy.deepcopy(params)
    metrics = out.get("metrics") or {}
    out["metrics"] = {
        **metrics,
        "mode": metrics.get("mode", "write"),
        "plot": False,
    }
    out["robustness"] = {"enabled": False}
    out["stability"] = {"enabled": False}
    out["streaming"] = {"enabled": False}
    out["panel"] = {"enabled": False}
    out["uncertainty"] = {"enabled": False}
    return out


def _merge_bkpts(
    locked: list[int],
    proposed: list[int],
    *,
    tolerance: int,
) -> list[int]:
    """Union locked + proposed, collapsing near-duplicates within ``tolerance``."""
    merged = sorted(set(int(b) for b in locked) | set(int(b) for b in proposed))
    if not merged:
        return []
    out = [merged[0]]
    for b in merged[1:]:
        if b - out[-1] <= tolerance:
            # Prefer the newer proposal when colliding near the frontier
            out[-1] = b
        else:
            out.append(b)
    return out


def advance_incremental(
    df: pl.DataFrame,
    params: dict,
    state: IncrementalState | None = None,
) -> tuple[SegmentResults, IncrementalState]:
    """Advance streaming Graph 1 by one ``streaming.step``.

    Expanding mode segments ``df[0:t_end]``; rolling mode segments a trailing
    window of length ``streaming.window``. Confirmed breakpoints left of the
    frontier can be locked via ``streaming.lock_prefix``.
    """
    cfg = params.get("streaming") or {}
    step = max(1, int(cfg.get("step", 50)))
    min_history = max(step, int(cfg.get("min_history", 150)))
    mode = str(cfg.get("mode", "expanding")).lower()
    window = max(min_history, int(cfg.get("window", 250)))
    lock_prefix = bool(cfg.get("lock_prefix", True))
    tolerance = int(cfg.get("match_tolerance", 5))

    n = df.height
    state = state or IncrementalState()
    if state.last_t <= 0:
        t_end = min(n, min_history)
    else:
        t_end = min(n, state.last_t + step)
    t_end = max(t_end, min(n, min_history))

    slim = _slim_params(params)
    if mode == "rolling":
        t0 = max(0, t_end - window)
        slice_df = frames.slice_rows(df, t0, t_end)
        local = single_run.run_single_segmentation(slice_df, slim)
        proposed = [int(b) + t0 for b in local.bkpts if 0 < int(b) < slice_df.height]
        stats = {int(k) + t0: v for k, v in (local.stats or {}).items()}
        hierarchy = {int(k) + t0: v for k, v in (local.hierarchy or {}).items()}
    else:
        slice_df = frames.slice_rows(df, 0, t_end)
        local = single_run.run_single_segmentation(slice_df, slim)
        proposed = [int(b) for b in local.bkpts if 0 < int(b) < t_end]
        stats = dict(local.stats or {})
        hierarchy = dict(local.hierarchy or {})

    if lock_prefix and state.confirmed_bkpts:
        # Keep locked breaks well behind the frontier; refresh near the edge.
        frontier = max(0, t_end - 2 * step)
        locked = [b for b in state.confirmed_bkpts if b < frontier]
        near = [b for b in proposed if b >= frontier]
        mid = [b for b in proposed if b < frontier]
        # Mid-range: prefer locked if already confirmed nearby
        kept_mid = []
        for b in mid:
            if any(abs(b - L) <= tolerance for L in locked):
                continue
            kept_mid.append(b)
        merged = _merge_bkpts(locked + kept_mid, near, tolerance=tolerance)
    else:
        merged = _merge_bkpts([], proposed, tolerance=tolerance)

    merged = [b for b in merged if 0 < b < n]
    labels = utils.convert_bkpts_to_labels(merged, n)
    res = SegmentResults(
        bkpts=merged,
        invalid_bkpts=[],
        stats={b: stats[b] for b in merged if b in stats},
        hierarchy={b: hierarchy.get(b, 1) for b in merged},
        labels=labels,
        params=params,
    )
    new_state = IncrementalState(
        confirmed_bkpts=list(merged),
        last_t=t_end,
        snapshots=list(state.snapshots)
        + [{"t_end": t_end, "bkpts": list(merged), "mode": mode}],
    )
    logger.info(
        "Streaming step mode=%s t_end=%s/%s bkpts=%s",
        mode,
        t_end,
        n,
        merged,
    )
    return res, new_state


def run_streaming_graph1(df: pl.DataFrame, params: dict) -> SegmentResults:
    """Run expanding/rolling Graph 1 until the series end; return final result."""
    cfg = params.get("streaming") or {}
    if not cfg.get("enabled", False):
        raise ValueError("run_streaming_graph1 requires streaming.enabled=true")

    state: IncrementalState | None = None
    res: SegmentResults | None = None
    n = df.height
    guard = 0
    max_steps = max(2, n)  # safety
    while guard < max_steps:
        res, state = advance_incremental(df, params, state)
        guard += 1
        if state.last_t >= n:
            break
    assert res is not None and state is not None
    res.params = params
    res.stats = {
        **(res.stats or {}),
        "_streaming": {
            "mode": cfg.get("mode", "expanding"),
            "steps": len(state.snapshots),
            "last_t": state.last_t,
        },
    }
    return res
