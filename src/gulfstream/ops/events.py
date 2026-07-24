"""Dashboard-friendly NDJSON event stream for gulfstream runs."""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gulfstream.common.results import SegmentResults

logger = logging.getLogger(__name__)


def resolve_events_path(params: dict, *, default_dir: str | None = None) -> str | None:
    """Resolve NDJSON path from ``events`` config (path / dir / filename)."""
    cfg = params.get("events") or {}
    if not cfg.get("enabled", False):
        return None
    if cfg.get("path"):
        return str(cfg["path"])

    filename = cfg.get("filename") or "events.ndjson"
    base = (
        cfg.get("dir")
        or default_dir
        or (params.get("metrics") or {}).get("image_dir")
        or (params.get("metrics") or {}).get("dir")
        or "."
    )
    return str(Path(base) / filename)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_records(path: str, records: list[dict[str, Any]], *, append: bool) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    mode = "a" if append else "w"
    with open(path, mode, encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, default=str) + "\n")


def emit_event(
    params: dict,
    event: str,
    payload: dict[str, Any] | None = None,
    *,
    default_dir: str | None = None,
) -> str | None:
    """Append one NDJSON event; return the path written (or None if disabled)."""
    path = resolve_events_path(params, default_dir=default_dir)
    if path is None:
        return None
    cfg = params.get("events") or {}
    record = {
        "ts": _utc_now(),
        "event": event,
        **(payload or {}),
    }
    _write_records(path, [record], append=bool(cfg.get("append", True)))
    logger.debug("Event %s → %s", event, path)
    return path


def emit_run_events(
    params: dict,
    res: SegmentResults,
    *,
    default_dir: str | None = None,
    run_id: str | None = None,
) -> str | None:
    """Emit breakpoint + run_complete events for a finished segmentation.

    When ``events.append`` is false, the file is replaced with this run's events.
    When true, events are appended (one NDJSON object per line).
    """
    path = resolve_events_path(params, default_dir=default_dir)
    if path is None:
        return None

    cfg = params.get("events") or {}
    append = bool(cfg.get("append", True))
    rid = run_id or f"test_{params.get('test_num', 0)}"
    ci = {int(k): list(v) for k, v in (res.bkpt_ci or {}).items()}
    support = {int(k): v for k, v in (res.panel_support or {}).items()}
    low = set(res.low_confidence_bkpts or [])
    ts = _utc_now()

    records: list[dict[str, Any]] = [
        {
            "ts": ts,
            "event": "run_started",
            "run_id": rid,
            "detection_backend": (params.get("algo") or {}).get("detection_backend"),
            "dimred": (params.get("algo") or {}).get("dimred"),
            "search_method": (params.get("algo") or {}).get("search_method"),
        }
    ]
    for b in res.bkpts:
        records.append(
            {
                "ts": ts,
                "event": "breakpoint_confirmed",
                "run_id": rid,
                "bkpt": int(b),
                "ci": ci.get(int(b)),
                "panel_support": support.get(int(b)),
                "low_confidence": int(b) in low,
                "hierarchy": (res.hierarchy or {}).get(b),
                "stat": (res.stats or {}).get(b),
            }
        )
    for b in res.invalid_bkpts or []:
        records.append(
            {
                "ts": ts,
                "event": "breakpoint_rejected",
                "run_id": rid,
                "bkpt": int(b),
                "stat": (res.stats or {}).get(b),
            }
        )
    records.append(
        {
            "ts": ts,
            "event": "run_complete",
            "run_id": rid,
            "n_bkpts": len(res.bkpts),
            "n_invalid": len(res.invalid_bkpts or []),
            "n_low_confidence": len(low),
            "bkpts": list(res.bkpts),
            "bkpt_ci": ci,
            "panel_support": support,
            "stability_score": res.stability_score,
        }
    )
    _write_records(path, records, append=append)
    logger.info("Wrote event stream (%d events) to %s", len(records), path)
    return path
