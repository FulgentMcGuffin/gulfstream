"""Minimal results writers (display/write stubs for core path)."""
from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

import polars as pl

from gulfstream.common.results import SegmentResults

logger = logging.getLogger(__name__)


def write_results_header(params: dict, results_writer) -> dict:
    """Write a Parameter sheet header; return a blank template row.

    ``results_writer`` is a pandas ``ExcelWriter`` (Excel boundary).
    """
    import pandas as pd

    template = {
        "test_num": None,
        "choice": None,
        "dimred": None,
        "depth": None,
        "n_bkpts": None,
        "n_invalid": None,
    }
    pd.DataFrame([template]).to_excel(results_writer, sheet_name="Parameters", index=False)
    return template


def report_regime_statistics(
    df: pl.DataFrame,
    params: dict,
    proc_res: SegmentResults,
    unproc_res: SegmentResults,
) -> None:
    logger.info(
        "Regime stats: processed bkpts=%s invalid=%s (raw bkpts=%s)",
        proc_res.bkpts,
        proc_res.invalid_bkpts,
        unproc_res.bkpts,
    )


def report_performance(
    df: pl.DataFrame,
    params: dict,
    proc_res: SegmentResults,
    unproc_res: SegmentResults,
) -> int:
    import pandas as pd

    row = int(params.get("row", 0))
    writer = params.get("results_writer")
    if writer is None:
        return row + 1
    template = params.get("template") or {}
    new_row = dict(template)
    new_row.update(
        {
            "test_num": params.get("test_num"),
            "choice": params.get("test", {}).get("choice"),
            "dimred": params.get("algo", {}).get("dimred"),
            "depth": params.get("algo", {}).get("depth"),
            "n_bkpts": len(proc_res.bkpts),
            "n_invalid": len(proc_res.invalid_bkpts),
        }
    )
    try:
        sheet = "Results"
        existing = None
        if sheet in writer.book.sheetnames:
            existing = pd.read_excel(writer, sheet_name=sheet)
        out = (
            pd.concat([existing, pd.DataFrame([new_row])], ignore_index=True)
            if existing is not None
            else pd.DataFrame([new_row])
        )
        out.to_excel(writer, sheet_name=sheet, index=False)
    except Exception:
        logger.exception("Failed to write performance row")
    return row + 1


def resolve_excel_export_path(
    params: dict,
    *,
    default_dir: str | None = None,
) -> str | None:
    """Resolve workbook path from ``export.excel`` (path / dir / filename)."""
    export = params.get("export") or {}
    cfg = export.get("excel") if isinstance(export.get("excel"), dict) else {}
    if not cfg and isinstance(export, dict) and export.get("enabled") is not None:
        # Flat ``export.enabled`` / ``export.path`` shape
        cfg = export
    if not isinstance(cfg, dict) or not cfg.get("enabled", False):
        return None
    if cfg.get("path"):
        return str(cfg["path"])

    filename = cfg.get("filename") or "bkpt_export.xlsx"
    if not str(filename).lower().endswith((".xlsx", ".xlsm", ".xls")):
        filename = f"{filename}.xlsx"
    base = (
        cfg.get("dir")
        or default_dir
        or (params.get("metrics") or {}).get("image_dir")
        or (params.get("metrics") or {}).get("dir")
        or "."
    )
    return str(Path(base) / filename)


def export_breakpoint_excel(
    res: SegmentResults,
    params: dict,
    *,
    default_dir: str | None = None,
    dates: list | None = None,
) -> str | None:
    """Write Breakpoints / CI / PanelSupport sheets; return path or None."""
    import pandas as pd

    path = resolve_excel_export_path(params, default_dir=default_dir)
    if path is None:
        return None

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ci = {int(k): v for k, v in (res.bkpt_ci or {}).items()}
    support = {int(k): v for k, v in (res.panel_support or {}).items()}
    low = set(res.low_confidence_bkpts or [])
    hierarchy = res.hierarchy or {}

    bkpt_rows = []
    for b in res.bkpts:
        lo_hi = ci.get(int(b))
        date_str = None
        if dates is not None and 0 <= int(b) < len(dates):
            d = dates[int(b)]
            date_str = d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)
        bkpt_rows.append(
            {
                "bkpt": int(b),
                "date": date_str,
                "hierarchy": hierarchy.get(b),
                "low_confidence": int(b) in low,
                "ci_lo": None if lo_hi is None else int(lo_hi[0]),
                "ci_hi": None if lo_hi is None else int(lo_hi[1]),
                "panel_support": support.get(int(b)),
                "persistence": (res.persistence or {}).get(b),
            }
        )

    ci_rows = [
        {"bkpt": b, "ci_lo": int(lo), "ci_hi": int(hi), "width": int(hi) - int(lo)}
        for b, (lo, hi) in sorted(ci.items())
    ]
    support_rows = [
        {"bkpt": b, "panel_support": float(s)} for b, s in sorted(support.items())
    ]
    meta = pd.DataFrame(
        [
            {
                "exported_at": datetime.now().isoformat(timespec="seconds"),
                "n_bkpts": len(res.bkpts),
                "n_invalid": len(res.invalid_bkpts or []),
                "stability_score": res.stability_score,
                "test_num": params.get("test_num"),
                "dimred": str((params.get("algo") or {}).get("dimred")),
                "search_method": str((params.get("algo") or {}).get("search_method")),
            }
        ]
    )

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame(bkpt_rows).to_excel(writer, sheet_name="Breakpoints", index=False)
        pd.DataFrame(ci_rows).to_excel(writer, sheet_name="CI", index=False)
        pd.DataFrame(support_rows).to_excel(writer, sheet_name="PanelSupport", index=False)
        meta.to_excel(writer, sheet_name="Meta", index=False)

    logger.info("Exported breakpoint workbook to %s", path)
    return path
