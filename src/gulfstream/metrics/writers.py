"""Minimal results writers (display/write stubs for core path)."""
from __future__ import annotations

import logging

import polars as pl

from gulfstream.common.results import SegmentResults

logger = logging.getLogger(__name__)


def _write_results_header(params: dict, results_writer) -> dict:
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


def _report_regime_statistics(
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


def _report_performance(
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
