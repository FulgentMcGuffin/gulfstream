"""Breakpoint hierarchy tree visualization.

NetworkX graph layout stays on matplotlib; statistical gallery plots use plotnine.
"""
from __future__ import annotations

import logging
import os
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import polars as pl

from gulfstream.detection import time_index as bkpt_timeindexing_conversions
from gulfstream.common import utils
from gulfstream.common.results import SegmentResults

logger = logging.getLogger(__name__)


def attach_retrain_tree(
    bkpt_index_dict: dict,
    hierarchy: dict,
    start: int,
    end: int,
) -> dict:
    """Merge a **local** (slice-relative) hierarchy into the global tree.

    Offsets local keys by ``start`` and keeps only breakpoints strictly inside
    ``[start, end)``. Callers must pass unshifted hierarchy from the slice run.
    """
    out = dict(bkpt_index_dict)
    for bkpt, level in hierarchy.items():
        global_bkpt = int(bkpt) + int(start)
        if start < global_bkpt < end:
            out[global_bkpt] = int(level)
    return out


def _hierarchy_graph(hierarchy: dict[int, Any], length: int) -> nx.DiGraph:
    """Build a simple parent→child digraph from hierarchy levels."""
    g = nx.DiGraph()
    root = "root"
    g.add_node(root, label=f"[0,{length})", level=0)
    bkpts = sorted(int(b) for b in hierarchy.keys() if 0 < int(b) < length)
    # Nodes are intervals; edges connect nested splits by level order.
    intervals = [(0, length, root)]
    for level in sorted(set(int(hierarchy[b]) for b in bkpts) or [1]):
        level_bkpts = [b for b in bkpts if int(hierarchy[b]) == level]
        new_intervals = []
        for a, b, parent in intervals:
            splits = [s for s in level_bkpts if a < s < b]
            if not splits:
                new_intervals.append((a, b, parent))
                continue
            edges = [a] + splits + [b]
            for i in range(len(edges) - 1):
                left, right = edges[i], edges[i + 1]
                node = f"{left}:{right}"
                g.add_node(node, label=f"[{left},{right})", level=level)
                g.add_edge(parent, node)
                new_intervals.append((left, right, node))
            for s in splits:
                g.add_node(f"bkpt:{s}", label=str(s), level=level, bkpt=s)
        intervals = new_intervals
    return g


def _plot_tree(
    hierarchy: dict,
    length: int,
    dates: list[str],
    *,
    title: str,
    gallery_key: str,
    img_dir: str | None,
    draw: bool,
    save: bool,
) -> None:
    g = _hierarchy_graph(hierarchy or {}, length)
    if g.number_of_nodes() == 0:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    try:
        pos = nx.nx_agraph.graphviz_layout(g, prog="dot")  # type: ignore[attr-defined]
    except Exception:
        pos = nx.spring_layout(g, seed=0)
    labels = {n: g.nodes[n].get("label", str(n)) for n in g.nodes}
    nx.draw_networkx(
        g,
        pos=pos,
        ax=ax,
        with_labels=True,
        labels=labels,
        node_size=800,
        font_size=7,
        arrows=True,
    )
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    if save and img_dir:
        os.makedirs(img_dir, exist_ok=True)
        path = os.path.join(img_dir, utils.img_gallery_filename(gallery_key).lstrip("/"))
        fig.savefig(path, dpi=120)
        logger.info("Saved breakpoint tree to %s", path)
    if draw:
        try:
            plt.show()
        except Exception:
            pass
    plt.close(fig)

    # Also dump a text summary of hierarchy with dates.
    if save and img_dir:
        lines = []
        for b in sorted(hierarchy.keys()):
            date = dates[b] if 0 <= b < len(dates) else str(b)
            lines.append(f"{b}\t{date}\tlevel={hierarchy[b]}")
        with open(os.path.join(img_dir, "bkpt_hierarchy.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines))


def build_and_plot_bkpt_trees(
    df: pl.DataFrame,
    params: dict,
    proc_res: SegmentResults,
    unproc_res: SegmentResults | None = None,
    *,
    draw: bool = False,
    save: bool = True,
    file_dir: str | None = None,
) -> None:
    """Plot processed (and optional unprocessed) breakpoint hierarchy trees."""
    metrics = params.get("metrics", {})
    img_dir = file_dir or metrics.get("image_dir") or metrics.get("dir")
    mode = metrics.get("mode", "write")
    draw = draw or mode in ("display", "display_and_write")
    save = save or mode in ("write", "display_and_write")
    dates = bkpt_timeindexing_conversions.get_strs_from_df_index(df)
    hierarchy = proc_res.hierarchy or {b: 1 for b in proc_res.bkpts}
    _plot_tree(
        hierarchy,
        df.height,
        dates,
        title="Processed breakpoint hierarchy",
        gallery_key="bkpt_tree",
        img_dir=img_dir,
        draw=draw,
        save=save,
    )
    if unproc_res is not None:
        unproc_h = unproc_res.hierarchy or {b: 1 for b in unproc_res.bkpts}
        _plot_tree(
            unproc_h,
            df.height,
            dates,
            title="Unprocessed breakpoint hierarchy",
            gallery_key="unproc_tree",
            img_dir=img_dir,
            draw=draw,
            save=save,
        )
