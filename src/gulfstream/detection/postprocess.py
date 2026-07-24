"""Post-processing to filter oversegmentation."""
from __future__ import annotations

import itertools
import logging
from typing import Iterator

import numpy as np
import polars as pl

from gulfstream.common import utils
from gulfstream.common.options import PostProcessing
from gulfstream.common.results import AlgoResults, SegmentResults
from gulfstream.detection.time_index import convert_results

logger = logging.getLogger(__name__)


def combine_params(mapping_params: list[dict]) -> dict:
    """Merge per-PC mapping param dicts; varying values become tuples."""
    if not mapping_params:
        return {}
    if len(mapping_params) == 1:
        return dict(mapping_params[0])
    keys = set().union(*(m.keys() for m in mapping_params))
    out = {}
    for key in keys:
        vals = [m.get(key) for m in mapping_params]
        if all(v == vals[0] for v in vals):
            out[key] = vals[0]
        else:
            out[key] = tuple(vals)
    return out


def combine_results(length: int, results: list[SegmentResults]) -> SegmentResults:
    """Merge per-PC results (iterative_pca) or return the single full result."""
    if not results:
        return SegmentResults(bkpts=[], labels=[0] * length)
    if len(results) == 1:
        return results[0]

    bkpts = sorted(set().union(*(r.bkpts for r in results)))
    invalid = sorted(set().union(*(r.invalid_bkpts for r in results)) - set(bkpts))
    stats = {}
    hierarchy = {}
    for r in results:
        stats.update(r.stats)
        hierarchy.update(r.hierarchy)
    labels = utils.convert_bkpts_to_labels(bkpts, length)
    return SegmentResults(
        bkpts=bkpts,
        invalid_bkpts=invalid,
        stats=stats,
        hierarchy=hierarchy,
        labels=labels,
        params=results[0].params,
    )


def post_processing_params_generator(test_choice: str, params: dict) -> Iterator[dict]:
    methods = params["algo"].get(
        "post_processing_method", [PostProcessing.NO_POST_PROCESSING]
    )
    min_lens = params["algo"].get("min_regime_length", [1])
    include_last = params["algo"].get("include_last_regime", [True])
    entropy_windows = params["algo"].get("entropy_window", [10])

    for method in methods:
        if method in (PostProcessing.MAJORITY_VOTING, PostProcessing.ENTROPY):
            for ml, il in itertools.product(min_lens, include_last):
                out = {
                    "post_processing_method": method,
                    "min_regime_length": ml,
                    "include_last_regime": il,
                }
                if method == PostProcessing.ENTROPY:
                    for ew in entropy_windows:
                        yield {**out, "entropy_window": ew}
                else:
                    yield out
        elif method == PostProcessing.NEIGHBOR_COMPARISON:
            for ml in min_lens:
                yield {"post_processing_method": method, "min_regime_length": ml}
        else:
            yield {"post_processing_method": method}


def _majority_voting(
    res: SegmentResults,
    *,
    length: int,
    min_regime_length: int,
    include_last_regime: bool,
) -> SegmentResults:
    """Drop breakpoints that create regimes shorter than min_regime_length."""
    bkpts = sorted(b for b in res.bkpts if 0 < b < length - 1)
    if not bkpts:
        labels = [0] * length
        return SegmentResults(
            bkpts=[],
            invalid_bkpts=list(res.invalid_bkpts),
            stats=dict(res.stats),
            hierarchy={},
            labels=labels,
            params=res.params,
        )

    kept = []
    edges = [0] + bkpts + [length]
    for i, b in enumerate(bkpts):
        left = edges[i + 1] - edges[i]
        right = edges[i + 2] - edges[i + 1]
        if left >= min_regime_length and right >= min_regime_length:
            kept.append(b)
        else:
            logger.debug(
                "Dropping bkpt %s due to short regime (left=%s right=%s min=%s)",
                b,
                left,
                right,
                min_regime_length,
            )

    if not include_last_regime and kept:
        # Optionally drop the last breakpoint (merge last two regimes).
        pass  # keep all kept; flag reserved for parity with original API

    labels = utils.convert_bkpts_to_labels(kept, length)
    hierarchy = {b: res.hierarchy.get(b, 1) for b in kept}
    stats = {b: res.stats[b] for b in kept if b in res.stats}
    return SegmentResults(
        bkpts=kept,
        invalid_bkpts=list(res.invalid_bkpts),
        stats=stats,
        hierarchy=hierarchy,
        labels=labels,
        params=res.params,
    )


def post_process(
    *,
    res: SegmentResults,
    processing_params: dict,
    params: dict,
    length: int,
    df_dimred: pl.DataFrame,
    mapped_dfs: list[pl.DataFrame] | None = None,
    mapped_df_diffs=None,
    df_pca: pl.DataFrame | None = None,
) -> SegmentResults:
    """Post-process breakpoints; return results in original-index units."""
    method = processing_params.get(
        "post_processing_method", PostProcessing.NO_POST_PROCESSING
    )
    # Work in dimred units first, then convert to original length.
    if method == PostProcessing.NO_POST_PROCESSING:
        processed = SegmentResults(
            bkpts=list(res.bkpts),
            invalid_bkpts=list(res.invalid_bkpts),
            stats=dict(res.stats),
            hierarchy=dict(res.hierarchy),
            labels=res.labels,
            params=res.params,
        )
    elif method == PostProcessing.MAJORITY_VOTING:
        processed = _majority_voting(
            res,
            length=df_dimred.height,
            min_regime_length=int(processing_params.get("min_regime_length", 1)),
            include_last_regime=bool(processing_params.get("include_last_regime", True)),
        )
    else:
        logger.warning("Post-processing method %s not fully implemented; passing through.", method)
        processed = res

    return convert_results(processed, length)


def classical_post_processing_params_generator(params: dict) -> Iterator[dict]:
    """Yield post-processing param dicts for classical (hard-label) methods."""
    methods = params["algo"].get(
        "post_processing_method", [PostProcessing.NO_POST_PROCESSING]
    )
    min_lens = params["algo"].get("min_regime_length", [1])
    include_last = params["algo"].get("include_last_regime", [True])

    for method in methods:
        # Accept YAML alias "no_post_processing" as nopost_processing
        method_s = str(method)
        if method_s in ("no_post_processing", "nopost_processing"):
            method = PostProcessing.NO_POST_PROCESSING
        if method == PostProcessing.MAJORITY_VOTING or method_s == "majority_voting":
            for ml, il in itertools.product(min_lens, include_last):
                yield {
                    "post_processing_method": PostProcessing.MAJORITY_VOTING,
                    "min_regime_length": ml,
                    "include_last_regime": il,
                }
        elif method == PostProcessing.NEIGHBOR_COMPARISON or method_s == "neighbor_comparison":
            for ml in min_lens:
                yield {
                    "post_processing_method": PostProcessing.NEIGHBOR_COMPARISON,
                    "min_regime_length": ml,
                }
        else:
            yield {"post_processing_method": method}


def _classical_majority_voting(
    res: AlgoResults,
    *,
    length: int,
    min_regime_length: int,
    include_last_regime: bool = True,
) -> AlgoResults:
    """Drop breakpoints that create regimes shorter than min_regime_length."""
    from gulfstream.common.results import AlgoResults as AR

    bkpts = sorted(b for b in res.bkpts if 0 < b < length - 1)
    if not bkpts:
        return AR(bkpts=[], labels=[0] * length, params=res.params)

    kept: list[int] = []
    edges = [0] + bkpts + [length]
    for i, b in enumerate(bkpts):
        left = edges[i + 1] - edges[i]
        right = edges[i + 2] - edges[i + 1]
        if left >= min_regime_length and right >= min_regime_length:
            kept.append(b)

    if not include_last_regime and kept:
        pass

    labels = utils.convert_bkpts_to_labels(kept, length)
    return AR(bkpts=kept, labels=labels, params=res.params)


def classical_post_process(
    *,
    res,
    processing_params: dict,
    params: dict,
    length: int,
    df_dimred: pl.DataFrame,
) -> SegmentResults:
    """Post-process classical AlgoResults and remap to SegmentResults."""
    from gulfstream.common.results import AlgoResults as AR

    method = processing_params.get(
        "post_processing_method", PostProcessing.NO_POST_PROCESSING
    )
    method_s = str(method)
    if method_s in ("no_post_processing", "nopost_processing"):
        method = PostProcessing.NO_POST_PROCESSING
    work_len = df_dimred.height
    if method == PostProcessing.NO_POST_PROCESSING:
        processed = AR(
            bkpts=list(res.bkpts),
            labels=list(res.labels) if res.labels is not None else None,
            params=res.params,
        )
    elif method == PostProcessing.MAJORITY_VOTING:
        processed = _classical_majority_voting(
            res,
            length=work_len,
            min_regime_length=int(processing_params.get("min_regime_length", 1)),
            include_last_regime=bool(processing_params.get("include_last_regime", True)),
        )
    else:
        logger.warning(
            "Classical post-processing method %s not fully implemented; passing through.",
            method,
        )
        processed = res

    converted = convert_results(processed, length)
    return algo_results_to_segment(converted)


def algo_results_to_segment(res: AlgoResults | SegmentResults) -> SegmentResults:
    """Normalize classical AlgoResults into SegmentResults."""
    if isinstance(res, SegmentResults):
        hierarchy = res.hierarchy or {b: 1 for b in res.bkpts}
        return SegmentResults(
            bkpts=list(res.bkpts),
            invalid_bkpts=list(res.invalid_bkpts or []),
            stats=dict(res.stats or {}),
            hierarchy=hierarchy,
            labels=list(res.labels) if res.labels is not None else None,
            params=res.params,
            persistence=dict(res.persistence or {}),
            low_confidence_bkpts=list(res.low_confidence_bkpts or []),
            stability_score=res.stability_score,
        )
    hierarchy = {b: 1 for b in res.bkpts}
    return SegmentResults(
        bkpts=list(res.bkpts),
        invalid_bkpts=[],
        stats={},
        hierarchy=hierarchy,
        labels=list(res.labels) if res.labels is not None else None,
        params=res.params,
    )


# Back-compat aliases
legacy_post_processing_params_generator = classical_post_processing_params_generator
legacy_post_process = classical_post_process
