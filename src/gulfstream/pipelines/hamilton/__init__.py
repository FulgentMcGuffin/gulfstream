"""Apache Hamilton dataflows used by gulfstream pipelines.

- ``segmentation`` — one Graph 1/2 pass (dimred → map → ruptures → postprocess)
- ``load_features`` — source YAML → dated feature frame
- ``driver`` — execute / optionally visualize those DAGs

Docs: https://hamilton.apache.org/
"""
from __future__ import annotations

from gulfstream.pipelines.hamilton.driver import (
    load_features,
    run_segmentation_pair,
    visualize_segmentation_dag,
)

__all__ = [
    "load_features",
    "run_segmentation_pair",
    "visualize_segmentation_dag",
]
