"""Apache Hamilton nodes for loading a gulfstream feature matrix from source YAML."""
from __future__ import annotations

from pathlib import Path

import polars as pl

from gulfstream.data.source_loader import load_features_from_source_yaml


def source_config_path(source_config: str | Path) -> Path:
    return Path(source_config)


def project_root_path(project_root: str | Path | None = None) -> Path | None:
    return Path(project_root) if project_root is not None else None


def load_bundle(
    source_config_path: Path, project_root_path: Path | None
) -> tuple[pl.DataFrame, str]:
    """Single load so Hamilton does not hit the source twice."""
    return load_features_from_source_yaml(
        source_config_path, project_root=project_root_path
    )


def features_df(load_bundle: tuple[pl.DataFrame, str]) -> pl.DataFrame:
    return load_bundle[0]


def source_type(load_bundle: tuple[pl.DataFrame, str]) -> str:
    return load_bundle[1]
