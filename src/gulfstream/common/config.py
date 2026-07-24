"""Typed configuration models for gulfstream pipelines.

YAML still uses the same keys; pydantic validates them and coerces bare
scalars into one-element lists where the grid semantics require lists.
Pipeline internals continue to consume plain dicts via ``Config.to_params()``.
Runtime-only state (Excel writers, image dirs, counters) lives on ``RunContext``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from gulfstream.common.options import (
    DimredMethod,
    FeatureMapApprox,
    GammaMethod,
    HyperparamMethod,
    MetricsMode,
    PostProcessing,
    RankSelection,
    RecursiveMethod,
    RegimeClusterAlgorithm,
    SearchMethod,
    StatTest,
)


def _as_list(v: Any) -> list:
    if v is None:
        return []
    if isinstance(v, list):
        return v
    return [v]


class AlgoConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    dimred: list[DimredMethod | str] = Field(default_factory=lambda: [DimredMethod.PCA])
    rank_selection_method: list[RankSelection | str] = Field(
        default_factory=lambda: [RankSelection.EXPLAINED_VARIANCE]
    )
    threshold: list[float] = Field(default_factory=lambda: [0.9])
    rank: list[int] = Field(default_factory=list)
    recursive_method: list[RecursiveMethod | str] = Field(
        default_factory=lambda: [RecursiveMethod.FULL]
    )
    depth: list[int] = Field(default_factory=lambda: [1])
    search_method: list[SearchMethod | str] = Field(
        default_factory=lambda: [SearchMethod.PELT]
    )
    feature_map_approx_method: list[FeatureMapApprox | str] = Field(
        default_factory=lambda: [FeatureMapApprox.RFF]
    )
    feature_map_kernel_params: list[dict[str, Any]] = Field(default_factory=list)
    kpca_kernel_params: list[dict[str, Any]] = Field(default_factory=list)
    num_features: list[int] = Field(default_factory=lambda: [50])
    num_mappings: list[int] = Field(default_factory=lambda: [1])
    ruptures_kernel_params: list[dict[str, Any]] = Field(default_factory=list)
    post_processing_method: list[PostProcessing | str] = Field(
        default_factory=lambda: [PostProcessing.MAJORITY_VOTING]
    )
    min_regime_length: list[int] = Field(default_factory=lambda: [20])
    include_last_regime: list[bool] = Field(default_factory=lambda: [True])
    entropy_window: list[int] = Field(default_factory=list)
    regimes: list[int] = Field(default_factory=list)
    dmd_stride: list[int] = Field(default_factory=list)
    dmd_rolling_window: list[int] = Field(default_factory=list)
    tsne_perplexity: list[float] = Field(default_factory=list)
    tsne_n_iter: list[int] = Field(default_factory=list)
    umap_num_neighbors: list[float] = Field(default_factory=list)
    umap_min_dist: list[float] = Field(default_factory=list)
    umap_metric: list[str] = Field(default_factory=list)

    @field_validator(
        "dimred",
        "rank_selection_method",
        "threshold",
        "rank",
        "recursive_method",
        "depth",
        "search_method",
        "feature_map_approx_method",
        "num_features",
        "num_mappings",
        "post_processing_method",
        "min_regime_length",
        "include_last_regime",
        "entropy_window",
        "regimes",
        "dmd_stride",
        "dmd_rolling_window",
        "tsne_perplexity",
        "tsne_n_iter",
        "umap_num_neighbors",
        "umap_min_dist",
        "umap_metric",
        mode="before",
    )
    @classmethod
    def coerce_list(cls, v: Any) -> list:
        return _as_list(v)


class TestConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    choice: list[StatTest | str] = Field(default_factory=lambda: [StatTest.MMD_NO_TS])
    lag: list[dict[str, Any]] = Field(default_factory=list)
    window: list[dict[str, Any]] = Field(default_factory=list)
    sample_size: list[dict[str, Any]] = Field(default_factory=list)
    significance_level: list[float] = Field(default_factory=lambda: [0.05])

    @field_validator("choice", "significance_level", mode="before")
    @classmethod
    def coerce_list(cls, v: Any) -> list:
        return _as_list(v)

    @field_validator("lag", "window", "sample_size", mode="before")
    @classmethod
    def coerce_param_blocks(cls, v: Any) -> list:
        if v is None:
            return []
        if isinstance(v, dict):
            return [v]
        return list(v)


class MetricsConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    mode: MetricsMode | str = MetricsMode.WRITE
    dir: str | None = None
    plot: bool = True
    num_samples: int = 100
    num_features: int = 5
    num_components: int = 3
    num_shap: int = 5
    num_top_features_for_distances: int = 5
    requested_features_for_distances: list[str] = Field(default_factory=list)
    explainability_features: list[str] | dict[str, Any] | str = Field(
        default_factory=lambda: ["__auto__"]
    )
    features_to_plot: list[str] | str = Field(default_factory=lambda: ["__auto__"])
    warn_threshold: float = 0.5
    exp_tree_accuracy: list[float] = Field(default_factory=lambda: [0.9])
    exp_tree_bps_decimals: bool = False
    developer_mode: bool = False
    regime_cluster_algorithms: list[RegimeClusterAlgorithm | str] = Field(
        default_factory=lambda: [RegimeClusterAlgorithm.KMEANS]
    )
    image_dir: str | None = None

    @field_validator("exp_tree_accuracy", "regime_cluster_algorithms", mode="before")
    @classmethod
    def coerce_list(cls, v: Any) -> list:
        return _as_list(v)

    @field_validator("features_to_plot", "explainability_features", mode="before")
    @classmethod
    def coerce_features(cls, v: Any) -> Any:
        if v is None:
            return ["__auto__"]
        if isinstance(v, str):
            return [v]
        return v


class LogConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    dir: str | None = None
    level: str = "INFO"


class RobustnessConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    enabled: bool = False
    perturbations: dict[str, Any] = Field(default_factory=dict)
    match_tolerance: int = 5
    low_persistence_threshold: float = 0.5


class StabilityConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    enabled: bool = False
    leave_one_source_out: bool = True
    leave_one_tenor_out: bool = True
    time_windows: dict[str, Any] = Field(default_factory=dict)
    match_tolerance: int = 5
    stability_floor: float = 0.5


class RetrainConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    interactive: bool = False
    features: list[str] | dict[str, Any] | str = Field(
        default_factory=lambda: ["__auto__"]
    )
    num_worst_features: int = 5
    threshold: float = 0.01
    max_iter: int = 10
    regimes_df: Any = None


class Config(BaseModel):
    """Top-level algo YAML configuration."""

    model_config = ConfigDict(extra="allow")

    algo: AlgoConfig = Field(default_factory=AlgoConfig)
    test: TestConfig = Field(default_factory=TestConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    log: LogConfig = Field(default_factory=LogConfig)
    robustness: RobustnessConfig = Field(default_factory=RobustnessConfig)
    stability: StabilityConfig = Field(default_factory=StabilityConfig)
    retrain: RetrainConfig | None = None

    # Legacy detector key (optional)
    regime_detection_algorithm: list[str] | None = None

    @model_validator(mode="before")
    @classmethod
    def ensure_sections(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data.setdefault("algo", {})
        data.setdefault("test", {})
        data.setdefault("metrics", {})
        data.setdefault("log", {})
        data.setdefault("robustness", {"enabled": False})
        data.setdefault("stability", {"enabled": False})
        return data

    def to_params(self) -> dict[str, Any]:
        """Plain dict for pipeline internals (string enums → strings)."""
        return self.model_dump(mode="json", exclude_none=False)


@dataclass
class RunContext:
    """Runtime state injected during a pipeline run (not part of YAML)."""

    test_num: int = 0
    row: int = 0
    image_dir: str | None = None
    results_writer: Any = None
    template: Any = None
    data_params: dict[str, Any] = field(default_factory=dict)
    pipeline_params: dict[str, Any] | None = None

    def into_misc(self, metrics: dict[str, Any]) -> dict[str, Any]:
        """Shape expected by Graph 1's misc_params / case_params merge."""
        return {
            "test_num": self.test_num,
            "row": self.row,
            "results_writer": self.results_writer,
            "image_dir": self.image_dir,
            "template": self.template,
            "metrics": metrics,
            "data_params": self.data_params,
        }


def coerce_params(config: Config | dict[str, Any]) -> dict[str, Any]:
    """Accept Config or dict; always return a validated params dict."""
    if isinstance(config, Config):
        return config.to_params()
    return Config.model_validate(config).to_params()


def load_config(
    path: str | Path,
    *,
    img_dir: str,
    log_dir: str,
) -> Config:
    """Read YAML, substitute path placeholders, validate into ``Config``."""
    import yaml

    path = Path(path)
    text = path.read_text(encoding="utf-8")
    text = text.replace("${LOG_DIR}", log_dir).replace("${IMG_DIR}", img_dir)
    raw = yaml.safe_load(text) or {}
    return Config.model_validate(raw)


# Silence unused-import noise for option types documented in field defaults.
_ = (GammaMethod, HyperparamMethod, SearchMethod)
