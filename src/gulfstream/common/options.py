"""Canonical option enums for gulfstream configuration.

Single source of truth for every string-valued option that appears in YAML
configs. All enums are ``StrEnum`` so existing YAML string values keep working
unchanged, and enum members compare equal to their string values.
"""
from __future__ import annotations

from enum import StrEnum


class DimredMethod(StrEnum):
    """Dimensionality-reduction methods accepted by ``algo.dimred``."""

    PCA = "pca"
    KPCA = "kpca"
    RAW = "raw"
    DMD = "dmd"
    TSNE = "tsne"
    UMAP = "umap"
    ICA = "ica"
    FPCA = "fpca"
    NELSON_SIEGEL = "nelson_siegel"
    DYNAMIC_FACTOR = "dynamic_factor"
    DIEBOLD_LI = "diebold_li"
    SPARSE_PCA = "sparse_pca"
    ROBUST_PCA = "robust_pca"
    AUTOENCODER = "autoencoder"
    BAYESIAN_GMM = "bayesian_gmm"
    HMM = "hmm"
    KMEANS = "kmeans"
    HDBSCAN = "hdbscan"
    OPTICS = "optics"
    MSAR = "msar"
    WASSERSTEIN = "wasserstein"
    RUPTURES = "ruptures"
    TFT = "tft"


class RankSelection(StrEnum):
    """Rank-selection methods for classical dimred (``algo.rank_selection_method``)."""

    ENTROPY = "entropy"
    EXPLAINED_VARIANCE = "explained_variance"
    SVHT = "svht"
    USER_SPECIFIED = "user_specified"


class StatTest(StrEnum):
    """Breakpoint validation tests (``test.choice``)."""

    MMD_NO_TS = "mmd_no_ts"
    MMD_TS = "mmd_ts"
    MMD_PERM = "mmd_perm"
    MMD_UNBIASED = "mmd_unbiased"
    MMD_LINEAR = "mmd_linear"
    ENERGY_DISTANCE = "energy_distance"
    HOTELLING_T2 = "hotelling_t2"
    MULTIVARIATE_CUSUM = "multivariate_cusum"
    KS_PCA = "ks_pca"


class SearchMethod(StrEnum):
    """Candidate breakpoint search algorithms (``algo.search_method``)."""

    PELT = "pelt"
    BINSEG = "binseg"
    BOTTOMUP = "bottomup"
    WBS = "wbs"
    BOCPD = "bocpd"


class DetectionBackend(StrEnum):
    """Graph 1 / Graph 2 detection backend (``algo.detection_backend``)."""

    KERNEL_RUPTURES = "kernel_ruptures"
    CLASSICAL = "classical"


class ClassicalDetector(StrEnum):
    """Hard-label classical detectors (``algo.regime_detection_algorithm``)."""

    BAYESIAN_GMM = "bayesian_gmm"
    HMM = "hmm"
    KMEANS = "kmeans"
    HDBSCAN = "hdbscan"
    OPTICS = "optics"
    MSAR = "msar"
    RUPTURES = "ruptures"
    WASSERSTEIN = "wasserstein"
    JUMP_MODEL = "jump_model"
    STICKY_HDP_HMM = "sticky_hdp_hmm"
    GARCH = "garch"
    MS_VAR = "ms_var"
    STOCHASTIC_VOL = "stochastic_vol"
    CHANGE_IN_COVARIANCE = "change_in_covariance"


class PostProcessing(StrEnum):
    """Breakpoint post-processing methods (``algo.post_processing_method``)."""

    MAJORITY_VOTING = "majority_voting"
    INTERVAL_OVERLAP = "interval_overlap"
    INTERVAL_OVERLAP_EFFICIENT = "interval_overlap_efficient"
    NO_POST_PROCESSING = "nopost_processing"
    NEIGHBOR_COMPARISON = "neighbor_comparison"
    LAG_INTERVAL = "lag_interval"
    ENTROPY = "entropy"


class FeatureMapApprox(StrEnum):
    """Kernel feature-map approximations (``algo.feature_map_approx_method``)."""

    RFF = "rff"
    NYSTROEM = "nystroem"
    INV_RFF = "inv_rff"
    RAW = "raw"


class KernelName(StrEnum):
    """Kernel names used by feature maps / kPCA / ruptures costs."""

    RBF = "rbf"
    LAPLACIAN = "laplacian"
    CHI2 = "chi2"
    POLY = "poly"
    SIGMOID = "sigmoid"
    ADDITIVE_CHI2 = "additive_chi2"
    LINEAR = "linear"
    COSINE = "cosine"


class GammaMethod(StrEnum):
    """RBF bandwidth heuristics (``gamma_method`` / ``gamma`` string values)."""

    USER_SPECIFIED = "user_specified"
    MEDIAN = "median"
    SK_SCALE = "sk_scale"
    TOTAL_VARIANCE = "total_variance"
    UNSCALED_MAX_VARIANCE = "unscaled_max_variance"
    MAX_VARIANCE = "max_variance"


class MetricsMode(StrEnum):
    """Output modes for plots and reports (``metrics.mode``)."""

    DISPLAY = "display"
    WRITE = "write"
    DISPLAY_AND_WRITE = "display_and_write"

    @property
    def writes(self) -> bool:
        return self in (MetricsMode.WRITE, MetricsMode.DISPLAY_AND_WRITE)

    @property
    def displays(self) -> bool:
        return self in (MetricsMode.DISPLAY, MetricsMode.DISPLAY_AND_WRITE)


class RecursiveMethod(StrEnum):
    """Graph 1 case dispatch (``algo.recursive_method``)."""

    FULL = "full"
    ITERATIVE_PCA = "iterative_pca"


class HyperparamMethod(StrEnum):
    """Selection methods for test hyperparameters (lag / window / sample_size)."""

    USER_SPECIFIED = "user_specified"
    ACF_DECAY = "acf_decay"
    PROPORTIONAL = "proportional"
    ESS = "ess"


class SourceType(StrEnum):
    """Data source types (``type`` in source YAML)."""

    SYNTHETIC = "synthetic"
    FAKER = "faker"
    PARQUET = "parquet"
    DUCKDB = "duckdb"
    SQLITE = "sqlite"
    CSV = "csv"


class RegimeClusterAlgorithm(StrEnum):
    """Post-information regime clustering (``metrics.regime_cluster_algorithms``)."""

    KMEANS = "kmeans"
    HDBSCAN = "hdbscan"
    OPTICS = "optics"


def values(enum_cls: type[StrEnum]) -> list[str]:
    """Plain string values of an option enum (for messages and validators)."""
    return [str(m) for m in enum_cls]
