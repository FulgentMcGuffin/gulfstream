"""Classical hard-label regime detectors (k-means, HMM, HDBSCAN, …).

Used when ``algo.detection_backend: [classical]``. Soft embeddings of the same
families live under ``gulfstream.dimred.model_based``.
"""

from . import (
    bayesian_gmm,
    hdbscan,
    hmm,
    kmeans,
    msar,
    optics,
    ruptures_methods,
    tft,
    wasserstein,
)

CLASSICAL_METHODS = {
    "bayesian_gmm": bayesian_gmm.bayesian_gmm_predict_regimes,
    "hmm": hmm.hmm_predict_regimes,
    "kmeans": kmeans.kmeans_predict_regimes,
    "hdbscan": hdbscan.hdbscan_predict_regimes,
    "optics": optics.optics_predict_regimes,
    "msar": msar.msar_predict_regimes,
    "ruptures": ruptures_methods.ruptures_predict_regimes,
    "wasserstein": wasserstein.wasserstein_clustering_predict_regimes,
}

CLASSICAL_PARAM_GENERATORS = {
    "bayesian_gmm": bayesian_gmm.bayesian_gmm_param_generator,
    "hmm": hmm.hmm_param_generator,
    "kmeans": kmeans.kmeans_param_generator,
    "hdbscan": hdbscan.hdbscan_param_generator,
    "optics": optics.optics_param_generator,
    "msar": msar.msar_params_generator,
    "ruptures": ruptures_methods.ruptures_param_generator,
    "wasserstein": wasserstein.wass_clustering_param_generator,
}

__all__ = [
    "CLASSICAL_METHODS",
    "CLASSICAL_PARAM_GENERATORS",
    "bayesian_gmm",
    "hdbscan",
    "hmm",
    "kmeans",
    "msar",
    "optics",
    "ruptures_methods",
    "wasserstein",
    "tft",
]
