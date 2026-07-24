"""Classical hard-label regime detectors (k-means, HMM, HDBSCAN, …).

Used when ``algo.detection_backend: [classical]``. Soft embeddings of the same
families live under ``gulfstream.dimred.model_based``.
"""

from . import (
    bayesian_gmm,
    garch_regimes,
    hdbscan,
    hmm,
    jump_model,
    kmeans,
    msar,
    optics,
    regime_extras,
    ruptures_methods,
    sticky_hdp_hmm,
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
    "jump_model": jump_model.jump_model_predict_regimes,
    "sticky_hdp_hmm": sticky_hdp_hmm.sticky_hdp_hmm_predict_regimes,
    "garch": garch_regimes.garch_predict_regimes,
    "ms_var": regime_extras.ms_var_predict_regimes,
    "stochastic_vol": regime_extras.stochastic_vol_predict_regimes,
    "change_in_covariance": regime_extras.change_in_covariance_predict_regimes,
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
    "jump_model": jump_model.jump_model_param_generator,
    "sticky_hdp_hmm": sticky_hdp_hmm.sticky_hdp_hmm_param_generator,
    "garch": garch_regimes.garch_param_generator,
    "ms_var": regime_extras.ms_var_param_generator,
    "stochastic_vol": regime_extras.stochastic_vol_param_generator,
    "change_in_covariance": regime_extras.change_in_covariance_param_generator,
}

__all__ = [
    "CLASSICAL_METHODS",
    "CLASSICAL_PARAM_GENERATORS",
    "bayesian_gmm",
    "garch_regimes",
    "hdbscan",
    "hmm",
    "jump_model",
    "kmeans",
    "msar",
    "optics",
    "regime_extras",
    "ruptures_methods",
    "sticky_hdp_hmm",
    "wasserstein",
    "tft",
]
