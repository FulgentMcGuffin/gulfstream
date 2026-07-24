"""Model-based dimred backends: Bayesian GMM, HMM, k-means, HDBSCAN, OPTICS,
MSAR, Wasserstein, ruptures, and TFT.

These reuse the legacy detector models but emit continuous embeddings
(soft responsibilities / distances / attention) so Graph 1 / Graph 2 can
treat them like PCA/kPCA/raw/DMD/t-SNE/UMAP.
"""
from __future__ import annotations

from gulfstream.common.options import DimredMethod
from gulfstream.dimred.model_based.cluster import (
    _bayesian_gmm_generator,
    _hdbscan_generator,
    _kmeans_generator,
    _optics_generator,
)
from gulfstream.dimred.model_based.hmm import _hmm_generator, _msar_generator
from gulfstream.dimred.model_based.ruptures import _ruptures_generator
from gulfstream.dimred.model_based.tft import _tft_generator
from gulfstream.dimred.model_based.wasserstein import _wasserstein_generator

GENERATORS = {
    DimredMethod.BAYESIAN_GMM: _bayesian_gmm_generator,
    DimredMethod.HMM: _hmm_generator,
    DimredMethod.KMEANS: _kmeans_generator,
    DimredMethod.HDBSCAN: _hdbscan_generator,
    DimredMethod.OPTICS: _optics_generator,
    DimredMethod.MSAR: _msar_generator,
    DimredMethod.WASSERSTEIN: _wasserstein_generator,
    DimredMethod.RUPTURES: _ruptures_generator,
    DimredMethod.TFT: _tft_generator,
}

__all__ = ["GENERATORS"]
