from __future__ import annotations
"""
Regime detection via Wasserstein clustering.

Some of this code is taken from: https://github.com/Yubo02/Wasserstein-K-means
Many ideas and algorithms are from: https://hal.science/hal-01827184/
"""
import polars as pl
import numpy as np
from typing import Tuple, Callable
import math
import logging
import torch
from dataclasses import dataclass, field
from sklearn.metrics import silhouette_score

from gulfstream.common import frames
from gulfstream.common import utils
from gulfstream.common.results import AlgoResults
from gulfstream.legacy.detectors import common_validation as common

logger = logging.getLogger(__name__)

try:
    from pykeops.torch import generic_sum, generic_logsumexp
    BACKEND = "keops"  # Efficient GPU backend, which scales up to ~100,000 samples.
except ImportError:
    BACKEND = "pytorch"  # Vanilla torch backend. Beware of memory overflows above ~10,000 samples.


@dataclass
class Distribution:
    """
    Minimal version of the Distribution class from Babis' source directory.
    Since all we need are the container properties, making this a dataclass.
    """
    support: torch.Tensor
    weights: torch.Tensor = field(default_factory=lambda: torch.tensor([]))

    def __post_init__(self):
        # Default to uniform weights if not provided.
        if self.weights.numel() == 0:
            self.weights = torch.ones(self.support.size(0), 1, dtype=self.support.dtype)
        else:
            # Ensure weights has the correct shape.
            self.weights = self.weights.view(-1, 1)


# ============================================================
# Main function.
# ============================================================

def wasserstein_clustering_predict_regimes(
    df: pl.DataFrame,
    wass_window: int,
    wass_stride: int,
    regimes: int = None,
    distances: np.ndarray = None,
    opt_transport_reg: float = 0.001,
    **kwargs
) -> AlgoResults:
    """
    Find regimes by Wasserstein clustering.

    Parameters
    ----------
    wass_window, wass_stride : int
        Parameters for lifting df to produce sample distributions.
    regimes : int, optional
        Number of regimes (clusters). If None, will be calculated using silhouette scores.
    distances : np.ndarray, optional
        Matrix of Wasserstein distances. If None, will compute for sliding windows along the rows of df.
    opt_transport_reg : float, optional
        Regularization for optimal transport. Only required if distances is None. Default is 0.001.
    """
    raw_distributions = lift_datastream(df, wass_window, wass_stride)
    distributions = process_distributions(raw_distributions)

    if distances is None:
        distances = compute_distances(distributions, opt_transport_reg)

    if regimes is None:
        logger.info("Determining optimal number of regimes.")
        regimes = _determine_optimal_clusters_silhouette(distributions, distances)
        logger.info("Using %s regimes.", regimes)

    labels = _cluster_wasserstein_using_distributions_known_clusters(
        distributions, regimes, distance_matrix=distances
    )
    # Convert from labels for the lifted data to the original dataframe.
    raw_labels = recover_labels(df, labels, wass_window, wass_stride)
    labels = utils._map_labels_to_ordered_integers(raw_labels)
    bkpts = utils._convert_labels_to_bkpts(labels)

    return AlgoResults(bkpts=bkpts, labels=labels)


def wass_clustering_param_generator(params: dict):
    """
    Yield all valid combinations of parameters for Wasserstein clustering regime detection.
    """
    if 'wasserstein' in params['algo']['regime_detection_algorithm']:
        for wass_window in params['algo']['wass_window']:
            for wass_stride in params['algo']['wass_stride']:
                for regimes in params['algo'].get('regimes', [None]):
                    for opt_transport_reg in params['algo'].get('opt_transport_reg', [0.001]):
                        yield {
                            'regime_detection_algorithm': 'wasserstein',
                            'wass_window': wass_window,
                            'wass_stride': wass_stride,
                            'regimes': regimes,
                            'opt_transport_reg': opt_transport_reg
                        }


def wass_params_printout() -> dict:
    """
    Dict for writing heading of this algorithm. Each entry will be a
    column heading in an Excel sheet. Values are singleton lists with
    the heading to be used for the associated column. Keys should use
    the format "algoname_paramname".
    """
    return {
        'wasserstein_wass_window': ['rolling window size'],
        'wasserstein_wass_stride': ['stride for rolling window'],
        'wasserstein_regimes': ['number of regimes'],
        'wasserstein_opt_transport_reg': ['optimal transport regularization']
    }


# ============================================================
# INPUT VALIDATION
# ============================================================

def _valid_window(algo_params: dict) -> bool:
    window = algo_params.get('wass_window')
    if window is None:
        logger.error("'wass_window' (list[int]) in 'algo' must be provided for Wasserstein clustering.")
        return False
    elif not isinstance(window, list):
        logger.error("'wass_window' in 'algo' must be type list[int]. Got type %s.", type(window))
        return False
    elif not all(isinstance(x, int) and x > 0 for x in window):
        logger.error("All entries of 'wass_window' must be positive integers.")
        return False
    return True


def _valid_stride(algo_params: dict) -> bool:
    stride = algo_params.get('wass_stride')
    if stride is None:
        logger.error("'wass_stride' (list[int]) in 'algo' must be provided for Wasserstein clustering.")
        return False
    elif not isinstance(stride, list):
        logger.error("'wass_stride' in 'algo' must be type list[int]. Got type %s.", type(stride))
        return False
    elif not all(isinstance(x, int) and x > 0 for x in stride):
        logger.error("All entries of 'wass_stride' must be positive integers.")
        return False
    return True


def _valid_opt_transport_reg(algo_params: dict) -> bool:
    reg = algo_params.get('opt_transport_reg')
    if reg is None:
        return True  # Optional parameter.
    elif not isinstance(reg, list):
        logger.error("'opt_transport_reg' in 'algo' must be type list[float]. Got type %s.", type(reg))
        return False
    elif not all(isinstance(x, (int, float)) and x >= 0 for x in reg):
        logger.error("All entries of 'opt_transport_reg' must be non-negative floats.")
        return False
    return True


def wass_input_validator(algo_params: dict) -> bool:
    """
    Returns True if algo_params contains a valid set of parameters
    for Wasserstein clustering detection. May contain additional parameters as well.
    """
    valid = True
    if not common._valid_regimes(algo_params):
        valid = False
    if not _valid_window(algo_params):
        valid = False
    if not _valid_stride(algo_params):
        valid = False
    if not _valid_opt_transport_reg(algo_params):
        valid = False
    return valid


# ============================================================
# HELPERS
# ============================================================

def _get_lifted_single(df: pl.DataFrame, m: int, window: int, stride: int) -> pl.DataFrame:
    """
    Get a slice of df between rows stride*(m-1) and stride*(m-1)+window.
    """
    start_index = stride * (m - 1)
    end_index = start_index + window
    indices = list(range(start_index, end_index))
    return frames.slice_rows(df, start_index, end_index)


def lift_datastream(datastream: pl.DataFrame, window: int, stride: int) -> list[pl.DataFrame]:
    """
    Given a datastream, perform lifting as in https://arxiv.org/abs/2310.01285
    using the sliding window lift.
    """
    data_len = datastream.height
    M = math.floor((data_len - (window - stride)) / stride)
    raw_distributions = []
    for m in range(1, M + 1):
        selected_rows = _get_lifted_single(datastream, m, window, stride)
        raw_distributions.append(selected_rows)
    return raw_distributions


def get_lifted_indices(df: pl.DataFrame, m: int, window: int, stride: int) -> Tuple[int, int]:
    """
    Return the start and end indices of df for the lifted slice from _get_lifted_single().
    """
    start_index = stride * (m - 1)
    end_index = start_index + window
    return start_index, end_index


def process_distributions(raw_distributions: list[pl.DataFrame], keep_top: int = None) -> list[Distribution]:
    """
    Convert the dataframes in raw_distributions into Distribution objects.
    Use only the first keep_top dataframes if keep_top is not None.
    """
    n_distributions = len(raw_distributions)
    if keep_top is not None and keep_top <= n_distributions:
        n_distributions = keep_top
        raw_distributions = raw_distributions[0:keep_top]
    distrib = []
    for i in range(n_distributions):
        n_samples = raw_distributions[i].height
        support = torch.tensor(frames.to_numpy(raw_distributions[i]), dtype=torch.float32)
        weights = torch.ones(n_samples) / n_samples
        new_distrib = Distribution(support, weights)
        distrib.append(new_distrib)
    return distrib


def compute_distances(distributions: list[Distribution], reg: float = 0.001) -> np.ndarray:
    """
    Return matrix of Wasserstein distances between the distributions.
    """
    num_distribs = len(distributions)
    distance_matrix = np.full((num_distribs, num_distribs), 0.0)
    for i in range(1, num_distribs):
        for j in range(0, i):
            distance_matrix[i][j] = sinkhorn_divergence(
                distributions[j].weights,
                distributions[j].support,
                distributions[i].weights,
                distributions[i].support,
                eps=reg
            )[0]
    return distance_matrix + distance_matrix.T


def recover_labels(
    df: pl.DataFrame,
    condensed_labels: list[int],
    window: int,
    stride: int
) -> list[int]:
    """
    Convert from condensed labels for the lifted data into labels for the original data df.
    """
    data_len = df.height
    M = math.floor((data_len - (window - stride)) / stride)
    # Temporarily use numpy array for vectorized assignment to slices.
    labels = np.full(data_len, -1)
    for m in range(1, M + 1):
        start, end = get_lifted_indices(df, m, window, stride)
        # Note that here, later labels are replacing earlier labels if the lifted sequences overlap.
        labels[start:end] = condensed_labels[m - 1]

    bad_indices = np.where(labels == -1)[0]
    first_minus_one = bad_indices[0] if bad_indices.size > 0 else len(labels)
    if first_minus_one < len(labels):
        if first_minus_one > 0:
            logger.warning("Some labels are invalid. Assigning last valid label to all labels past the first invalid.")
            labels[first_minus_one:] = labels[first_minus_one - 1]
        else:
            logger.warning("All labels are invalid. There is probably a bug. Assigning label 0 to all.")
            labels[:] = 0

    return labels.tolist()


def _determine_optimal_clusters_silhouette(
    distributions: list[Distribution],
    distances: np.ndarray,
    max_clusters: int = 10
) -> int:
    """
    Return the optimal number of clusters using silhouette scores.
    """
    silhouette_scores = []
    cluster_range = range(2, max_clusters + 1)
    for n_clusters in cluster_range:
        cluster_labels = _cluster_wasserstein_using_distributions_known_clusters(
            distributions, n_clusters, distances
        )
        silhouette_avg = silhouette_score(distances, cluster_labels, metric='precomputed')
        silhouette_scores.append(silhouette_avg)
        logger.info("For %s clusters, silhouette score is %s.", n_clusters, silhouette_avg)
    optimal_clusters = cluster_range[np.argmax(silhouette_scores)]
    logger.info("Optimal number of clusters is %s.", optimal_clusters)
    return optimal_clusters


def _cluster_wasserstein_using_distributions_known_clusters(
    distributions: list[Distribution],
    n_clusters: int,
    distance_matrix: np.ndarray,
    reg: float = 0.001,
    kmeans_iteration_max: int = 100
) -> list[int]:
    """
    Return class labels for Wasserstein clustering with n_clusters clusters.

    Parameters
    ----------
    distributions : list[Distribution]
        Sample distributions to perform clustering on.
    distance_matrix : np.ndarray
        Matrix of pairwise distances between the distributions in distributions.
    reg : float, optional
        Regularization for optimal transport. Default is 0.001.

    Returns
    -------
    list[int]
        Cluster labels.
    """
    num_distribs = len(distributions)
    ind_rand = np.random.randint(0, num_distribs)
    # Initialize the centroids.
    first_centroid = distributions[ind_rand]
    centroids_distrib = _initial_plus(first_centroid, distributions, n_clusters, reg)

    groups, groups_ind = _partition_into_groups_withind(distributions, centroids_distrib, reg)
    kmeans_iteration = 0
    # Previous cluster labels. Stop early if the algorithm doesn't update the clusters.
    groups_ind0 = []
    while kmeans_iteration < kmeans_iteration_max and groups_ind != groups_ind0:
        groups_ind0 = groups_ind
        # Update the clusters.
        groups_ind = _partition_into_groups_DWKM(distributions, distance_matrix, groups_ind, n_clusters)
        kmeans_iteration += 1

    # Convert to a single list of labels. Currently, each group is a list of indices.
    labels = [None] * num_distribs
    for cluster_label, cluster_indices in enumerate(groups_ind):
        for index in cluster_indices:
            labels[index] = cluster_label

    return labels


def _initial_plus(
    chosen_random: Distribution,
    data: list[Distribution],
    num_groups: int,
    reg: float = 0.001
) -> list[Distribution]:
    """
    Return initial clusters for distance-based Wasserstein K-means (D-WKM).
    Taken from https://github.com/Yubo02/Wasserstein-K-means-for-clustering-probability-distributions
    """
    centroids = []
    centroids.append(chosen_random)
    # -1 because we already added the first centroid chosen_random.
    for k in range(num_groups - 1):
        distances = np.zeros(len(data))
        for i in range(len(data)):
            distances[i] = sinkhorn_divergence(
                chosen_random.weights,
                chosen_random.support,
                data[i].weights,
                data[i].support,
                eps=reg
            )[0]
        probs = distances ** 2
        index = np.random.choice(np.arange(0, len(data)), p=probs / sum(probs))
        centroids.append(data[index])
        chosen_random = data[index]
    return centroids


def _partition_into_groups_withind(
    data: list[Distribution],
    centroids: list[Distribution],
    reg: float = 0.001
) -> Tuple[list[list[Distribution]], list[list[int]]]:
    """
    Assign the distributions in data to the nearest clusters based on the centroids.
    Taken from https://github.com/Yubo02/Wasserstein-K-means-for-clustering-probability-distributions

    Returns
    -------
    list[list[Distribution]]
        Each sublist is a cluster of distributions from data based on the centroids.
    list[list[int]]
        Each sublist is the list of indices of the distributions from data in each cluster.
    """
    groups = [[] for i in range(len(centroids))]
    groups_ind = [[] for i in range(len(centroids))]

    for i in range(len(data)):
        min_dist = 100
        for k in range(len(centroids)):
            dist = sinkhorn_divergence(
                centroids[k].weights,
                centroids[k].support,
                data[i].weights,
                data[i].support,
                eps=reg
            )[0]
            if dist < min_dist:
                tmp_c = k
                min_dist = dist
        groups[tmp_c].append(data[i])
        groups_ind[tmp_c].append(i)
    return groups, groups_ind


def _partition_into_groups_DWKM(
    data: list[Distribution],
    distance_matrix: np.ndarray,
    groups_ind0: list[int],
    num_groups: int
) -> list[list[int]]:
    """
    Update the clustering by finding the index of the current cluster that each
    distribution in data is closest to on average. Return the new clusters (as lists
    of lists of indices). Taken from
    https://github.com/Yubo02/Wasserstein-K-means-for-clustering-probability-distributions
    """
    groups_ind = [[] for i in range(num_groups)]
    for i in range(len(data)):
        min_dist = float('inf')
        for k in range(len(groups_ind0)):
            dist = _sinkhorn_divergence_cluster(groups_ind0[k], i, distance_matrix)
            if dist < min_dist:
                tmp_c = k
                min_dist = dist
        groups_ind[tmp_c].append(i)
    return groups_ind


def _sinkhorn_divergence_cluster(
    groups_ind_0: list[int],
    idx: int,
    distance_matrix: np.ndarray
) -> float:
    """
    Return the average squared distance from distribution with index idx to each
    distribution in the cluster with indices groups_ind_0. Taken from
    https://github.com/Yubo02/Wasserstein-K-means-for-clustering-probability-distributions/
    """
    total_dist = 0
    for i in range(len(groups_ind_0)):
        # Distance from distribution idx to the i'th distribution in the cluster.
        total_dist += distance_matrix[groups_ind_0[i]][idx]
    return total_dist / len(groups_ind_0)


# ============================================================
# SINKHORN ALGORITHM IMPLEMENTATIONS
# ============================================================

def _scal(alpha: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    """
    Dot (scalar) product between alpha and f. Result is a 0-dimensional tensor.
    """
    return torch.dot(alpha.view(-1), f.view(-1))


def _lse(v_ij: torch.Tensor) -> torch.Tensor:
    """
    [lse(v_ij)]_i = log sum_j exp(v_ij), with numerical accuracy.
    """
    V_i = torch.max(v_ij, 1)[0].view(-1, 1)
    return V_i + (v_ij - V_i).exp().sum(1).log().view(-1, 1)


def _sinkhorn_ops(p: int, eps: float, x: torch.Tensor, y: torch.Tensor) -> Tuple[Callable, Callable]:
    """
    Return routines S_x and S_y such that
        [S_x(f_i)]_j = -log sum_i exp(f_i - |x-y|^p / eps)
        [S_y(f_j)]_i = -log sum_j exp(f_j - |x-y|^p / eps)

    Parameters
    ----------
    p : int
        Distance exponent. Currently only support p==1 and p==2 if using BACKEND=='keops'.
    eps : float
        Regularization strength.
    x, y : torch.Tensor
        Data matrices. Shapes N-by-D and M-by-D respectively.

    Notes
    -----
    This may look like a strange level of abstraction, but it is the most convenient
    way of working with KeOps and vanilla pytorch (with a pre-computed cost matrix) at the same time.
    """
    if BACKEND == "keops":
        # Memory-efficient GPU implementation: ONline logsumexp routine.
        if p == 1:
            formula = "Fj - (Sqrt(SqDist(Xi, Yj)) / E)"
        elif p == 2:
            formula = "Fj - (SqDist(Xi, Yj) / E)"
        else:
            formula = "Fj - (Powf(SqDist(Xi, Yj), R) / E)"
            raise NotImplementedError("I should fix the derivative at 0 of Powf, in Keops's core.")
        D = x.shape[1]  # Dimension of the ambient space (typically 2 or 3)
        routine = generic_logsumexp(
            formula, "outi = Vx(1)",
            "E = Pm(1)", "R = Pm(1)",
            "Xi = Vx({})".format(D), "Yj = Vy({})".format(D), "Fj = Vy(1)",
        )
        eps_t, r = torch.Tensor([eps]).type_as(x), torch.Tensor([p / 2]).type_as(x)
        S_x = lambda f_i: -routine(eps_t, r, x, y, f_i)
        S_y = lambda f_j: -routine(eps_t, r, y, x, f_j)
        return S_x, S_y

    elif BACKEND == "pytorch":
        # Naive matrix-vector implementation: OFFline logsumexp.
        # Precompute the |x-y|^p matrix once and for all.
        x_y = x.unsqueeze(1) - y.unsqueeze(0)
        if p == 1:
            C_e = x_y.norm(dim=2) / eps
        elif p == 2:
            C_e = (x_y ** 2).sum(2) / eps
        else:
            C_e = x_y.norm(dim=2) ** (p / 2) / eps
        CT_e = C_e.t()

        # Don't forget the minus!
        S_x = lambda f_i: -_lse(f_i.view(1, -1) - CT_e)
        S_y = lambda f_j: -_lse(f_j.view(1, -1) - C_e)
        return S_x, S_y


def sink(
    alpha: torch.Tensor,
    x: torch.Tensor,
    beta: torch.Tensor,
    y: torch.Tensor,
    p: int = 1,
    eps: float = 0.1,
    nits: int = 100,
    tol: float = 1e-3,
    assume_convergence: bool = False,
    **kwargs
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimal transport between probability measures with entropic regularization
    with coefficient eps. See Algorithm 1 in https://hal.science/hal-01827184/

    Parameters
    ----------
    x, alpha, y, beta : torch.Tensor
        Probability measures. Point masses at each entry of x with mass of
        corresponding value of alpha. Likewise with y and beta.
    p : int, optional
        Distance exponent. Currently only support 1 and 2 if using BACKEND=='keops'.
        Default is 1 (Earth Mover's distance).
    eps : float, optional
        Regularization factor for entropy. Default is 0.1.
    nits : int, optional
        Number of iterations. Default is 100.
    tol : float, optional
        L1 error threshold for stopping the algorithm. Default is 1e-3.
    assume_convergence : bool, optional
        If True, will disable gradient tracking for the main Sinkhorn loop (more efficient).
        Default is False.

    Returns
    -------
    torch.Tensor, torch.Tensor
        Final influence fields.

    Notes
    -----
    The original version of this included code for generating heatmaps, but it was
    referencing a nonexistent global variable "grid", causing it to break.
    I (Charley) removed it for this implementation.
    """
    # Sinkhorn loop with A = a/eps, B = b/eps.
    # Precompute the logs of the measures' weights for efficiency.
    alpha_log, beta_log = alpha.log(), beta.log()
    # Sampled influence fields.
    B_i, A_j = torch.zeros_like(alpha), torch.zeros_like(beta)
    # If we assume convergence, we can skip all the "save computational history" stuff.
    torch.set_grad_enabled(not assume_convergence)

    # Softmin operators (divided by eps, as it's slightly cheaper).
    S_x, S_y = _sinkhorn_ops(p, eps, x, y)
    for i in range(nits - 1):
        B_i_prev = B_i
        # a(y)/eps = Smin_eps, x~alpha [C(x,y) - b(x)] / eps
        A_j = S_x(B_i + alpha_log)
        # b(x)/eps = Smin_eps, y~beta [C(x,y) - a(y)] / eps
        B_i = S_y(A_j + beta_log)
        # Stopping criterion: L1 norm of the updates
        err = eps * (B_i - B_i_prev).abs().mean()
        if err.item() < tol:
            break
    torch.set_grad_enabled(True)

    # One last step, which allows us to bypass PyTorch's backprop engine if required
    # (as explained in the paper).
    if not assume_convergence:
        A_j = S_x(B_i + alpha_log)
        B_i = S_y(A_j + beta_log)
    else:
        # Assume that we have converged, and can thus use the "exact" (and cheap!) gradient's formula.
        S_x, _ = _sinkhorn_ops(p, eps, x.detach(), y)
        _, S_y = _sinkhorn_ops(p, eps, x, y.detach())
        A_j = S_x((B_i + alpha_log).detach())
        B_i = S_y((A_j + beta_log).detach())

    a_y, b_x = eps * A_j.view(-1), eps * B_i.view(-1)
    return a_y, b_x


def sym_sink(
    alpha: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor = None,
    p: int = 1,
    eps: float = 0.1,
    nits: int = 100,
    tol: float = 1e-3,
    assume_convergence: bool = False,
    **kwargs
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimal transport between probability measures with entropic regularization
    with coefficient eps. See Algorithm 2 in https://hal.science/hal-01827184/

    Parameters
    ----------
    x, alpha : torch.Tensor
        Probability measure. Point masses at each entry of x with mass of
        corresponding value of alpha.
    y : torch.Tensor
        Target point cloud. If None, just compute the influence field for x.
        Default is None.
    p : int, optional
        Distance exponent. Currently only support 1 and 2 if using BACKEND=='keops'.
        Default is 1 (Earth Mover's distance).
    eps : float, optional
        Regularization factor for entropy. Default is 0.1.
    nits : int, optional
        Number of iterations. Default is 100.
    tol : float, optional
        L1 error threshold for stopping the algorithm. Default is 1e-3.
    assume_convergence : bool, optional
        If True, will disable gradient tracking for the main Sinkhorn loop (more efficient).
        Default is False.

    Returns
    -------
    torch.Tensor, torch.Tensor
        Final influence fields.

    Notes
    -----
    The original version of this included code for generating heatmaps, but it was
    referencing a nonexistent global variable "grid", causing it to break.
    I (Charley) removed it for this implementation.
    """
    # Sinkhorn loop.
    # Precompute the logs of the measure's weights for efficiency.
    alpha_log = alpha.log()
    # Sampled influence field.
    A_i = torch.zeros_like(alpha)

    # Sinkhorn operator from x to x (divided by eps, as it's slightly cheaper).
    S_x, _ = _sinkhorn_ops(p, eps, x, x)
    # If we assume convergence, we can skip all the "save computational history" stuff.
    torch.set_grad_enabled(not assume_convergence)

    for i in range(nits - 1):
        A_i_prev = A_i
        # a(x)/eps = .5*(a(x)/eps + Smin_eps, y~alpha [C(x,y) - a(y)] / eps)
        A_i = 0.5 * (A_i + S_x(A_i + alpha_log))
        # Stopping criterion: L1 norm of the updates
        err = eps * (A_i - A_i_prev).abs().mean()
        if err.item() < tol:
            break
    torch.set_grad_enabled(True)

    # One last step.
    if not assume_convergence:
        W_i = A_i + alpha_log
        if y is not None:
            # Sinkhorn operator from x to y (divided by eps).
            S2_x, _ = _sinkhorn_ops(p, eps, x, y)
    else:
        W_i = (A_i + alpha_log).detach()
        S_x, _ = _sinkhorn_ops(p, eps, x.detach(), x)
        if y is not None:
            S2_x, _ = _sinkhorn_ops(p, eps, x.detach(), y)

    # a(x) = Smin_e, z~alpha [C(x, z) - a(z)]
    a_x = eps * S_x(W_i).view(-1)

    if y is None:
        return None, a_x
    else:
        # Extrapolate "a" to the point cloud "y".
        # a(z) = Smin_e, z~alpha [C(y, z) - a(z)]
        a_y = eps * S2_x(W_i).view(-1)
        return a_y, a_x


def sinkhorn_divergence(
    alpha: torch.Tensor,
    x: torch.Tensor,
    beta: torch.Tensor,
    y: torch.Tensor,
    **params
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Return the sinkhorn divergence and influence fields from regularized optimal transport.

    Parameters
    ----------
    x, alpha, y, beta : torch.Tensor
        Probability measures. Point masses at each entry of x with mass of
        corresponding value of alpha. Likewise with y and beta.
    params : dict
        Keyword arguments for sink and sym_sink.
    """
    a_y, b_x = sink(alpha, x, beta, y, **params)
    _, a_x = sym_sink(alpha, x, **params)
    _, b_y = sym_sink(beta, y, **params)

    cost = _scal(alpha, b_x - a_x) + _scal(beta, a_y - b_y)
    return cost, a_y, a_x, b_x, b_y
