"""Input validation for dimension reduction parameters."""
import logging
from . import common

logger = logging.getLogger(__name__)

DIMRED_METHODS = [
    'pca', 'kpca', 'raw', 'dmd', 'tsne', 'umap',
    'bayesian_gmm', 'hmm', 'kmeans', 'hdbscan', 'optics', 'msar', 'wasserstein',
    'ruptures', 'tft',
]


def _provided_dimred(params: dict) -> bool:
    if not common._provided_algo(params):
        return False
    algo = params['algo']
    dimred = algo.get('dimred')
    if not dimred:
        logger.error("'dimred' must be provided.")
        return False
    elif not isinstance(dimred, list):
        logger.error("'dimred' must be a list.")
        return False
    elif len(dimred) == 0:
        logger.error("'dimred' must be a nonempty list.")
        return False
    invalid_dimred_methods = [x for x in dimred if x not in DIMRED_METHODS]
    if invalid_dimred_methods:
        invalid_dimred_methods = list(set(invalid_dimred_methods))
        logger.error("Invalid 'dimred' methods: %s.", ', '.join(invalid_dimred_methods))
        return False
    return True


def _using_dmd(params: dict) -> bool:
    if not _provided_dimred(params):
        return False
    if 'dmd' in params['algo']['dimred']:
        return True
    return False


def _valid_dmd_stride(params: dict) -> bool:
    if not common._is_nonempty_list(params['algo'], 'dmd_stride', int):
        return False
    elif not all(x > 0 for x in params['algo']['dmd_stride']):
        logger.error("All entries of 'dmd_stride' must be positive integers.")
        return False
    return True


def _valid_dmd_rolling_window(params: dict) -> bool:
    if not common._is_nonempty_list(params['algo'], 'dmd_rolling_window', int):
        return False
    elif not all(x > 0 for x in params['algo']['dmd_rolling_window']):
        logger.error("All entries of 'dmd_rolling_window' must be positive integers.")
        return False
    return True


def _valid_rank_selection_method(params: dict) -> bool:
    if not common._is_nonempty_list(params['algo'], 'rank_selection_method', str):
        return False
    elif not all(
        x in ['entropy', 'explained_variance', 'user_specified', 'svht']
        for x in params['algo']['rank_selection_method']
    ):
        logger.error(
            "Invalid entries of 'rank_selection_method'. Valid entries are "
            "'entropy', 'explained_variance', 'user_specified', and 'svht'."
        )
        return False
    return True


def _valid_entropy(params: dict) -> bool:
    if not common._is_nonempty_list(params['algo'], 'threshold', (int, float)):
        return False
    elif not all(0 < x < 1 for x in params['algo']['threshold']):
        logger.error("All entries of 'threshold' must be between 0 and 1.")
        return False
    return True


def _valid_explained_variance(params: dict) -> bool:
    if not common._is_nonempty_list(params['algo'], 'threshold', (int, float)):
        return False
    elif not all(0 < x < 1 for x in params['algo']['threshold']):
        logger.error("All entries of 'threshold' must be between 0 and 1.")
        return False
    return True


def _valid_user_specified_rank(params: dict) -> bool:
    if not common._is_nonempty_list(params['algo'], 'rank', int):
        return False
    elif not all(x > 0 for x in params['algo']['rank']):
        logger.error("All entries of 'rank' must be positive integers.")
        return False
    return True


def _valid_svht(params: dict) -> bool:
    return True


def _valid_dmd_params(params: dict) -> bool:
    if not _using_dmd(params):
        return True
    valid = _valid_dmd_stride(params)
    if not _valid_dmd_rolling_window(params):
        valid = False
    if not _valid_rank_selection_method(params):
        valid = False
    else:
        methods = set(params['algo']['rank_selection_method'])
        handlers = {
            'entropy': _valid_entropy,
            'explained_variance': _valid_explained_variance,
            'user_specified': _valid_user_specified_rank,
            'svht': _valid_svht
        }
        invalid_methods = []
        for method in methods:
            handler = handlers[method]
            if not handler(params):
                invalid_methods.append(method)
        if invalid_methods:
            valid = False
    return valid


def _using_tsne(params: dict) -> bool:
    if not _provided_dimred(params):
        return False
    return 'tsne' in params['algo']['dimred']


def _valid_tsne_perplexity(params: dict) -> bool:
    if not common._is_nonempty_list(params['algo'], 'tsne_perplexity', (int, float)):
        return False
    elif not all(x > 0 for x in params['algo']['tsne_perplexity']):
        logger.error("All entries of 'tsne_perplexity' must be positive.")
        return False
    return True


def _valid_tsne_n_iter(params: dict) -> bool:
    if not common._is_nonempty_list(params['algo'], 'tsne_n_iter', int):
        return False
    elif not all(x > 0 for x in params['algo']['tsne_n_iter']):
        logger.error("All entries of 'tsne_n_iter' must be positive integers.")
        return False
    return True


def _valid_tsne_params(params: dict) -> bool:
    if not _using_tsne(params):
        return True
    valid = _valid_tsne_perplexity(params)
    if not _valid_tsne_n_iter(params):
        valid = False
    # t-SNE only supports user_specified rank.
    ranks = params['algo'].get('rank')
    if ranks is None:
        logger.error("'rank' must be provided when using 'tsne'.")
        valid = False
    elif not _valid_user_specified_rank(params):
        valid = False
    return valid


def _using_umap(params: dict) -> bool:
    if not _provided_dimred(params):
        return False
    return 'umap' in params['algo']['dimred']


def _valid_umap_num_neighbors(params: dict) -> bool:
    if not common._is_nonempty_list(params['algo'], 'umap_num_neighbors', (int, float)):
        return False
    elif not all(x > 0 for x in params['algo']['umap_num_neighbors']):
        logger.error("All entries of 'umap_num_neighbors' must be positive.")
        return False
    return True


def _valid_umap_min_dist(params: dict) -> bool:
    if not common._is_nonempty_list(params['algo'], 'umap_min_dist', (int, float)):
        return False
    elif not all(0 <= x <= 1 for x in params['algo']['umap_min_dist']):
        logger.error("All entries of 'umap_min_dist' must be in [0, 1].")
        return False
    return True


def _valid_umap_metric(params: dict) -> bool:
    if not common._is_nonempty_list(params['algo'], 'umap_metric', str):
        return False
    return True


def _valid_umap_params(params: dict) -> bool:
    if not _using_umap(params):
        return True
    valid = _valid_umap_num_neighbors(params)
    if not _valid_umap_min_dist(params):
        valid = False
    if not _valid_umap_metric(params):
        valid = False
    ranks = params['algo'].get('rank')
    if ranks is None:
        logger.error("'rank' must be provided when using 'umap'.")
        valid = False
    elif not _valid_user_specified_rank(params):
        valid = False
    return valid


def _using_pca(params: dict) -> bool:
    if not _provided_dimred(params):
        return False
    if 'pca' in params['algo']['dimred']:
        return True
    return False


def _valid_pca_params(params: dict) -> bool:
    if not _using_pca(params):
        return True
    if not _valid_rank_selection_method(params):
        return False
    methods = set(params['algo']['rank_selection_method'])
    handlers = {
        'entropy': _valid_entropy,
        'explained_variance': _valid_explained_variance,
        'user_specified': _valid_user_specified_rank,
        'svht': _valid_svht
    }
    invalid_methods = []
    for method in methods:
        handler = handlers[method]
        if not handler(params):
            invalid_methods.append(method)
    if invalid_methods:
        return False
    return True


def _valid_raw_dimred_params(params: dict) -> bool:
    return True


def _using_kpca(params: dict) -> bool:
    if not _provided_dimred(params):
        return False
    if 'kpca' in params['algo']['dimred']:
        return True
    return False


def _valid_kpca_kernel_params(params: dict) -> bool:
    required_params = {
        'rbf': ['gamma'],
        'poly': ['degree', 'coef0', 'gamma'],
        'sigmoid': ['coef0', 'gamma'],
        'linear': [],
        'cosine': [],
    }
    kernel_params_list = params['algo'].get('kpca_kernel_params')
    if kernel_params_list is None:
        logger.error("'kpca_kernel_params' must be provided when using 'kpca'.")
        return False
    if not isinstance(kernel_params_list, list):
        logger.error("'kpca_kernel_params' must be a list of dicts.")
        return False
    invalid_dicts = []
    for d in kernel_params_list:
        kernel = d.get('kernel')
        if not kernel:
            logger.error("'kernel' must be specified in each dict in 'kpca_kernel_params'.")
            invalid_dicts.append(d)
            continue
        required = required_params.get(kernel)
        if required is None:
            logger.error("Unknown kernel %s for 'kpca_kernel_params'.", kernel)
            invalid_dicts.append(d)
            continue
        missing = [x for x in required if x not in d]
        if missing:
            logger.error(
                "Kernel %s for kernel PCA is missing required parameters %s.",
                kernel, ', '.join(missing)
            )
            invalid_dicts.append(d)
            continue
    if invalid_dicts:
        return False
    return True


def _valid_kpca_params(params: dict) -> bool:
    if not _using_kpca(params):
        return True
    valid = _valid_kpca_kernel_params(params)
    if not _valid_rank_selection_method(params):
        valid = False
    else:
        methods = set(params['algo']['rank_selection_method'])
        handlers = {
            'entropy': _valid_entropy,
            'explained_variance': _valid_explained_variance,
            'user_specified': _valid_user_specified_rank,
            'svht': _valid_svht
        }
        invalid_methods = []
        for method in methods:
            handler = handlers[method]
            if not handler(params):
                invalid_methods.append(method)
        if invalid_methods:
            valid = False
    return valid


def _valid_model_based_regimes(params: dict) -> bool:
    """Model-based dimred uses algo.regimes as embedding dimensionality."""
    regimes = params['algo'].get('regimes')
    dimreds = set(params['algo']['dimred'])
    # These may omit regimes (silhouette / pelt penalty / tft uses rank).
    # Density / auto-k methods discover cluster count; TFT uses rank; ruptures uses penalty.
    optional = {'wasserstein', 'ruptures', 'tft', 'hdbscan', 'optics'}
    if regimes is None:
        if dimreds <= optional or dimreds & optional:
            # Still require regimes for any non-optional model-based method present.
            needed = dimreds - optional - {'pca', 'kpca', 'raw', 'dmd', 'tsne', 'umap'}
            if not needed:
                return True
        logger.error("'regimes' (list[int]) must be provided for model-based dimred.")
        return False
    if not isinstance(regimes, list) or len(regimes) == 0:
        logger.error("'regimes' must be a nonempty list[int].")
        return False
    if not all(isinstance(x, int) and x > 0 for x in regimes):
        logger.error("All entries of 'regimes' must be positive ints.")
        return False
    return True


def _valid_bayesian_gmm_dimred_params(params: dict) -> bool:
    if 'bayesian_gmm' not in params['algo']['dimred']:
        return True
    valid = _valid_model_based_regimes(params)
    reg_covar = params['algo'].get('reg_covar')
    if reg_covar is not None:
        if not isinstance(reg_covar, list) or not all(
            isinstance(x, (int, float)) and x >= 0 for x in reg_covar
        ):
            logger.error("'reg_covar' must be list of non-negative floats.")
            valid = False
    return valid


def _valid_hmm_dimred_params(params: dict) -> bool:
    if 'hmm' not in params['algo']['dimred']:
        return True
    valid = _valid_model_based_regimes(params)
    emissions = params['algo'].get('hmm_emissions', ['gaussian'])
    if not isinstance(emissions, list) or not emissions:
        logger.error("'hmm_emissions' must be a nonempty list[str].")
        return False
    if not all(e == 'gaussian' for e in emissions):
        logger.error("HMM dimred currently supports only 'gaussian' emissions.")
        valid = False
    n_iter = params['algo'].get('hmm_n_iter')
    if n_iter is not None and (
        not isinstance(n_iter, list) or not all(isinstance(x, int) and x > 0 for x in n_iter)
    ):
        logger.error("'hmm_n_iter' must be list of positive ints.")
        valid = False
    return valid


def _valid_kmeans_dimred_params(params: dict) -> bool:
    if 'kmeans' not in params['algo']['dimred']:
        return True
    valid = _valid_model_based_regimes(params)
    rs = params['algo'].get('random_state')
    if rs is not None and (
        not isinstance(rs, list)
        or not all(x is None or (isinstance(x, int) and x >= 0) for x in rs)
    ):
        logger.error("'random_state' must be list of non-negative ints or null.")
        valid = False
    return valid


def _valid_hdbscan_dimred_params(params: dict) -> bool:
    if 'hdbscan' not in params['algo']['dimred']:
        return True
    from gulfstream.legacy.detectors.hdbscan import hdbscan_input_validator

    return hdbscan_input_validator(params['algo'])


def _valid_optics_dimred_params(params: dict) -> bool:
    if 'optics' not in params['algo']['dimred']:
        return True
    from gulfstream.legacy.detectors.optics import optics_input_validator

    return optics_input_validator(params['algo'])


def _valid_msar_dimred_params(params: dict) -> bool:
    if 'msar' not in params['algo']['dimred']:
        return True
    return _valid_model_based_regimes(params)


def _valid_wasserstein_dimred_params(params: dict) -> bool:
    if 'wasserstein' not in params['algo']['dimred']:
        return True
    valid = True
    for key in ('wass_window', 'wass_stride'):
        vals = params['algo'].get(key)
        if not isinstance(vals, list) or not vals or not all(
            isinstance(x, int) and x > 0 for x in vals
        ):
            logger.error("'%s' must be a nonempty list of positive ints.", key)
            valid = False
    regimes = params['algo'].get('regimes')
    if regimes is not None and (
        not isinstance(regimes, list)
        or not all(isinstance(x, int) and x > 0 for x in regimes)
    ):
        logger.error("'regimes' must be list of positive ints when provided.")
        valid = False
    reg = params['algo'].get('opt_transport_reg')
    if reg is not None and (
        not isinstance(reg, list)
        or not all(isinstance(x, (int, float)) and x > 0 for x in reg)
    ):
        logger.error("'opt_transport_reg' must be list of positive floats.")
        valid = False
    return valid


def _valid_ruptures_dimred_params(params: dict) -> bool:
    if 'ruptures' not in params['algo']['dimred']:
        return True
    valid = True
    algos = params['algo'].get('ruptures_algorithm')
    if not isinstance(algos, list) or not algos:
        logger.error("'ruptures_algorithm' must be a nonempty list[str].")
        return False
    allowed = {'pelt', 'window', 'dynp'}
    if not all(a in allowed for a in algos):
        logger.error(
            "Unknown ruptures_algorithm. Valid: %s.", ", ".join(sorted(allowed))
        )
        valid = False
    costs = params['algo'].get('ruptures_cost_params')
    if not isinstance(costs, list) or not costs or not all(
        isinstance(c, dict) and 'model' in c for c in costs
    ):
        logger.error(
            "'ruptures_cost_params' must be a nonempty list of dicts with key 'model'."
        )
        valid = False
    if 'dynp' in algos or 'window' in algos:
        # dynp always needs regimes; window needs one of regimes/penalty/rec_error.
        if 'dynp' in algos and not _valid_model_based_regimes(params):
            valid = False
        if 'window' in algos:
            has_stop = any(
                k in params['algo']
                for k in ('regimes', 'ruptures_penalty', 'ruptures_rec_error')
            )
            if not has_stop:
                logger.error(
                    "window ruptures dimred needs regimes, ruptures_penalty, "
                    "or ruptures_rec_error."
                )
                valid = False
            windows = params['algo'].get('ruptures_window')
            if not isinstance(windows, list) or not all(
                isinstance(x, int) and x > 0 for x in windows
            ):
                logger.error("'ruptures_window' must be nonempty list of positive ints.")
                valid = False
    return valid


def _valid_tft_dimred_params(params: dict) -> bool:
    if 'tft' not in params['algo']['dimred']:
        return True
    valid = True
    ranks = params['algo'].get('rank')
    if ranks is None:
        logger.error("'rank' must be provided when using 'tft' dimred.")
        valid = False
    elif not _valid_user_specified_rank(params):
        valid = False
    for key, kind in (
        ('tft_encoder_length', int),
        ('tft_prediction_length', int),
        ('tft_max_epochs', int),
    ):
        vals = params['algo'].get(key)
        if vals is None:
            continue
        if not isinstance(vals, list) or not all(
            isinstance(x, kind) and x > 0 for x in vals
        ):
            logger.error("'%s' must be a list of positive %ss.", key, kind.__name__)
            valid = False
    modes = params['algo'].get('tft_mode', ['univariate'])
    if not isinstance(modes, list) or not all(
        m in ('univariate', 'multivariate') for m in modes
    ):
        logger.error("'tft_mode' must be list of 'univariate'/'multivariate'.")
        valid = False
    return valid


def _valid_dimred_params(params: dict) -> bool:
    if not _provided_dimred(params):
        return False
    handlers = {
        'pca': _valid_pca_params,
        'dmd': _valid_dmd_params,
        'raw': _valid_raw_dimred_params,
        'kpca': _valid_kpca_params,
        'tsne': _valid_tsne_params,
        'umap': _valid_umap_params,
        'bayesian_gmm': _valid_bayesian_gmm_dimred_params,
        'hmm': _valid_hmm_dimred_params,
        'kmeans': _valid_kmeans_dimred_params,
        'hdbscan': _valid_hdbscan_dimred_params,
        'optics': _valid_optics_dimred_params,
        'msar': _valid_msar_dimred_params,
        'wasserstein': _valid_wasserstein_dimred_params,
        'ruptures': _valid_ruptures_dimred_params,
        'tft': _valid_tft_dimred_params,
    }
    invalid_methods = []
    for dimred in params['algo']['dimred']:
        handler = handlers.get(dimred)
        if not handler:
            logger.error("Unknown dimred method %s.", dimred)
            invalid_methods.append(dimred)
            continue
        if not handler(params):
            invalid_methods.append(dimred)
            continue
    if invalid_methods:
        return False
    return True
