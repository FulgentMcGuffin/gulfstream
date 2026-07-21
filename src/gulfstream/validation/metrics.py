"""Input validation for the 'metrics' section of the params dict."""
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _provided_metrics(params: dict) -> bool:
    metrics = params.get('metrics')
    if not metrics:
        logger.error("Must provide 'metrics' (dict).")
        return False
    elif not isinstance(metrics, dict):
        logger.error("'metrics' must be a dict. Got type %s.", type(metrics))
        return False
    return True


def _valid_mode(params: dict) -> bool:
    mode = params['metrics'].get('mode')
    if not mode:
        logger.error(
            "Must specify 'mode' (str) in 'metrics'. Valid options are 'display', "
            "'write', and 'display_and_write'."
        )
        return False
    elif mode not in ['display', 'write', 'display_and_write']:
        logger.error("'mode' must be 'display', 'write', or 'display_and_write'. Got %s.", mode)
        return False
    return True


def _valid_plot(params: dict) -> bool:
    plot = params['metrics'].get('plot')
    if plot is None:
        logger.error("Must specify 'plot' (bool) in 'metrics'.")
        return False
    elif not isinstance(plot, bool):
        logger.error("'plot' must be bool, got %s.", type(plot))
        return False
    return True


def _valid_num_samples(params: dict) -> bool:
    plot = params['metrics']['plot']
    if not plot:
        return True
    num_samples = params['metrics'].get('num_samples')
    if not num_samples:
        logger.error("Must provide 'num_samples' (int) in 'metrics' if 'plot' is True.")
        return False
    elif not isinstance(num_samples, int):
        logger.error("'num_samples' must be type int. Got type %s.", type(num_samples))
        return False
    elif num_samples < 1:
        logger.error("'num_samples' must be a positive integer. Got %s.", num_samples)
        return False
    return True


def _valid_num_features(params: dict) -> bool:
    plot = params['metrics']['plot']
    if not plot:
        return True
    num_features = params['metrics'].get('num_features')
    if not num_features:
        logger.error("Must provide 'num_features' (int) in 'metrics' if 'plot' is True.")
        return False
    elif not isinstance(num_features, int):
        logger.error("'num_features' must be type int. Got type %s.", type(num_features))
        return False
    elif num_features < 1:
        logger.error("'num_features' must be a positive integer. Got %s.", num_features)
        return False
    return True


def _valid_num_components(params: dict) -> bool:
    plot = params['metrics']['plot']
    if not plot:
        return True
    num_components = params['metrics'].get('num_components')
    if not num_components:
        logger.error("Must provide 'num_components' (int) in 'metrics' if 'plot' is True.")
        return False
    elif not isinstance(num_components, int):
        logger.error("'num_components' must be type int. Got type %s.", type(num_components))
        return False
    elif num_components < 1:
        logger.error("'num_components' must be a positive integer. Got %s.", num_components)
        return False
    return True


def _valid_num_shap(params: dict) -> bool:
    plot = params['metrics']['plot']
    if not plot:
        return True
    num_shap = params['metrics'].get('num_shap')
    if not num_shap:
        logger.error("Must provide 'num_shap' (int) in 'metrics' if 'plot' is True.")
        return False
    elif not isinstance(num_shap, int):
        logger.error("'num_shap' must be type int. Got type %s.", type(num_shap))
        return False
    elif num_shap < 1:
        logger.error("'num_shap' must be a positive integer. Got %s.", num_shap)
        return False
    return True


def _valid_num_top_features_for_distances(params: dict) -> bool:
    plot = params['metrics']['plot']
    if not plot:
        return True
    num_features = params['metrics'].get('num_top_features_for_distances')
    if not num_features:
        logger.error(
            "Must provide 'num_top_features_for_distances' (int) in 'metrics' if 'plot' is True."
        )
        return False
    elif not isinstance(num_features, int):
        logger.error(
            "'num_top_features_for_distances' must be type int. Got type %s.", type(num_features)
        )
        return False
    elif num_features < 1:
        logger.error(
            "'num_top_features_for_distances' must be a positive integer. Got %s.", num_features
        )
        return False
    return True


def _valid_requested_features_for_distances(params: dict) -> bool:
    plot = params['metrics']['plot']
    if not plot:
        return True
    features = params['metrics'].get('requested_features_for_distances')
    if not features:
        # Empty list and None are allowed.
        return True
    elif not isinstance(features, (dict, list)):
        logger.error(
            "'requested_features_for_distances' must be type dict or list[str]. Got %s.",
            type(features)
        )
        return False
    elif isinstance(features, list) and not all(isinstance(x, str) for x in features):
        logger.error(
            "If passing a list for 'requested_features_for_distances', all entries must be type str."
        )
        return False
    return True


def _valid_explainability_features(params: dict) -> bool:
    plot = params['metrics']['plot']
    if not plot:
        return True
    exp_features = params['metrics'].get('explainability_features')
    if not exp_features:
        logger.error(
            "Must provide 'explainability_features' (dict or list[str]) in 'metrics' if 'plot' is True."
        )
        return False
    elif not isinstance(exp_features, (dict, list)):
        logger.error(
            "'explainability_features' must be type dict or list[str]. Got %s.", type(exp_features)
        )
        return False
    elif isinstance(exp_features, list) and not all(isinstance(x, str) for x in exp_features):
        logger.error(
            "If passing a list for 'explainability_features', all entries must be type str."
        )
        return False
    return True


def _valid_features_to_plot(params: dict) -> bool:
    plot = params['metrics']['plot']
    if not plot:
        return True
    features = params['metrics'].get('features_to_plot')
    if not features:
        logger.error(
            "Must provide 'features_to_plot' (list[str]) in 'metrics' if 'plot' is True."
        )
        return False
    elif not isinstance(features, list):
        logger.error("'features_to_plot' must be a list. Got type %s.", type(features))
        return False
    elif not all(isinstance(x, str) for x in features):
        logger.error("All entries of 'features_to_plot' must be strings.")
        return False
    return True


def _valid_warn_threshold(params: dict) -> bool:
    plot = params['metrics']['plot']
    if not plot:
        return True
    threshold = params['metrics'].get('warn_threshold')
    if not threshold:
        logger.error("Must provide 'warn_threshold' (float) in 'metrics' if 'plot' is True.")
        return False
    elif not isinstance(threshold, (int, float)):
        logger.error("'warn_threshold' must be type float. Got type %s.", type(threshold))
        return False
    return True


def _valid_exp_tree_accuracy(params: dict) -> bool:
    plot = params['metrics']['plot']
    if not plot:
        return True
    acc = params['metrics'].get('exp_tree_accuracy')
    if not acc:
        logger.error(
            "Must provide 'exp_tree_accuracy' (list[float]) in 'metrics' if 'plot' is True."
        )
        return False
    elif not isinstance(acc, list):
        logger.error("'exp_tree_accuracy' must be a list of floats. Got type %s.", type(acc))
        return False
    elif not all(isinstance(x, (int, float)) for x in acc):
        logger.error("All entries of 'exp_tree_accuracy' must be floats.")
        return False
    elif not all(0 <= x <= 1 for x in acc):
        logger.error("All entries of 'exp_tree_accuracy' must be between 0 and 1.")
        return False
    return True


def _valid_exp_tree_bps_decimals(params: dict) -> bool:
    plot = params['metrics']['plot']
    if not plot:
        return True
    dec = params['metrics'].get('exp_tree_bps_decimals')
    if dec is None:
        logger.error(
            "Must provide 'exp_tree_bps_decimals' (bool) in 'metrics' if 'plot' is True."
        )
        return False
    elif not isinstance(dec, bool):
        logger.error("'exp_tree_bps_decimals' must be type bool. Got type %s.", type(dec))
        return False
    return True


def _valid_dir(params: dict) -> bool:
    my_dir = params['metrics'].get('dir')
    if not my_dir:
        logger.error("Must specify 'dir' (str) in 'metrics'.")
        return False
    elif not isinstance(my_dir, str):
        logger.error("'dir' in 'metrics' must be type str. Got type %s.", type(my_dir))
        return False
    try:
        # Basic cross-platform sanity check.
        p = Path(my_dir)
        return bool(p.name or p.root)
    except:
        return False


def _valid_regime_cluster_algorithms(params: dict) -> bool:
    algorithms = params["metrics"].get("regime_cluster_algorithms")
    if algorithms is None:
        return True
    if isinstance(algorithms, str):
        algorithms = [algorithms]
    if not isinstance(algorithms, list) or not algorithms:
        logger.error(
            "'regime_cluster_algorithms' must be a nonempty list[str] "
            "(kmeans/hdbscan/optics)."
        )
        return False
    allowed = {"kmeans", "hdbscan", "optics"}
    bad = [a for a in algorithms if str(a).lower() not in allowed]
    if bad:
        logger.error(
            "Unknown regime_cluster_algorithms %s. Valid: %s.",
            bad,
            ", ".join(sorted(allowed)),
        )
        return False
    return True


def _valid_metrics_params(params: dict) -> bool:
    if not _provided_metrics(params):
        return False
    valid = True
    if not _valid_mode(params):
        valid = False
    if not _valid_dir(params):
        valid = False
    if not _valid_plot(params):
        valid = False
    if not _valid_regime_cluster_algorithms(params):
        valid = False
    elif params['metrics']['plot']:
        validators = [
            _valid_num_samples,
            _valid_num_features,
            _valid_num_components,
            _valid_num_shap,
            _valid_num_top_features_for_distances,
            _valid_requested_features_for_distances,
            _valid_explainability_features,
            _valid_features_to_plot,
            _valid_warn_threshold,
            _valid_exp_tree_accuracy,
            _valid_exp_tree_bps_decimals
        ]
        for validator in validators:
            if not validator(params):
                valid = False
    return valid
