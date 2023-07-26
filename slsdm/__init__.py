def get_distance_metric(X, metric, **metric_kwargs):
    # A bit of a hack to allow for partial import during build
    try:
        from ._dist_metrics import get_distance_metric

        return get_distance_metric(X, metric, **metric_kwargs)
    except ModuleNotFoundError:
        return None


__version__ = "0.2.1"


def get_best_arch():
    try:
        from ._dist_metrics import get_best_arch

        return get_best_arch()
    except ModuleNotFoundError:
        return None


__all__ = ["get_distance_metric", "get_best_arch"]
