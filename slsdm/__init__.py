def get_distance_metric(X, metric, **metric_kwargs):
    # A bit of a hack to allow for partial import during build
    try:
        from ._dist_metrics import get_distance_metric

        return get_distance_metric(X, metric, **metric_kwargs)
    except ModuleNotFoundError:
        return None


__all__ = ["get_distance_metric"]
