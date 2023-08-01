def get_distance_metric(metric, dtype, **metric_kwargs):
    # A bit of a hack to allow for partial import during build
    try:
        from ._dist_metrics import get_distance_metric

        return get_distance_metric(metric, dtype, **metric_kwargs)
    except ModuleNotFoundError:
        return None


__version__ = "0.2.dev0"


__all__ = ["get_distance_metric"]
