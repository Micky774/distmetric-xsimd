def get_distance_metric(metric, dtype, **metric_kwargs):
    # A bit of a hack to allow for partial import during build
    try:
        from ._dist_metrics import get_distance_metric

        return get_distance_metric(metric, dtype, **metric_kwargs)
    except ModuleNotFoundError:
        return None


def avx_available():
    try:
        from ._dist_metrics import avx_available

        return avx_available()
    except ModuleNotFoundError:
        return None


def avx512f_available():
    try:
        from ._dist_metrics import avx512f_available

        return avx512f_available()
    except ModuleNotFoundError:
        return None


def get_available_architectures():
    try:
        from ._dist_metrics import get_available_architectures

        return get_available_architectures()
    except ModuleNotFoundError:
        return None


__version__ = "0.2.dev0"

__all__ = [
    "get_distance_metric",
    "avx_available",
    "avx512f_available",
    "get_available_architectures",
]
