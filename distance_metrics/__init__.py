from ._dist_metrics import get_distance_metric
from ._config import global_config

ARCHITECTURES = global_config["archs"]

__all__ = ["get_distance_metric", "ARCHITECTURES"]
