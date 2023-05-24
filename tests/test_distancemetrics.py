from distance_metrics import get_distance_metric
from sklearn.metrics._dist_metrics import DistanceMetric, DistanceMetric32
import numpy as np
from numpy.testing import assert_allclose
import pytest

IMPLEMENTED_METRICS = ("euclidean", "manhattan", "chebyshev")
DISTANCE_METRIC_SK = {
    "64": DistanceMetric,
    "32": DistanceMetric32,
}


@pytest.mark.parametrize("metric", IMPLEMENTED_METRICS)
@pytest.mark.parametrize("bit_width", ("32", "64"))
def test_metric_matches(metric, bit_width):
    rng = np.random.default_rng(42)
    data_dtype = np.float32 if bit_width == "32" else np.float64
    X = rng.random(size=(20, 20), dtype=data_dtype)

    dst = get_distance_metric(X, metric)
    dst_sk = DISTANCE_METRIC_SK[bit_width].get_metric(metric)

    assert_allclose(dst.pairwise(X), dst_sk.pairwise(X), atol=np.finfo(data_dtype).eps)
