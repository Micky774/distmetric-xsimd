from slsdm import get_distance_metric
from sklearn.metrics._dist_metrics import DistanceMetric, DistanceMetric32
import numpy as np
from numpy.testing import assert_allclose
import pytest

# Note we use `p` to test for the usual minkowski, and `minkowski` to instead
# test for the weighted minkowski metric since `wminkowski` is an entirely
# separate metric.
IMPLEMENTED_METRICS = (
    "euclidean",
    "manhattan",
    "chebyshev",
    "minkowski",
    "p",
    "seuclidean",
)
DISTANCE_METRIC_SK = {
    "64": DistanceMetric,
    "32": DistanceMetric32,
}


@pytest.mark.parametrize("metric", IMPLEMENTED_METRICS)
@pytest.mark.parametrize("bit_width", ("32", "64"))
def test_metric_matches(metric, bit_width):
    rng = np.random.default_rng(42)
    n_samples = 300
    n_features = 200
    metric_kwargs = {
        "p": {"p": 14},
        "minkowski": {"p": 3, "w": rng.uniform(1, 10, size=(n_features))},
        "seuclidean": {"V": rng.uniform(1, 10, size=(n_features))},
    }.get(metric, {})
    data_dtype = np.float32 if bit_width == "32" else np.float64
    X = rng.random(size=(n_samples, n_features), dtype=data_dtype)

    dst = get_distance_metric(X, metric, **metric_kwargs)
    dst_sk = DISTANCE_METRIC_SK[bit_width].get_metric(metric, **metric_kwargs)
    print(dst_sk)

    pairs = dst.pairwise(X)
    pairs_sk = dst_sk.pairwise(X)

    assert_allclose(pairs, pairs_sk, atol=np.finfo(data_dtype).eps, rtol=3e-7)
