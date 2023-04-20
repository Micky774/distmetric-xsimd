from distance_metrics import SimdDistanceMetric, SimdDistanceMetric32
from sklearn.metrics._dist_metrics import DistanceMetric, DistanceMetric32
import numpy as np
from numpy.testing import assert_allclose
import pytest

IMPLEMENTED_METRICS = ('euclidean', 'manhattan')
DISTANCE_METRIC = {
    '64':(SimdDistanceMetric, DistanceMetric),
    '32':(SimdDistanceMetric32, DistanceMetric32),
    }

@pytest.mark.parametrize("metric", IMPLEMENTED_METRICS)
@pytest.mark.parametrize("bit_width", ('32', '64'))
def test_metric_matches_sklearn(metric, bit_width):
    rng = np.random.default_rng(42)
    X = rng.random(size=(20, 20))
    dst_parent, dst_sk_parent = DISTANCE_METRIC[bit_width]
    dst = dst_parent.get_metric(metric)
    dst_sk = dst_sk_parent.get_metric(metric)
    assert_allclose(dst.pairwise(X), dst_sk.pairwise(X))