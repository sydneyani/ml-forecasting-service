import numpy as np
import pytest

from forecasting.evaluation.conformal import conformal_quantile, conformal_rank


def test_conformal_rank_and_quantile_use_the_finite_sample_order_statistic():
    scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert conformal_rank(5, 0.2) == 5
    assert conformal_quantile(scores, 0.2) == pytest.approx(5.0)
    assert conformal_rank(5, 0.5) == 3
    assert conformal_quantile(np.array([5.0, 1.0, 4.0, 2.0, 3.0]), 0.5) == pytest.approx(3.0)
    assert conformal_rank(90, 0.1) == 82


def test_conformal_quantile_is_infinite_when_the_rank_exceeds_the_sample():
    assert conformal_quantile(np.array([3.0]), 0.1) == float("inf")
