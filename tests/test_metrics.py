import numpy as np
import pytest

from forecasting.evaluation.metrics import mae_ratio, regression_scores


def test_scores_match_hand_calculated_errors():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 200.0, 270.0])
    scores = regression_scores(y_true, y_pred)
    assert scores["mae"] == pytest.approx((10 + 0 + 30) / 3)
    assert scores["rmse"] == pytest.approx(np.sqrt((100 + 0 + 900) / 3))
    assert scores["mape"] == pytest.approx(100 * (0.1 + 0 + 0.1) / 3)


def test_mape_rejects_a_zero_observation():
    with pytest.raises(ValueError, match="undefined"):
        regression_scores(np.array([0.0, 1.0]), np.array([1.0, 1.0]))


def test_mae_ratio():
    assert mae_ratio(80, 100) == pytest.approx(0.8)
    with pytest.raises(ValueError):
        mae_ratio(1, 0)
