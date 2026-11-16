"""Point-forecast scores.

MAE and RMSE are in the same unit as the target (GWh per day). MAPE is a percent:
100 times the mean of |y - yhat| / |y|.
"""

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    if np.any(y_true == 0):
        raise ValueError("MAPE is undefined when an observed value is zero")
    return float(mean_absolute_percentage_error(y_true, y_pred) * 100.0)


def _as_1d(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return array


def regression_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = _as_1d("y_true", y_true)
    y_pred = _as_1d("y_pred", y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred shapes differ")
    if y_true.size == 0:
        raise ValueError("cannot score an empty prediction")
    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError("scores require finite values")
    return {"mae": mae(y_true, y_pred), "rmse": rmse(y_true, y_pred), "mape": mape(y_true, y_pred)}


def mae_ratio(model_mae: float, baseline_mae: float) -> float:
    """Model MAE divided by baseline MAE. Values below 1 mean the model is closer."""
    if baseline_mae <= 0:
        raise ValueError("baseline MAE must be positive")
    return float(model_mae / baseline_mae)
