"""Split-conformal intervals from absolute residuals.

`q_hat` is the k-th smallest calibration residual, with
k = ceil((n + 1) * (1 - alpha)). When k is inside 1..n, the interval
[yhat - q_hat, yhat + q_hat] has finite-sample marginal coverage at least
1 - alpha under exchangeable residuals (split conformal; see Angelopoulos
and Bates). Load residuals are not exchangeable across seasons, so the
backtest also publishes the empirical coverage on the test year.
"""

import math

import numpy as np


def conformal_rank(n_calibration: int, alpha: float) -> int:
    if n_calibration < 1:
        raise ValueError("calibration set is empty")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    return math.ceil((n_calibration + 1) * (1.0 - alpha))


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Return the split-conformal quantile of non-negative scores.

    The result is infinite when the finite-sample rank falls past the last
    calibration score. Callers should treat that as "the calibration set is
    too small for this alpha", not as a usable interval.
    """
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("scores must be a non-empty one-dimensional array")
    if not np.isfinite(values).all():
        raise ValueError("scores must be finite")
    if np.any(values < 0):
        raise ValueError("conformal scores must be non-negative")
    ordered = np.sort(values)
    rank = conformal_rank(ordered.size, alpha)
    if rank > ordered.size:
        return float("inf")
    return float(ordered[rank - 1])
