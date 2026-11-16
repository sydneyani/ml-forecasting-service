"""Seasonal naive forecasts read straight from the observed history.

The prediction for date t at lag k is the load on date t-k. This is computed
from the calendar date, not from row position, and it is the number the
boosted model has to beat.
"""

import numpy as np
import pandas as pd

from forecasting.features.build import ForecastInputError


def lagged_actual(history: pd.DataFrame, target_dates: pd.Series, lag: int) -> np.ndarray:
    if lag < 1:
        raise ValueError("lag must be at least 1")
    if "date" not in history.columns or "consumption" not in history.columns:
        raise ForecastInputError("history must include date and consumption columns")

    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(history["date"]).dt.normalize(),
            "consumption": pd.to_numeric(history["consumption"], errors="coerce"),
        }
    )
    if frame["date"].duplicated().any():
        raise ForecastInputError("duplicate dates in history")

    lookup = dict(zip(frame["date"], frame["consumption"], strict=True))
    predictions = np.empty(len(target_dates), dtype=np.float64)
    for index, day in enumerate(pd.to_datetime(target_dates)):
        key = pd.Timestamp(day).normalize() - pd.Timedelta(days=lag)
        value = lookup.get(pd.Timestamp(key), np.nan)
        predictions[index] = np.nan if pd.isna(value) else float(value)
    return predictions
