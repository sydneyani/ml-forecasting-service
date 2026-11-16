import numpy as np

from forecasting.data.dataset import load_daily_load
from forecasting.features.build import supervised_frame
from forecasting.models.baselines import lagged_actual


def test_lagged_actual_matches_calendar_lag_features(opsd_csv):
    raw = load_daily_load(opsd_csv)
    supervised = supervised_frame(raw)
    for lag in (1, 7, 364):
        predicted = lagged_actual(raw, supervised["date"], lag)
        assert np.allclose(predicted, supervised[f"consumption_lag_{lag}"].to_numpy())
