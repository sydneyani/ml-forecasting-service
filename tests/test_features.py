import numpy as np
import pandas as pd
import pytest

from forecasting.data.dataset import load_daily_load
from forecasting.features.build import (
    FEATURE_SCHEMA_VERSION,
    build_feature_frame,
    feature_columns,
    inference_features,
    supervised_frame,
)


def _toy_frame(n_days: int = 40) -> pd.DataFrame:
    dates = pd.date_range("2021-01-01", periods=n_days, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "consumption": np.linspace(1000, 1100, n_days),
            "wind": np.linspace(50, 80, n_days),
            "solar": np.linspace(5, 20, n_days),
        }
    )


def test_feature_list_is_the_training_contract():
    assert FEATURE_SCHEMA_VERSION == 1
    assert feature_columns() == [
        "consumption_lag_1",
        "consumption_lag_2",
        "consumption_lag_3",
        "consumption_lag_7",
        "consumption_lag_14",
        "consumption_lag_21",
        "consumption_lag_28",
        "consumption_lag_364",
        "consumption_roll_mean_7",
        "consumption_roll_std_7",
        "consumption_roll_mean_14",
        "consumption_roll_std_14",
        "consumption_roll_mean_28",
        "consumption_roll_std_28",
        "wind_lag_1",
        "wind_lag_7",
        "solar_lag_1",
        "solar_lag_7",
        "dow_0",
        "dow_1",
        "dow_2",
        "dow_3",
        "dow_4",
        "dow_5",
        "dow_6",
        "month_1",
        "month_2",
        "month_3",
        "month_4",
        "month_5",
        "month_6",
        "month_7",
        "month_8",
        "month_9",
        "month_10",
        "month_11",
        "month_12",
        "dayofyear",
        "is_holiday",
    ]


def test_lags_and_rolling_windows_exclude_the_current_day():
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=30, freq="D"),
            "consumption": np.arange(1, 31, dtype=float),
        }
    )
    featured = build_feature_frame(frame)
    # Row 10 is 2020-01-11, consumption 11. The previous 7 values are 4..10.
    assert featured.loc[10, "consumption_lag_1"] == pytest.approx(10)
    assert featured.loc[10, "consumption_lag_7"] == pytest.approx(4)
    assert featured.loc[10, "consumption_roll_mean_7"] == pytest.approx(7)


def test_lags_follow_the_calendar_when_a_day_is_missing():
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    frame = pd.DataFrame({"date": dates, "consumption": np.arange(10, dtype=float)})
    frame = frame.loc[frame["date"] != "2020-01-05"].copy()
    featured = build_feature_frame(frame)
    row = featured.loc[featured["date"] == "2020-01-06"].iloc[0]
    assert pd.isna(row["consumption_lag_1"])
    assert row["consumption_lag_2"] == pytest.approx(3)


def test_calendar_dummies_and_german_holidays(opsd_csv):
    featured = build_feature_frame(load_daily_load(opsd_csv))
    dow_columns = [f"dow_{day}" for day in range(7)]
    month_columns = [f"month_{month}" for month in range(1, 13)]
    assert featured[dow_columns].sum(axis=1).eq(1).all()
    assert featured[month_columns].sum(axis=1).eq(1).all()

    christmas = featured.loc[featured["date"] == "2017-12-25"].iloc[0]
    assert christmas["is_holiday"] == 1
    assert christmas["dow_0"] == 1
    assert featured.loc[featured["date"] == "2017-01-01", "is_holiday"].iloc[0] == 1
    assert featured.loc[featured["date"] == "2017-12-26", "is_holiday"].iloc[0] == 1
    assert featured.loc[featured["date"] == "2017-12-24", "is_holiday"].iloc[0] == 0
    assert featured.loc[featured["date"] == "2017-07-11", "is_holiday"].iloc[0] == 0


def test_same_day_actuals_do_not_change_that_days_features():
    frame = _toy_frame()
    original = build_feature_frame(frame)
    mutated = frame.copy()
    mutated.loc[mutated.index[-1], "consumption"] = 99999
    mutated.loc[mutated.index[-1], "wind"] = 0
    mutated.loc[mutated.index[-1], "solar"] = 0
    changed = build_feature_frame(mutated)
    columns = feature_columns()
    pd.testing.assert_frame_equal(original.iloc[:-1][columns], changed.iloc[:-1][columns])
    pd.testing.assert_frame_equal(
        original.iloc[[-1]][columns].reset_index(drop=True),
        changed.iloc[[-1]][columns].reset_index(drop=True),
    )

    mutated = frame.copy()
    mutated.loc[10, "consumption"] = 12345
    changed = build_feature_frame(mutated)
    assert changed.iloc[11]["consumption_lag_1"] == pytest.approx(12345)


def test_supervised_rows_keep_missing_solar_and_drop_the_lag_warmup(opsd_csv):
    raw = load_daily_load(opsd_csv)
    supervised = supervised_frame(raw)
    assert len(supervised) == len(raw) - 364
    assert supervised["date"].iloc[0] == raw["date"].iloc[0] + pd.Timedelta(days=364)
    assert supervised["solar_lag_1"].isna().any()
    assert supervised["consumption"].notna().all()


def test_wind_lag_matches_the_previous_calendar_day(opsd_csv):
    raw = load_daily_load(opsd_csv)
    featured = build_feature_frame(raw)
    day = pd.Timestamp("2017-06-15")
    previous = day - pd.Timedelta(days=1)
    row = featured.loc[featured["date"] == day].iloc[0]
    expected = raw.loc[raw["date"] == previous, "wind"].iloc[0]
    assert row["wind_lag_1"] == pytest.approx(expected)
    assert "wind" not in feature_columns()
    assert "consumption" not in feature_columns()


def test_inference_row_matches_the_training_row_and_ignores_later_actuals(opsd_csv):
    raw = load_daily_load(opsd_csv)
    featured = build_feature_frame(raw)
    target = pd.Timestamp("2017-06-15")
    expected = featured.loc[featured["date"] == target].iloc[0]
    row = inference_features(raw, target)
    for column in feature_columns():
        if pd.isna(expected[column]):
            assert pd.isna(row[column])
        else:
            assert row[column] == pytest.approx(expected[column])

    next_day = raw["date"].max() + pd.Timedelta(days=1)
    future = inference_features(raw, next_day)
    assert future["consumption_lag_1"] == pytest.approx(raw.iloc[-1]["consumption"])
