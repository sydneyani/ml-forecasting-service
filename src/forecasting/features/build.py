"""Features for a day-ahead load forecast.

The forecast origin for target date t is the end of day t-1. Every model input
on row t is either a calendar fact about day t, which is known ahead of time,
or a transformation of load, wind, and solar observed on days strictly before t.

Same-day wind and solar in this file are generation actuals, not a published
day-ahead forecast, so they are not model inputs. Rolling windows are computed
on load that has already been shifted by one day, which keeps day t's load out
of its own features. Missing wind and solar stay missing: filling them with
zero would look like a real zero-generation day in the years before those
series exist.
"""

import holidays
import numpy as np
import pandas as pd

FEATURE_SCHEMA_VERSION = 1
HORIZON_DAYS = 1
CONSUMPTION_LAGS = (1, 2, 3, 7, 14, 21, 28, 364)
ROLL_WINDOWS = (7, 14, 28)
EXOGENOUS_LAGS = (1, 7)
MIN_HISTORY_DAYS = max(CONSUMPTION_LAGS)


class ForecastInputError(ValueError):
    """The history or target date cannot be turned into model features."""


def feature_columns() -> list[str]:
    """Column order of the model matrix. Training and serving both use this list."""
    columns = [f"consumption_lag_{lag}" for lag in CONSUMPTION_LAGS]
    for window in ROLL_WINDOWS:
        columns.append(f"consumption_roll_mean_{window}")
        columns.append(f"consumption_roll_std_{window}")
    for name in ("wind", "solar"):
        for lag in EXOGENOUS_LAGS:
            columns.append(f"{name}_lag_{lag}")
    columns.extend(f"dow_{day}" for day in range(7))
    columns.extend(f"month_{month}" for month in range(1, 13))
    columns.append("dayofyear")
    columns.append("is_holiday")
    return columns


def consumption_feature_columns() -> list[str]:
    return [column for column in feature_columns() if column.startswith("consumption_")]


def assert_feature_contract(metadata: dict) -> None:
    """Refuse an artifact built with a different feature schema."""
    version = metadata.get("feature_schema_version")
    if version != FEATURE_SCHEMA_VERSION:
        raise RuntimeError(
            f"artifact feature schema version is {version}, code expects {FEATURE_SCHEMA_VERSION}"
        )
    columns = metadata.get("feature_columns")
    if list(columns or []) != feature_columns():
        raise RuntimeError(
            "artifact feature columns do not match this code revision; retrain the champion"
        )


def _numeric_column(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns:
        return np.full(len(frame), np.nan, dtype=np.float64)
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)


def _reject_negative(values: np.ndarray, name: str) -> None:
    finite = values[np.isfinite(values)]
    if (finite < 0).any():
        raise ForecastInputError(f"{name} must be non-negative")


def prepare_daily(frame: pd.DataFrame) -> pd.DataFrame:
    """Sort by calendar day and insert any missing days as empty rows.

    Lags are then shifts on a daily index, so a missing day cannot silently
    become "yesterday".
    """
    if frame is None or len(frame) == 0:
        raise ForecastInputError("history is empty")
    if "date" not in frame.columns or "consumption" not in frame.columns:
        raise ForecastInputError("history must include date and consumption columns")

    parsed = pd.to_datetime(frame["date"], errors="coerce")
    if parsed.isna().any():
        raise ForecastInputError("history contains an unparseable date")
    if getattr(parsed.dt, "tz", None) is not None:
        raise ForecastInputError("dates must be timezone-naive calendar days")

    consumption = _numeric_column(frame, "consumption")
    wind = _numeric_column(frame, "wind")
    solar = _numeric_column(frame, "solar")
    _reject_negative(consumption, "consumption")
    _reject_negative(wind, "wind")
    _reject_negative(solar, "solar")

    out = pd.DataFrame(
        {
            "date": parsed.dt.normalize().to_numpy(),
            "consumption": consumption,
            "wind": wind,
            "solar": solar,
        }
    )
    duplicated = out["date"].duplicated()
    if duplicated.any():
        examples = out.loc[duplicated, "date"].dt.strftime("%Y-%m-%d").unique()[:5]
        raise ForecastInputError(f"duplicate dates in history: {list(examples)}")

    out = out.sort_values("date")
    full_index = pd.date_range(out["date"].min(), out["date"].max(), freq="D")
    out = out.set_index("date").reindex(full_index)
    out.index.name = "date"
    return out.reset_index()


def _german_holiday_flags(dates: pd.Series) -> np.ndarray:
    start_year = int(dates.dt.year.min())
    end_year = int(dates.dt.year.max())
    calendar = holidays.country_holidays("DE", years=range(start_year, end_year + 1))
    return np.asarray([int(day in calendar) for day in dates.dt.date], dtype=np.int64)


def build_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per calendar day, including rows whose features are still incomplete."""
    base = prepare_daily(frame)
    featured = base.copy()
    consumption = base["consumption"]
    for lag in CONSUMPTION_LAGS:
        featured[f"consumption_lag_{lag}"] = consumption.shift(lag)
    # shift(1) drops the current day before the window is applied.
    shifted = consumption.shift(1)
    for window in ROLL_WINDOWS:
        rolling = shifted.rolling(window, min_periods=window)
        featured[f"consumption_roll_mean_{window}"] = rolling.mean()
        featured[f"consumption_roll_std_{window}"] = rolling.std()
    for name in ("wind", "solar"):
        for lag in EXOGENOUS_LAGS:
            featured[f"{name}_lag_{lag}"] = base[name].shift(lag)

    dates = featured["date"]
    day_of_week = dates.dt.day_of_week.to_numpy()
    month = dates.dt.month.to_numpy()
    for day in range(7):
        featured[f"dow_{day}"] = (day_of_week == day).astype(np.int64)
    for month_number in range(1, 13):
        featured[f"month_{month_number}"] = (month == month_number).astype(np.int64)
    featured["dayofyear"] = dates.dt.day_of_year.astype(np.int64)
    featured["is_holiday"] = _german_holiday_flags(dates)
    return featured


def supervised_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows that have a load target and a complete load-feature history.

    Wind and solar lags may still be missing. Tree models accept that missingness.
    """
    featured = build_feature_frame(frame)
    required = consumption_feature_columns() + ["consumption"]
    return featured.dropna(subset=required).reset_index(drop=True)


def design_matrix(frame: pd.DataFrame) -> np.ndarray:
    columns = feature_columns()
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ForecastInputError(f"feature frame is missing columns: {missing}")
    return frame.loc[:, columns].to_numpy(dtype=np.float64)


def missing_required_features(row: pd.Series) -> list[str]:
    """Load and calendar inputs that are missing. Exogenous lags may be missing."""
    missing: list[str] = []
    for column in feature_columns():
        if column.startswith(("wind_", "solar_")):
            continue
        if pd.isna(row[column]):
            missing.append(column)
    return missing


def inference_features(history: pd.DataFrame, target_date: object) -> pd.Series:
    """Build the feature row for `target_date` using observations strictly before it.

    Observations on or after the target date are dropped before features are built,
    so a caller can pass a frame that already contains the eventual actual.
    """
    try:
        target = pd.Timestamp(target_date)
    except (TypeError, ValueError) as exc:
        raise ForecastInputError(f"invalid target date: {target_date}") from exc
    if pd.isna(target):
        raise ForecastInputError("invalid target date")
    if target.tzinfo is not None:
        raise ForecastInputError("dates must be timezone-naive calendar days")
    target = target.normalize()

    known = history.copy()
    known["date"] = pd.to_datetime(known["date"], errors="coerce")
    known = known.loc[known["date"] < target]
    if known.empty:
        raise ForecastInputError("history has no observations before the target date")

    keep = [
        column for column in ("date", "consumption", "wind", "solar") if column in known.columns
    ]
    placeholder = pd.DataFrame(
        {"date": [target], "consumption": [np.nan], "wind": [np.nan], "solar": [np.nan]}
    )
    featured = build_feature_frame(pd.concat([known.loc[:, keep], placeholder], ignore_index=True))
    rows = featured.loc[featured["date"] == target]
    if rows.empty:
        raise ForecastInputError("target date was not found after feature construction")
    row = rows.iloc[0]
    missing = missing_required_features(row)
    if missing:
        preview = ", ".join(missing[:6])
        raise ForecastInputError(
            f"Not enough history before {target.date().isoformat()} to build {preview}. "
            f"Provide at least {MIN_HISTORY_DAYS} uninterrupted daily load observations "
            "before the target date."
        )
    return row
