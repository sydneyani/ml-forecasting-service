"""Germany daily electricity load, wind, and solar.

The vendored file is `opsd_germany_daily.csv` from https://github.com/jenfly/opsd
(SHA-256 pinned in this module). The companion notebook
`time-series-preprocessing.ipynb` builds it from Open Power System Data hourly
series for Germany, complete years 2006-2017:

- hourly megawatts are converted to gigawatts
- days are summed to energy, so each value is gigawatt-hours per day
- Consumption is ENTSO-E power-statistics load
- Wind and Solar are actual generation (not a day-ahead forecast)

Wind is absent early in the file and solar is absent until the early 2010s.
Those gaps stay missing in the model matrix. Upstream OPSD series can carry
source-specific terms; see https://open-power-system-data.org/legal/ and cite
Open Power System Data, Data Package Time series.
"""

from pathlib import Path

import pandas as pd

OPSD_DAILY_URL = "https://raw.githubusercontent.com/jenfly/opsd/master/opsd_germany_daily.csv"
OPSD_SHA256 = "729b025d984497827cb6b89048d7c2c6a551e7aa727bc47ea0b74a2d1e84675d"
DATASET_NAME = "OPSD Germany daily electricity"
TARGET_COLUMN = "consumption"
TARGET_UNIT = "GWh"
OBSERVED_COLUMNS = ("date", "consumption", "wind", "solar")


def load_daily_load(path: Path) -> pd.DataFrame:
    """Return observed daily rows sorted by date.

    Columns are `date`, `consumption`, `wind`, and `solar`. The `Wind+Solar`
    total in the raw file is dropped because it repeats the other two series.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"daily load file not found: {path}")

    frame = pd.read_csv(path)
    expected = ["Date", "Consumption", "Wind", "Solar"]
    missing = [column for column in expected if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing columns {missing}")

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(frame["Date"], format="%Y-%m-%d"),
            "consumption": pd.to_numeric(frame["Consumption"], errors="coerce"),
            "wind": pd.to_numeric(frame["Wind"], errors="coerce"),
            "solar": pd.to_numeric(frame["Solar"], errors="coerce"),
        }
    )
    if out["date"].isna().any() or out["consumption"].isna().any():
        raise ValueError(f"{path} has an unparseable date or consumption value")
    if out["date"].duplicated().any():
        raise ValueError(f"{path} has duplicate dates")
    if (out["consumption"] < 0).any():
        raise ValueError(f"{path} has negative consumption")
    return out.sort_values("date").reset_index(drop=True)


def calendar_gaps(frame: pd.DataFrame) -> pd.DatetimeIndex:
    """Calendar days missing between the first and last date."""
    span = pd.date_range(frame["date"].min(), frame["date"].max(), freq="D")
    return span.difference(pd.DatetimeIndex(frame["date"]))
