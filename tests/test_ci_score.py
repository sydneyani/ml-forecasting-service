"""Scores the fast model produced by configs/ci.yaml. Skipped until that file exists."""

from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from forecasting.data.dataset import load_daily_load
from forecasting.serving.app import create_app

ROOT = Path(__file__).resolve().parents[1]


def test_ci_model_returns_a_plausible_load():
    model_dir = ROOT / "models" / "ci"
    if not (model_dir / "metadata.json").is_file():
        pytest.skip("configs/ci.yaml has not been trained in this workspace")
    raw = load_daily_load(ROOT / "data" / "raw" / "opsd_germany_daily.csv")
    window = raw.loc[raw["date"] < "2017-12-31"].tail(400)
    history = []
    for row in window.itertuples(index=False):
        history.append(
            {
                "date": row.date.date().isoformat(),
                "consumption": float(row.consumption),
                "wind": None if pd.isna(row.wind) else float(row.wind),
                "solar": None if pd.isna(row.solar) else float(row.solar),
            }
        )
    client = TestClient(create_app(model_dir))
    response = client.post(
        "/predict",
        json={"target_date": "2017-12-31", "history": history},
    )
    assert response.status_code == 200
    prediction = response.json()["prediction_gwh"]
    assert 700 < prediction < 2200
