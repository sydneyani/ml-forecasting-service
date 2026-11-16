import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from forecasting.data.dataset import load_daily_load
from forecasting.serving.app import create_app

ROOT = Path(__file__).resolve().parents[1]


def _history(target: str, days: int | None = None) -> list[dict]:
    raw = load_daily_load(ROOT / "data" / "raw" / "opsd_germany_daily.csv")
    window = raw.loc[raw["date"] < target]
    if days is not None:
        window = window.tail(days)
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
    return history


def test_health_and_ready_without_a_model(tmp_path):
    client = TestClient(create_app(tmp_path))
    assert client.get("/health").json() == {"status": "ok"}
    ready = client.get("/ready")
    assert ready.status_code == 503
    assert (
        client.post("/predict", json={"target_date": "2017-12-31", "history": []}).status_code
        == 422
    )


def test_champion_prediction_matches_the_holdout_file():
    client = TestClient(create_app(ROOT / "models" / "champion"))
    ready = client.get("/ready")
    assert ready.status_code == 200
    body = ready.json()
    assert body["model_type"] == "xgboost"
    assert body["model_version"] == "5828f2d8d5a4"
    assert body["target_unit"] == "GWh"

    response = client.post(
        "/predict",
        json={"target_date": "2017-12-31", "history": _history("2017-12-31", days=400)},
    )
    assert response.status_code == 200
    scored = response.json()
    predictions = pd.read_csv(ROOT / "reports" / "test_predictions.csv")
    saved = predictions.loc[predictions["date"] == "2017-12-31"].iloc[0]
    assert scored["prediction_gwh"] == pytest.approx(saved["y_pred"], abs=1e-4)
    assert scored["interval_low_gwh"] == pytest.approx(saved["y_low"], abs=1e-4)
    assert scored["interval_high_gwh"] == pytest.approx(saved["y_high"], abs=1e-4)
    assert scored["interval_low_gwh"] <= scored["prediction_gwh"] <= scored["interval_high_gwh"]

    batch = client.post(
        "/predict/batch",
        json={
            "target_dates": ["2017-07-15", "2017-12-31"],
            "history": _history("2017-12-31"),
        },
    )
    assert batch.status_code == 200
    payload = batch.json()
    assert [item["target_date"] for item in payload["predictions"]] == [
        "2017-07-15",
        "2017-12-31",
    ]
    assert payload["model_version"] == body["model_version"]


def test_short_history_and_bad_inputs_are_rejected():
    client = TestClient(create_app(ROOT / "models" / "champion"))
    short = client.post(
        "/predict",
        json={
            "target_date": "2017-12-31",
            "history": _history("2017-12-31", days=30),
        },
    )
    assert short.status_code == 422
    assert "Not enough history" in short.json()["detail"]

    negative = client.post(
        "/predict",
        json={
            "target_date": "2017-12-31",
            "history": [{"date": "2017-12-30", "consumption": -1}],
        },
    )
    assert negative.status_code == 422

    duplicate = client.post(
        "/predict",
        json={
            "target_date": "2017-12-31",
            "history": _history("2017-12-31")
            + [{"date": "2017-12-30", "consumption": 1200.0, "wind": 10.0, "solar": 1.0}],
        },
    )
    assert duplicate.status_code == 422
    assert "duplicate dates" in duplicate.json()["detail"]


def test_metadata_version_constant_is_the_file_on_disk():
    metadata = json.loads((ROOT / "models" / "champion" / "metadata.json").read_text())
    assert metadata["model_version"] == "5828f2d8d5a4"
    assert date.fromisoformat("2017-12-31")
