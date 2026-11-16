import json

import numpy as np
import pandas as pd
import pytest

from forecasting.config import (
    CVConfig,
    DataConfig,
    ModelSearchConfig,
    OutputConfig,
    TrainConfig,
)
from forecasting.data.dataset import load_daily_load
from forecasting.features.build import assert_feature_contract, design_matrix, inference_features
from forecasting.models.boosting import load_regressor, predict
from forecasting.training.backtest import format_summary, run_backtest


def _write_synthetic(path, n_days: int = 800) -> None:
    rng = np.random.default_rng(0)
    dates = pd.date_range("2014-01-01", periods=n_days, freq="D")
    weekend = (dates.day_of_week >= 5).astype(float)
    step = np.arange(n_days)
    consumption = (
        1400 + 80 * np.sin(2 * np.pi * step / 365.25) - 160 * weekend + rng.normal(0, 15, n_days)
    )
    wind = 80 + rng.normal(0, 5, n_days)
    solar = 15 + 25 * np.clip(np.sin(2 * np.pi * step / 365.25), 0, None)
    pd.DataFrame(
        {
            "Date": dates.strftime("%Y-%m-%d"),
            "Consumption": consumption,
            "Wind": wind,
            "Solar": solar,
            "Wind+Solar": wind + solar,
        }
    ).to_csv(path, index=False)


def test_backtest_holds_out_test_and_calibration_and_reloads(tmp_path):
    csv_path = tmp_path / "synthetic.csv"
    _write_synthetic(csv_path)
    config = TrainConfig(
        data=DataConfig(csv_path=csv_path, test_days=50, calibration_days=30),
        cv=CVConfig(n_splits=2, valid_days=20, min_train_days=120),
        search=[ModelSearchConfig(model_type="xgboost", n_trials=2, seed=7)],
        fast=True,
        alpha=0.1,
        output=OutputConfig(reports_dir=tmp_path / "reports", model_dir=tmp_path / "model"),
    )
    result = run_backtest(config)

    assert result["splits"]["develop_end"] < result["splits"]["calibration_start"]
    assert result["splits"]["calibration_end"] < result["splits"]["test_start"]
    rows = result["rows"]
    assert rows["develop"] + rows["calibration"] + rows["test"] == rows["supervised"]
    assert result["champion_model_type"] == "xgboost"
    assert result["models"]["xgboost"]["role"] == "champion"
    champion_mae = result["models"]["xgboost"]["test"]["mae"]
    assert champion_mae < result["baselines"]["naive_lag_1"]["test"]["mae"]
    assert 0 < result["interval"]["test_coverage"] <= 1

    predictions = pd.read_csv(tmp_path / "reports" / "test_predictions.csv")
    q_hat = result["interval"]["q_hat"]
    assert np.allclose(predictions["y_low"], predictions["y_pred"] - q_hat)
    assert np.allclose(predictions["y_high"], predictions["y_pred"] + q_hat)
    covered = (predictions["y_true"] >= predictions["y_low"]) & (
        predictions["y_true"] <= predictions["y_high"]
    )
    assert covered.mean() == pytest.approx(result["interval"]["test_coverage"])

    metadata = json.loads((tmp_path / "model" / "metadata.json").read_text())
    assert_feature_contract(metadata)
    summary = format_summary(result)
    assert "Champion: xgboost" in summary
    assert "seasonal_naive_lag_7" in summary

    raw = load_daily_load(csv_path)
    target = pd.Timestamp(predictions["date"].iloc[len(predictions) // 2])
    row = inference_features(raw, target)
    model_path = tmp_path / "model" / metadata["model_filename"]
    reloaded = load_regressor(metadata["model_type"], model_path)
    scored = float(predict(reloaded, design_matrix(pd.DataFrame([row])))[0])
    saved_row = predictions.loc[predictions["date"] == target.strftime("%Y-%m-%d"), "y_pred"]
    saved = float(saved_row.iloc[0])
    assert scored == pytest.approx(saved, rel=1e-5, abs=1e-4)
