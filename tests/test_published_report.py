"""Consistency checks for the committed 2017 holdout, not a fresh training run."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from forecasting.data.dataset import load_daily_load
from forecasting.features.build import assert_feature_contract, design_matrix, inference_features
from forecasting.models.boosting import load_regressor, predict

ROOT = Path(__file__).resolve().parents[1]


def test_published_backtest_matches_the_champion_artifact():
    report = json.loads((ROOT / "reports" / "backtest_results.json").read_text())
    rows = report["rows"]
    assert rows["develop"] + rows["calibration"] + rows["test"] == rows["supervised"]
    assert report["splits"]["develop_end"] < report["splits"]["calibration_start"]
    assert report["splits"]["calibration_end"] < report["splits"]["test_start"]

    champion = report["champion_model_type"]
    champion_mae = report["models"][champion]["test"]["mae"]
    naive_week = report["baselines"]["seasonal_naive_lag_7"]["test"]["mae"]
    naive_year = report["baselines"]["seasonal_naive_lag_364"]["test"]["mae"]
    assert champion_mae < naive_year < naive_week
    assert report["mae_ratio_vs_seasonal_naive_lag_7"] == pytest.approx(champion_mae / naive_week)
    assert report["mae_ratio_vs_seasonal_naive_lag_364"] == pytest.approx(champion_mae / naive_year)
    assert report["models"][champion]["role"] == "champion"

    metadata = json.loads((ROOT / "models" / "champion" / "metadata.json").read_text())
    assert_feature_contract(metadata)
    assert metadata["model_version"] == report["model_version"]
    assert metadata["test_metrics"]["mae"] == pytest.approx(champion_mae)

    predictions = pd.read_csv(ROOT / "reports" / "test_predictions.csv")
    assert len(predictions) == rows["test"]
    half_width = report["interval"]["q_hat"]
    assert np.allclose(predictions["y_high"] - predictions["y_low"], 2 * half_width)
    covered = (predictions["y_true"] >= predictions["y_low"]) & (
        predictions["y_true"] <= predictions["y_high"]
    )
    assert covered.mean() == pytest.approx(report["interval"]["test_coverage"])

    raw = load_daily_load(ROOT / "data" / "raw" / "opsd_germany_daily.csv")
    row = inference_features(raw, "2017-12-31")
    model = load_regressor(
        metadata["model_type"],
        ROOT / "models" / "champion" / metadata["model_filename"],
    )
    reloaded = float(predict(model, design_matrix(pd.DataFrame([row])))[0])
    saved = float(predictions.loc[predictions["date"] == "2017-12-31", "y_pred"].iloc[0])
    assert reloaded == pytest.approx(saved, rel=1e-5, abs=1e-4)
