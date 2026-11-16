import json
from pathlib import Path

import numpy as np
import pytest

from forecasting.monitoring.drift import build_drift_report
from forecasting.monitoring.psi import population_stability_index

ROOT = Path(__file__).resolve().parents[1]


def test_identical_samples_have_no_psi_and_a_shift_does():
    reference = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    assert population_stability_index(reference, reference.copy()) == pytest.approx(0.0, abs=1e-6)
    shifted = np.array([10.0, 11.0, 10.0, 12.0, 11.0, 10.0])
    assert population_stability_index(reference, shifted) > 0.25

    continuous = np.linspace(0.0, 1.0, 200)
    assert population_stability_index(continuous, continuous + 5.0) > 0.2
    assert population_stability_index(continuous, continuous.copy()) < 0.01


def test_published_drift_report_matches_a_fresh_computation():
    fresh = build_drift_report(
        ROOT / "data" / "raw" / "opsd_germany_daily.csv",
        ROOT / "reports" / "backtest_results.json",
        threshold=0.2,
    )
    saved = json.loads((ROOT / "reports" / "drift_report.json").read_text())
    assert fresh["n_alerts"] == saved["n_alerts"]
    assert [row["feature"] for row in fresh["features"]] == [
        row["feature"] for row in saved["features"]
    ]
    for left, right in zip(fresh["features"], saved["features"], strict=True):
        assert left["psi"] == pytest.approx(right["psi"], rel=1e-6, abs=1e-8)
        assert left["alert"] is right["alert"]
