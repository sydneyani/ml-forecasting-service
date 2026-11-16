import json
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from forecasting.config import TrackingConfig
from forecasting.tracking.registry import log_champion

ROOT = Path(__file__).resolve().parents[1]


def test_register_champion_sets_the_alias(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tracking = TrackingConfig(
        enabled=True,
        tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
        artifact_root=tmp_path / "artifacts",
        experiment_name="germany-daily-load",
        registered_model_name="germany-daily-load",
        run_name="test-register",
    )
    report = json.loads((ROOT / "reports" / "backtest_results.json").read_text())
    run_id = log_champion(ROOT / "models" / "champion", tracking, report)

    mlflow.set_tracking_uri(tracking.tracking_uri)
    run = mlflow.get_run(run_id)
    assert run.info.status == "FINISHED"
    assert run.data.metrics["test_mae"] == report["models"]["xgboost"]["test"]["mae"]
    assert "baseline_seasonal_naive_lag_364_mae" in run.data.metrics

    client = MlflowClient(tracking_uri=tracking.tracking_uri)
    version = client.get_model_version_by_alias("germany-daily-load", "champion")
    assert version.run_id == run_id
    assert (tmp_path / "artifacts").exists()
