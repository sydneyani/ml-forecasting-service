"""Log the champion artifact to a local MLflow registry.

File-only `mlruns/` tracking cannot host a model registry. The tracking URI is
SQLite, and artifacts live in a directory next to that database. Scoring does
not read the registry: the API loads `models/champion`. The registry is the
record of which run produced that directory.
"""

import json
from pathlib import Path

import mlflow
import numpy as np
from mlflow.tracking import MlflowClient

from forecasting.config import TrackingConfig
from forecasting.models.boosting import load_regressor


def log_champion(
    model_dir: Path,
    tracking: TrackingConfig,
    report: dict | None = None,
) -> str:
    """Log metrics and the saved model, then point the `champion` alias at it."""
    model_dir = Path(model_dir)
    metadata = json.loads((model_dir / "metadata.json").read_text())
    model = load_regressor(metadata["model_type"], model_dir / metadata["model_filename"])
    _ensure_experiment(tracking)
    with mlflow.start_run(run_name=tracking.run_name) as run:
        mlflow.set_tags(
            {
                "model_version": metadata["model_version"],
                "model_type": metadata["model_type"],
                "dataset": metadata["dataset"],
            }
        )
        mlflow.log_params(_params(metadata))
        mlflow.log_metrics(_metrics(metadata, report))
        mlflow.log_dict(metadata, "champion_metadata.json")
        _log_model(model, metadata, tracking.registered_model_name)
        _alias_latest(tracking)
        return run.info.run_id


def _ensure_experiment(tracking: TrackingConfig) -> None:
    tracking.artifact_root.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking.tracking_uri)
    client = MlflowClient(tracking_uri=tracking.tracking_uri)
    if client.get_experiment_by_name(tracking.experiment_name) is None:
        client.create_experiment(
            tracking.experiment_name,
            artifact_location=tracking.artifact_root.resolve().as_uri(),
        )
    mlflow.set_experiment(tracking.experiment_name)


def _params(metadata: dict) -> dict[str, str]:
    params = {key: _param(value) for key, value in metadata.get("params", {}).items()}
    params["model_type"] = str(metadata["model_type"])
    params["feature_schema_version"] = str(metadata["feature_schema_version"])
    return params


def _param(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.8g}"
    return str(value)


def _metrics(metadata: dict, report: dict | None) -> dict[str, float]:
    scores = dict(metadata.get("test_metrics", {}))
    metrics = {f"test_{name}": float(value) for name, value in scores.items()}
    if "cv_mae" in metadata:
        metrics["cv_mae"] = float(metadata["cv_mae"])
    metrics["interval_q_hat"] = float(metadata["interval_q_hat"])
    if report is None:
        return metrics
    metrics["mae_ratio_vs_seasonal_naive_lag_7"] = float(
        report["mae_ratio_vs_seasonal_naive_lag_7"]
    )
    metrics["mae_ratio_vs_seasonal_naive_lag_364"] = float(
        report["mae_ratio_vs_seasonal_naive_lag_364"]
    )
    metrics["interval_test_coverage"] = float(report["interval"]["test_coverage"])
    for name, payload in report["baselines"].items():
        metrics[f"baseline_{name}_mae"] = float(payload["test"]["mae"])
    return metrics


def _log_model(model, metadata: dict, registered_model_name: str) -> None:
    example = np.zeros((1, len(metadata["feature_columns"])), dtype=np.float64)
    signature = mlflow.models.infer_signature(example, np.asarray(model.predict(example)))
    requirements = [
        "pandas>=2.2,<3.2",
        "numpy>=1.26,<3",
        "scikit-learn>=1.5,<2",
        "xgboost==3.4.1",
        "lightgbm==4.7.0",
    ]
    kwargs = {
        "name": "model",
        "registered_model_name": registered_model_name,
        "signature": signature,
        "input_example": example,
        "pip_requirements": requirements,
    }
    if metadata["model_type"] == "xgboost":
        mlflow.xgboost.log_model(model, **kwargs)
        return
    if metadata["model_type"] == "lightgbm":
        mlflow.lightgbm.log_model(model, **kwargs)
        return
    raise ValueError(f"unsupported model type: {metadata['model_type']}")


def _alias_latest(tracking: TrackingConfig) -> None:
    client = MlflowClient(tracking_uri=tracking.tracking_uri)
    versions = client.search_model_versions(f"name='{tracking.registered_model_name}'")
    if not versions:
        raise RuntimeError(f"no versions registered for {tracking.registered_model_name}")
    latest = max(versions, key=lambda version: int(version.version))
    client.set_registered_model_alias(
        tracking.registered_model_name,
        "champion",
        latest.version,
    )
