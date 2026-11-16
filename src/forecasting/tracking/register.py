"""Register the champion directory that is already on disk."""

import argparse
import json
from pathlib import Path

from forecasting.config import TrackingConfig
from forecasting.tracking.registry import log_champion


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Log models/champion to the local MLflow registry."
    )
    parser.add_argument("--model-dir", type=Path, default=Path("models/champion"))
    parser.add_argument("--report", type=Path, default=Path("reports/backtest_results.json"))
    parser.add_argument("--tracking-uri", default="sqlite:///mlruns/mlflow.db")
    parser.add_argument("--artifact-root", type=Path, default=Path("mlruns/artifacts"))
    parser.add_argument("--experiment-name", default="germany-daily-load")
    parser.add_argument("--registered-model-name", default="germany-daily-load")
    args = parser.parse_args(argv)
    report = json.loads(args.report.read_text()) if args.report.exists() else None
    tracking = TrackingConfig(
        enabled=True,
        tracking_uri=args.tracking_uri,
        artifact_root=args.artifact_root,
        experiment_name=args.experiment_name,
        registered_model_name=args.registered_model_name,
    )
    run_id = log_champion(args.model_dir, tracking, report)
    print(f"MLflow run: {run_id}")
    print(f"Registered model: {args.registered_model_name} (alias champion)")


if __name__ == "__main__":
    main()
