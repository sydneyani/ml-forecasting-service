"""Load the champion directory and score a target date from caller-supplied history."""

import json
from datetime import date
from pathlib import Path

import pandas as pd

from forecasting.features.build import assert_feature_contract, design_matrix, inference_features
from forecasting.models.boosting import load_regressor, predict
from forecasting.serving.schemas import DailyObservation, PredictResponse, ReadyResponse


class LoadForecaster:
    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        metadata_path = self.model_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"champion metadata not found: {metadata_path}")
        self.metadata = json.loads(metadata_path.read_text())
        assert_feature_contract(self.metadata)
        self.model = load_regressor(
            self.metadata["model_type"],
            self.model_dir / self.metadata["model_filename"],
        )
        self.q_hat = float(self.metadata["interval_q_hat"])

    def ready(self) -> ReadyResponse:
        return ReadyResponse(
            status="ready",
            model_type=self.metadata["model_type"],
            model_version=self.metadata["model_version"],
            target_unit=self.metadata["target_unit"],
            interval_level=float(self.metadata["interval_level"]),
            test_mae_gwh=float(self.metadata["test_metrics"]["mae"]),
        )

    def predict(self, history: list[DailyObservation], target_date: date) -> PredictResponse:
        frame = pd.DataFrame([row.model_dump() for row in history])
        row = inference_features(frame, target_date)
        prediction = float(predict(self.model, design_matrix(pd.DataFrame([row])))[0])
        return PredictResponse(
            target_date=target_date,
            prediction_gwh=prediction,
            interval_low_gwh=prediction - self.q_hat,
            interval_high_gwh=prediction + self.q_hat,
            interval_level=float(self.metadata["interval_level"]),
            model_version=self.metadata["model_version"],
            model_type=self.metadata["model_type"],
        )
