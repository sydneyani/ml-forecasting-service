"""FastAPI app for single-day and batch day-ahead load scores.

The process loads the champion directory once. `/health` answers even when
that directory is missing. `/ready` and the predict routes return 503 until
a model is loaded. There is no authentication: this is a local scoring demo.
"""

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException

from forecasting import __version__
from forecasting.features.build import ForecastInputError
from forecasting.serving.predictor import LoadForecaster
from forecasting.serving.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
    ReadyResponse,
)


def create_app(model_dir: Path | None = None) -> FastAPI:
    if model_dir is None:
        model_dir = Path(os.environ.get("MODEL_DIR", "models/champion"))
    forecaster: LoadForecaster | None = None
    if Path(model_dir).joinpath("metadata.json").is_file():
        forecaster = LoadForecaster(model_dir)

    app = FastAPI(
        title="Germany daily load forecast",
        version=__version__,
        summary="Day-ahead national electricity load in GWh.",
    )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.get("/ready", response_model=ReadyResponse)
    def ready() -> ReadyResponse:
        if forecaster is None:
            raise HTTPException(status_code=503, detail="model is not loaded")
        return forecaster.ready()

    @app.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest) -> PredictResponse:
        loaded = _require(forecaster)
        try:
            return loaded.predict(request.history, request.target_date)
        except ForecastInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/predict/batch", response_model=BatchPredictResponse)
    def predict_batch(request: BatchPredictRequest) -> BatchPredictResponse:
        loaded = _require(forecaster)
        try:
            predictions = [loaded.predict(request.history, day) for day in request.target_dates]
        except ForecastInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return BatchPredictResponse(
            predictions=predictions,
            model_version=loaded.metadata["model_version"],
        )

    return app


def _require(forecaster: LoadForecaster | None) -> LoadForecaster:
    if forecaster is None:
        raise HTTPException(status_code=503, detail="model is not loaded")
    return forecaster


app = create_app()
