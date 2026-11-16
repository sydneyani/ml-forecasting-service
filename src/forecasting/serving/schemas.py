"""Request and response models for the scoring API."""

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class DailyObservation(BaseModel):
    """One published day of national load and optional generation actuals."""

    model_config = ConfigDict(extra="forbid")

    date: date
    consumption: float = Field(ge=0)
    wind: float | None = Field(default=None, ge=0)
    solar: float | None = Field(default=None, ge=0)


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_date: date
    history: list[DailyObservation] = Field(min_length=1, max_length=5000)


class PredictResponse(BaseModel):
    target_date: date
    prediction_gwh: float
    interval_low_gwh: float
    interval_high_gwh: float
    interval_level: float
    model_version: str
    model_type: str


class BatchPredictRequest(BaseModel):
    """Several target dates that share one history.

    Each date is scored from actual history only. The response does not feed
    one prediction back in as the next day's load.
    """

    model_config = ConfigDict(extra="forbid")

    target_dates: list[date] = Field(min_length=1, max_length=366)
    history: list[DailyObservation] = Field(min_length=1, max_length=5000)


class BatchPredictResponse(BaseModel):
    predictions: list[PredictResponse]
    model_version: str


class HealthResponse(BaseModel):
    status: Literal["ok"]


class ReadyResponse(BaseModel):
    status: Literal["ready"]
    model_type: str
    model_version: str
    target_unit: str
    interval_level: float
    test_mae_gwh: float
