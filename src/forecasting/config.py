"""Training configuration loaded from YAML."""

from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

ModelType = Literal["xgboost", "lightgbm"]


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    csv_path: Path
    test_days: int = Field(gt=0)
    calibration_days: int = Field(gt=0)


class CVConfig(BaseModel):
    """Expanding-window cross-validation inside the development period.

    `valid_days` counts supervised rows. After the feature build, one row is one day.
    """

    model_config = ConfigDict(extra="forbid")

    n_splits: int = Field(gt=0)
    valid_days: int = Field(gt=0)
    min_train_days: int = Field(gt=0)


class ModelSearchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_type: ModelType
    n_trials: int = Field(gt=0)
    seed: int = 42


class OutputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reports_dir: Path
    model_dir: Path


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data: DataConfig
    cv: CVConfig
    search: list[ModelSearchConfig] = Field(min_length=1)
    fast: bool = False
    alpha: float = Field(default=0.1, gt=0, lt=1)
    output: OutputConfig

    @model_validator(mode="after")
    def unique_model_types(self) -> Self:
        kinds = [item.model_type for item in self.search]
        if len(kinds) != len(set(kinds)):
            raise ValueError("search lists a model_type more than once")
        return self


def load_config(path: Path) -> TrainConfig:
    payload = yaml.safe_load(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return TrainConfig.model_validate(payload)


def resolve_paths(config: TrainConfig, base_dir: Path | None = None) -> TrainConfig:
    """Resolve relative paths against `base_dir` (the process working directory by default)."""
    root = Path.cwd() if base_dir is None else Path(base_dir)

    def resolve(path: Path) -> Path:
        if path.is_absolute():
            return path
        return (root / path).resolve()

    return config.model_copy(
        update={
            "data": config.data.model_copy(update={"csv_path": resolve(config.data.csv_path)}),
            "output": config.output.model_copy(
                update={
                    "reports_dir": resolve(config.output.reports_dir),
                    "model_dir": resolve(config.output.model_dir),
                }
            ),
        }
    )
