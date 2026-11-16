"""XGBoost and LightGBM regressors with a shared fit, save, and load path.

Missing feature values are left as NaN. Both libraries split on missingness
directly. Imputing those cells would erase the difference between "solar was
zero" and "solar was not measured yet".
"""

from pathlib import Path

import numpy as np


def _check_training_matrix(features: np.ndarray, target: np.ndarray) -> None:
    if features.ndim != 2:
        raise ValueError("features must be a two-dimensional array")
    if target.ndim != 1:
        raise ValueError("target must be one-dimensional")
    if features.shape[0] != target.shape[0]:
        raise ValueError("features and target have different numbers of rows")
    if features.shape[0] == 0:
        raise ValueError("cannot fit a model on zero rows")


def build_regressor(model_type: str, params: dict, seed: int):
    cleaned = dict(params)
    if model_type == "xgboost":
        from xgboost import XGBRegressor

        return XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            random_state=seed,
            n_jobs=1,
            verbosity=0,
            **cleaned,
        )
    if model_type == "lightgbm":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            objective="regression",
            random_state=seed,
            n_jobs=1,
            verbosity=-1,
            subsample_freq=1,
            force_row_wise=True,
            importance_type="gain",
            **cleaned,
        )
    raise ValueError(f"unsupported model type: {model_type}")


def fit_regressor(
    model_type: str,
    params: dict,
    seed: int,
    features: np.ndarray,
    target: np.ndarray,
):
    _check_training_matrix(features, target)
    model = build_regressor(model_type, params, seed)
    model.fit(features, target)
    return model


def predict(model, features: np.ndarray) -> np.ndarray:
    values = np.asarray(model.predict(features), dtype=np.float64)
    return values.reshape(-1)


def model_filename(model_type: str) -> str:
    if model_type == "xgboost":
        return "model.json"
    if model_type == "lightgbm":
        return "model.txt"
    raise ValueError(f"unsupported model type: {model_type}")


def model_format(model_type: str) -> str:
    if model_type == "xgboost":
        return "xgboost-json"
    if model_type == "lightgbm":
        return "lightgbm-text"
    raise ValueError(f"unsupported model type: {model_type}")


def save_regressor(model, model_type: str, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if model_type == "xgboost":
        model.save_model(path)
        return
    if model_type == "lightgbm":
        model.booster_.save_model(str(path))
        return
    raise ValueError(f"unsupported model type: {model_type}")


def load_regressor(model_type: str, path: Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if model_type == "xgboost":
        from xgboost import XGBRegressor

        model = XGBRegressor()
        model.load_model(path)
        return model
    if model_type == "lightgbm":
        import lightgbm as lgb

        return lgb.Booster(model_file=str(path))
    raise ValueError(f"unsupported model type: {model_type}")


def feature_gains(model, model_type: str, n_features: int) -> np.ndarray:
    """Total gain per column, aligned with the training matrix column order."""
    if model_type == "xgboost":
        scores = model.get_booster().get_score(importance_type="gain")
        gains = np.zeros(n_features, dtype=np.float64)
        for key, value in scores.items():
            if not str(key).startswith("f"):
                continue
            index = int(str(key)[1:])
            gains[index] = float(value)
        return gains
    if model_type == "lightgbm":
        gains = np.asarray(
            model.booster_.feature_importance(importance_type="gain"),
            dtype=np.float64,
        )
        if gains.shape != (n_features,):
            raise RuntimeError(
                f"lightgbm importance length {gains.shape[0]} does not match {n_features} features"
            )
        return gains
    raise ValueError(f"unsupported model type: {model_type}")
