"""Optuna search over boosted tree regressors.

The objective is the mean MAE across expanding-window folds. The study never
sees the calibration or test rows; the caller is responsible for passing only
the development matrix.
"""

from dataclasses import dataclass

import numpy as np
import optuna

from forecasting.evaluation.metrics import mae
from forecasting.evaluation.splits import Fold
from forecasting.models.boosting import fit_regressor, predict


@dataclass(frozen=True)
class SearchSpec:
    model_type: str
    n_trials: int
    seed: int
    fast: bool


@dataclass(frozen=True)
class SearchResult:
    model_type: str
    seed: int
    params: dict
    cv_mae: float
    cv_mae_std: float
    fold_maes: list[float]
    trials: list[dict]


def choose_champion(results: list[SearchResult]) -> SearchResult:
    """Pick the lowest development CV MAE. Equal scores keep XGBoost."""
    if not results:
        raise ValueError("no search results to choose from")
    return min(
        results,
        key=lambda result: (result.cv_mae, 0 if result.model_type == "xgboost" else 1),
    )


def _plain_params(params: dict) -> dict:
    plain: dict = {}
    for key, value in params.items():
        if isinstance(value, bool):
            plain[key] = value
        elif isinstance(value, (np.floating, float)):
            plain[key] = float(value)
        elif isinstance(value, (np.integer, int)):
            plain[key] = int(value)
        else:
            plain[key] = value
    return plain


def _fixed_params(model_type: str, fast: bool) -> dict:
    if not fast:
        return {}
    if model_type == "xgboost":
        return {
            "n_estimators": 40,
            "learning_rate": 0.1,
            "min_child_weight": 1.0,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "reg_lambda": 1.0,
            "gamma": 0.0,
        }
    if model_type == "lightgbm":
        return {
            "n_estimators": 40,
            "learning_rate": 0.1,
            "min_child_samples": 20,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "reg_lambda": 1.0,
        }
    raise ValueError(f"unsupported model type: {model_type}")


def _sampled_params(trial: optuna.Trial, model_type: str, fast: bool) -> dict:
    if model_type == "xgboost" and fast:
        return {"max_depth": trial.suggest_int("max_depth", 2, 3)}
    if model_type == "lightgbm" and fast:
        return {"num_leaves": trial.suggest_int("num_leaves", 8, 16)}
    if model_type == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 80, 360),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        }
    if model_type == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 80, 360),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, 64),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 60),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
        }
    raise ValueError(f"unsupported model type: {model_type}")


def suggest_params(trial: optuna.Trial, model_type: str, fast: bool) -> dict:
    params = _fixed_params(model_type, fast)
    params.update(_sampled_params(trial, model_type, fast))
    return params


def _fold_maes(
    model_type: str,
    params: dict,
    seed: int,
    features: np.ndarray,
    target: np.ndarray,
    folds: list[Fold],
) -> list[float]:
    scores: list[float] = []
    for fold in folds:
        model = fit_regressor(
            model_type,
            params,
            seed,
            features[fold.train_idx],
            target[fold.train_idx],
        )
        predicted = predict(model, features[fold.valid_idx])
        scores.append(mae(target[fold.valid_idx], predicted))
    return scores


def tune_model(
    spec: SearchSpec,
    features: np.ndarray,
    target: np.ndarray,
    folds: list[Fold],
) -> SearchResult:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=spec.seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    trial_rows: list[dict] = []

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, spec.model_type, spec.fast)
        scores = _fold_maes(spec.model_type, params, spec.seed, features, target, folds)
        mean_score = float(np.mean(scores))
        std_score = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
        trial.set_user_attr("fold_maes", scores)
        trial_rows.append(
            {
                "model_type": spec.model_type,
                "trial": trial.number,
                "mean_mae": mean_score,
                "std_mae": std_score,
                **params,
            }
        )
        return mean_score

    study.optimize(objective, n_trials=spec.n_trials, show_progress_bar=False)
    best = study.best_trial
    params = _plain_params({**_fixed_params(spec.model_type, spec.fast), **best.params})
    fold_maes = [float(value) for value in best.user_attrs["fold_maes"]]
    mean_score = float(np.mean(fold_maes))
    std_score = float(np.std(fold_maes, ddof=1)) if len(fold_maes) > 1 else 0.0
    return SearchResult(
        model_type=spec.model_type,
        seed=spec.seed,
        params=params,
        cv_mae=mean_score,
        cv_mae_std=std_score,
        fold_maes=fold_maes,
        trials=trial_rows,
    )
