"""Held-out backtest for the day-ahead load model.

Rows are ordered by date. The last `test_days` rows are the test set and are
not used for tuning or fitting. The `calibration_days` rows before that are
used only to set the conformal interval width. Optuna minimizes mean MAE on
expanding windows inside the remaining development rows. The champion is the
model with the lowest development CV MAE. Test metrics are computed after
that choice. The champion is then refit on the full development period, which
includes rows that individual CV folds held out. The test metric, not the CV
mean, is the number to quote.
"""

import argparse
import hashlib
import json
import logging
import math
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from forecasting.config import TrainConfig, load_config, resolve_paths
from forecasting.data.dataset import DATASET_NAME, TARGET_COLUMN, TARGET_UNIT, load_daily_load
from forecasting.evaluation.conformal import conformal_quantile, conformal_rank
from forecasting.evaluation.metrics import mae, mae_ratio, regression_scores
from forecasting.evaluation.splits import expanding_window_splits
from forecasting.features.build import (
    FEATURE_SCHEMA_VERSION,
    HORIZON_DAYS,
    MIN_HISTORY_DAYS,
    design_matrix,
    feature_columns,
    supervised_frame,
)
from forecasting.models.baselines import lagged_actual
from forecasting.models.boosting import (
    feature_gains,
    fit_regressor,
    model_filename,
    model_format,
    predict,
    save_regressor,
)
from forecasting.training.search import SearchResult, SearchSpec, choose_champion, tune_model

logger = logging.getLogger(__name__)

SELECTION_RULE = (
    "Lowest mean MAE on expanding-window cross-validation inside the development period. "
    "Ties break toward xgboost. Calibration residuals set the interval width. "
    "The test period is scored after the champion is fixed."
)
DOW_NAMES = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
BASELINES = (
    ("naive_lag_1", 1),
    ("seasonal_naive_lag_7", 7),
    ("seasonal_naive_lag_364", 364),
)


def _iso(value: object) -> str:
    return pd.Timestamp(value).date().isoformat()


def _display_path(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(resolved)


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        return "unknown"
    return "unknown"


def _library_versions() -> dict[str, str]:
    import holidays
    import lightgbm
    import numpy
    import optuna
    import pandas
    import sklearn
    import xgboost

    return {
        "pandas": pandas.__version__,
        "numpy": numpy.__version__,
        "scikit-learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "lightgbm": lightgbm.__version__,
        "optuna": optuna.__version__,
        "holidays": holidays.__version__,
    }


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return _display_path(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("refusing to write a non-finite metric")
        return number
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    return value


def _config_snapshot(config: TrainConfig) -> dict:
    dumped = config.model_dump(mode="json")
    dumped["data"]["csv_path"] = _display_path(Path(dumped["data"]["csv_path"]))
    dumped["output"]["reports_dir"] = _display_path(Path(dumped["output"]["reports_dir"]))
    dumped["output"]["model_dir"] = _display_path(Path(dumped["output"]["model_dir"]))
    return dumped


def _grouped_mae(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray) -> dict[int, float]:
    scores: dict[int, float] = {}
    for key in sorted(set(groups.tolist())):
        mask = groups == key
        scores[int(key)] = mae(y_true[mask], y_pred[mask])
    return scores


def _write_group_metrics(
    path: Path,
    group_name: str,
    labels: dict[int, str] | None,
    y_true: np.ndarray,
    champion_pred: np.ndarray,
    naive_pred: np.ndarray,
    groups: np.ndarray,
) -> None:
    champion_scores = _grouped_mae(y_true, champion_pred, groups)
    naive_scores = _grouped_mae(y_true, naive_pred, groups)
    rows = []
    for key in champion_scores:
        row = {
            group_name: key,
            "n": int(np.sum(groups == key)),
            "champion_mae": champion_scores[key],
            "seasonal_naive_lag_7_mae": naive_scores[key],
        }
        if labels is not None:
            row[f"{group_name}_name"] = labels[key]
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def run_backtest(config: TrainConfig) -> dict:
    started = time.perf_counter()
    raw = load_daily_load(config.data.csv_path)
    supervised = supervised_frame(raw)
    if not supervised["date"].is_monotonic_increasing:
        raise RuntimeError("supervised frame is not sorted by date")

    test_days = config.data.test_days
    calibration_days = config.data.calibration_days
    if test_days + calibration_days >= len(supervised):
        raise ValueError(
            "test_days and calibration_days leave no development rows "
            f"({len(supervised)} supervised rows)"
        )

    test = supervised.iloc[-test_days:].reset_index(drop=True)
    pre_test = supervised.iloc[:-test_days].reset_index(drop=True)
    calibration = pre_test.iloc[-calibration_days:].reset_index(drop=True)
    develop = pre_test.iloc[:-calibration_days].reset_index(drop=True)
    folds = expanding_window_splits(
        n_rows=len(develop),
        n_splits=config.cv.n_splits,
        valid_size=config.cv.valid_days,
        min_train_size=config.cv.min_train_days,
    )

    logger.info(
        "supervised rows=%s develop=%s to %s calibration=%s to %s test=%s to %s",
        len(supervised),
        _iso(develop["date"].iloc[0]),
        _iso(develop["date"].iloc[-1]),
        _iso(calibration["date"].iloc[0]),
        _iso(calibration["date"].iloc[-1]),
        _iso(test["date"].iloc[0]),
        _iso(test["date"].iloc[-1]),
    )

    x_develop = design_matrix(develop)
    y_develop = develop[TARGET_COLUMN].to_numpy(dtype=np.float64)
    searches: list[SearchResult] = []
    for item in config.search:
        logger.info("tuning %s for %s trials", item.model_type, item.n_trials)
        searches.append(
            tune_model(
                SearchSpec(
                    model_type=item.model_type,
                    n_trials=item.n_trials,
                    seed=item.seed,
                    fast=config.fast,
                ),
                x_develop,
                y_develop,
                folds,
            )
        )

    champion_search = choose_champion(searches)
    logger.info(
        "champion %s development CV MAE %.3f",
        champion_search.model_type,
        champion_search.cv_mae,
    )

    fitted = {
        result.model_type: fit_regressor(
            result.model_type,
            result.params,
            result.seed,
            x_develop,
            y_develop,
        )
        for result in searches
    }
    champion_model = fitted[champion_search.model_type]
    x_calibration = design_matrix(calibration)
    y_calibration = calibration[TARGET_COLUMN].to_numpy(dtype=np.float64)
    calibration_pred = predict(champion_model, x_calibration)
    residual = np.abs(y_calibration - calibration_pred)
    q_hat = conformal_quantile(residual, config.alpha)
    if not np.isfinite(q_hat):
        raise RuntimeError(
            "conformal quantile is infinite; increase calibration_days or alpha "
            f"(n={len(residual)}, alpha={config.alpha})"
        )

    x_test = design_matrix(test)
    y_test = test[TARGET_COLUMN].to_numpy(dtype=np.float64)
    test_predictions = {model_type: predict(model, x_test) for model_type, model in fitted.items()}
    champion_pred = test_predictions[champion_search.model_type]
    baseline_predictions = {name: lagged_actual(raw, test["date"], lag) for name, lag in BASELINES}
    for name, values in baseline_predictions.items():
        if not np.isfinite(values).all():
            raise RuntimeError(f"{name} has a non-finite prediction on the test window")

    baseline_payload = {
        name: {"lag": lag, "test": regression_scores(y_test, baseline_predictions[name])}
        for name, lag in BASELINES
    }
    model_payload: dict[str, dict] = {}
    ordered_results = sorted(
        searches,
        key=lambda result: {"xgboost": 0, "lightgbm": 1}.get(result.model_type, 2),
    )
    for result in ordered_results:
        model_payload[result.model_type] = {
            "role": "champion" if result.model_type == champion_search.model_type else "challenger",
            "seed": result.seed,
            "params": result.params,
            "cv_mae": result.cv_mae,
            "cv_mae_std": result.cv_mae_std,
            "fold_maes": result.fold_maes,
            "test": regression_scores(y_test, test_predictions[result.model_type]),
        }

    champion_scores = model_payload[champion_search.model_type]["test"]
    naive_mae = baseline_payload["seasonal_naive_lag_7"]["test"]["mae"]
    naive_year_mae = baseline_payload["seasonal_naive_lag_364"]["test"]["mae"]
    ratio = mae_ratio(champion_scores["mae"], naive_mae)
    ratio_year = mae_ratio(champion_scores["mae"], naive_year_mae)
    covered = (y_test >= champion_pred - q_hat) & (y_test <= champion_pred + q_hat)
    cv_folds = []
    for index, fold in enumerate(folds, start=1):
        train_dates = develop["date"].iloc[fold.train_idx]
        valid_dates = develop["date"].iloc[fold.valid_idx]
        cv_folds.append(
            {
                "fold": index,
                "train_start": _iso(train_dates.iloc[0]),
                "train_end": _iso(train_dates.iloc[-1]),
                "valid_start": _iso(valid_dates.iloc[0]),
                "valid_end": _iso(valid_dates.iloc[-1]),
                "n_train": int(len(fold.train_idx)),
                "n_valid": int(len(fold.valid_idx)),
            }
        )

    reports_dir = config.output.reports_dir
    model_dir = config.output.model_dir
    reports_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    filename = model_filename(champion_search.model_type)
    model_path = model_dir / filename
    save_regressor(champion_model, champion_search.model_type, model_path)
    model_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
    data_sha = hashlib.sha256(Path(config.data.csv_path).read_bytes()).hexdigest()
    trained_at = datetime.now(UTC).replace(microsecond=0).isoformat()

    columns = feature_columns()
    gains = feature_gains(champion_model, champion_search.model_type, len(columns))
    total_gain = float(gains.sum())
    shares = gains / total_gain if total_gain else gains
    importance = pd.DataFrame({"feature": columns, "gain": gains, "gain_share": shares})
    importance = importance.sort_values("gain_share", ascending=False)
    importance_path = reports_dir / "feature_importance.csv"
    importance.to_csv(importance_path, index=False)

    prediction_frame = pd.DataFrame(
        {
            "date": test["date"].dt.strftime("%Y-%m-%d"),
            "y_true": y_test,
            "y_pred": champion_pred,
            "y_low": champion_pred - q_hat,
            "y_high": champion_pred + q_hat,
            "naive_lag_1": baseline_predictions["naive_lag_1"],
            "seasonal_naive_lag_7": baseline_predictions["seasonal_naive_lag_7"],
            "seasonal_naive_lag_364": baseline_predictions["seasonal_naive_lag_364"],
        }
    )
    predictions_path = reports_dir / "test_predictions.csv"
    prediction_frame.to_csv(predictions_path, index=False)

    trial_frame = pd.DataFrame([row for result in searches for row in result.trials])
    trials_path = reports_dir / "optuna_trials.csv"
    trial_frame.to_csv(trials_path, index=False)

    dow = test["date"].dt.day_of_week.to_numpy()
    month = test["date"].dt.month.to_numpy()
    dow_path = reports_dir / "metrics_by_dow.csv"
    month_path = reports_dir / "metrics_by_month.csv"
    _write_group_metrics(
        dow_path,
        "dow",
        dict(enumerate(DOW_NAMES)),
        y_test,
        champion_pred,
        baseline_predictions["seasonal_naive_lag_7"],
        dow,
    )
    _write_group_metrics(
        month_path,
        "month",
        None,
        y_test,
        champion_pred,
        baseline_predictions["seasonal_naive_lag_7"],
        month,
    )

    metadata = {
        "model_type": champion_search.model_type,
        "model_format": model_format(champion_search.model_type),
        "model_filename": filename,
        "model_version": model_sha[:12],
        "model_sha256": model_sha,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_columns": columns,
        "target": TARGET_COLUMN,
        "target_unit": TARGET_UNIT,
        "horizon_days": HORIZON_DAYS,
        "min_history_days": MIN_HISTORY_DAYS,
        "interval_method": "split_conformal_absolute_residual",
        "interval_level": 1.0 - config.alpha,
        "interval_alpha": config.alpha,
        "interval_q_hat": q_hat,
        "dataset": DATASET_NAME,
        "data_sha256": data_sha,
        "trained_at": trained_at,
        "params": champion_search.params,
        "cv_mae": champion_search.cv_mae,
        "test_metrics": champion_scores,
        "library_versions": _library_versions(),
    }
    metadata_path = model_dir / "metadata.json"
    metadata_path.write_text(json.dumps(_json_ready(metadata), indent=2) + "\n")

    report_path = reports_dir / "backtest_results.json"
    report = {
        "dataset": DATASET_NAME,
        "target": TARGET_COLUMN,
        "target_unit": TARGET_UNIT,
        "horizon_days": HORIZON_DAYS,
        "selection_rule": SELECTION_RULE,
        "champion_model_type": champion_search.model_type,
        "model_version": model_sha[:12],
        "model_sha256": model_sha,
        "data_sha256": data_sha,
        "trained_at": trained_at,
        "rows": {
            "raw": int(len(raw)),
            "supervised": int(len(supervised)),
            "develop": int(len(develop)),
            "calibration": int(len(calibration)),
            "test": int(len(test)),
        },
        "splits": {
            "develop_start": _iso(develop["date"].iloc[0]),
            "develop_end": _iso(develop["date"].iloc[-1]),
            "calibration_start": _iso(calibration["date"].iloc[0]),
            "calibration_end": _iso(calibration["date"].iloc[-1]),
            "test_start": _iso(test["date"].iloc[0]),
            "test_end": _iso(test["date"].iloc[-1]),
        },
        "cv_folds": cv_folds,
        "baselines": baseline_payload,
        "models": model_payload,
        "mae_ratio_vs_seasonal_naive_lag_7": ratio,
        "mae_skill_vs_seasonal_naive_lag_7": 1.0 - ratio,
        "mae_ratio_vs_seasonal_naive_lag_364": ratio_year,
        "mae_skill_vs_seasonal_naive_lag_364": 1.0 - ratio_year,
        "mean_error_gwh": float(np.mean(champion_pred - y_test)),
        "interval": {
            "method": "split_conformal_absolute_residual",
            "level": 1.0 - config.alpha,
            "alpha": config.alpha,
            "calibration_size": int(len(calibration)),
            "rank": conformal_rank(len(calibration), config.alpha),
            "q_hat": q_hat,
            "mean_width": float(2.0 * q_hat),
            "test_coverage": float(np.mean(covered)),
        },
        "runtime_seconds": time.perf_counter() - started,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu": _cpu_model(),
        },
        "library_versions": metadata["library_versions"],
        "config": _config_snapshot(config),
        "artifacts": {
            "report_path": _display_path(report_path),
            "predictions_path": _display_path(predictions_path),
            "trials_path": _display_path(trials_path),
            "importance_path": _display_path(importance_path),
            "metrics_by_dow_path": _display_path(dow_path),
            "metrics_by_month_path": _display_path(month_path),
            "model_dir": _display_path(model_dir),
            "model_path": _display_path(model_path),
            "metadata_path": _display_path(metadata_path),
        },
    }
    # runtime is filled just before the write so the file includes the full run.
    report["runtime_seconds"] = time.perf_counter() - started
    payload = _json_ready(report)
    report_path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def format_summary(result: dict) -> str:
    splits = result["splits"]
    rows = result["rows"]
    champion = result["champion_model_type"]
    champion_payload = result["models"][champion]
    lines = [
        f"Test window: {splits['test_start']} to {splits['test_end']} ({rows['test']} days)",
        (
            f"Development: {splits['develop_start']} to {splits['develop_end']} "
            f"({rows['develop']} days)"
        ),
        (
            f"Calibration: {splits['calibration_start']} to {splits['calibration_end']} "
            f"({rows['calibration']} days)"
        ),
        (
            f"Champion: {champion} "
            f"(development CV MAE {champion_payload['cv_mae']:.2f} "
            f"± {champion_payload['cv_mae_std']:.2f} GWh)"
        ),
        result["selection_rule"],
        "",
        f"{'model':<28} {'CV MAE':>10} {'test MAE':>10} {'test RMSE':>10} {'test MAPE %':>12}",
    ]
    for name, payload in result["baselines"].items():
        scores = payload["test"]
        lines.append(
            f"{name:<28} {'-':>10} {scores['mae']:10.2f} "
            f"{scores['rmse']:10.2f} {scores['mape']:12.2f}"
        )
    for name, payload in result["models"].items():
        label = f"{name} (champion)" if name == champion else name
        scores = payload["test"]
        lines.append(
            f"{label:<28} {payload['cv_mae']:10.2f} {scores['mae']:10.2f} "
            f"{scores['rmse']:10.2f} {scores['mape']:12.2f}"
        )
    interval = result["interval"]
    lines.extend(
        [
            "",
            (
                f"{interval['level']:.0%} interval half-width {interval['q_hat']:.2f} GWh "
                f"(mean width {interval['mean_width']:.2f} GWh), "
                f"test coverage {interval['test_coverage']:.3f}"
            ),
            (
                "Champion test MAE is "
                f"{result['mae_ratio_vs_seasonal_naive_lag_7']:.3f} times "
                "the lag-7 seasonal naive, and "
                f"{result['mae_ratio_vs_seasonal_naive_lag_364']:.3f} times "
                "the lag-364 seasonal naive."
            ),
            (f"Champion mean error (prediction minus actual): {result['mean_error_gwh']:.2f} GWh"),
            f"Runtime: {result['runtime_seconds']:.1f}s on {result['environment']['cpu']}",
            f"Model version: {result['model_version']}",
            f"Report: {result['artifacts']['report_path']}",
            f"Champion artifact: {result['artifacts']['model_path']}",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Backtest the day-ahead load forecast and write a champion model."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--n-trials", type=int, default=None)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.n_trials is not None:
        if args.n_trials < 1:
            raise SystemExit("--n-trials must be at least 1")
        config = config.model_copy(
            update={
                "search": [
                    item.model_copy(update={"n_trials": args.n_trials}) for item in config.search
                ]
            }
        )
    result = run_backtest(resolve_paths(config))
    print(format_summary(result))


if __name__ == "__main__":
    main()
