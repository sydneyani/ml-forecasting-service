"""Compare development-period features with the holdout year.

The reference window is the development period from the backtest report.
The current window is the test year. Both are built with the training
feature function, so the check looks at the matrix the model sees.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from forecasting.data.dataset import load_daily_load
from forecasting.features.build import feature_columns, supervised_frame
from forecasting.monitoring.psi import population_stability_index


def build_drift_report(
    data_path: Path,
    backtest_path: Path,
    threshold: float = 0.2,
) -> dict:
    raw = load_daily_load(data_path)
    featured = supervised_frame(raw)
    backtest = json.loads(Path(backtest_path).read_text())
    reference = _slice(
        featured, backtest["splits"]["develop_start"], backtest["splits"]["develop_end"]
    )
    current = _slice(featured, backtest["splits"]["test_start"], backtest["splits"]["test_end"])
    rows = []
    for column in feature_columns():
        psi = population_stability_index(
            reference[column].to_numpy(),
            current[column].to_numpy(),
        )
        rows.append(
            {
                "feature": column,
                "psi": psi,
                "alert": psi >= threshold,
            }
        )
    rows.sort(key=lambda row: row["psi"], reverse=True)
    alerts = [row["feature"] for row in rows if row["alert"]]
    return {
        "method": "population_stability_index",
        "threshold": threshold,
        "reference_split": "develop",
        "current_split": "test",
        "reference_start": backtest["splits"]["develop_start"],
        "reference_end": backtest["splits"]["develop_end"],
        "current_start": backtest["splits"]["test_start"],
        "current_end": backtest["splits"]["test_end"],
        "n_reference": int(len(reference)),
        "n_current": int(len(current)),
        "n_alerts": len(alerts),
        "alerts": alerts,
        "features": rows,
    }


def _slice(frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    dates = frame["date"]
    window = frame.loc[(dates >= start) & (dates <= end)]
    if window.empty:
        raise ValueError(f"no feature rows between {start} and {end}")
    return window


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="PSI of holdout features against the development window."
    )
    parser.add_argument("--data", type=Path, default=Path("data/raw/opsd_germany_daily.csv"))
    parser.add_argument("--backtest", type=Path, default=Path("reports/backtest_results.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/drift_report.json"))
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when any feature PSI is at least the threshold.",
    )
    args = parser.parse_args(argv)
    report = build_drift_report(args.data, args.backtest, args.threshold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"PSI reference {report['reference_start']} to {report['reference_end']} "
        f"vs {report['current_start']} to {report['current_end']}"
    )
    print(f"Alerts at PSI >= {args.threshold}: {report['n_alerts']}")
    for row in report["features"]:
        if row["alert"]:
            print(f"  {row['feature']}: {row['psi']:.3f}")
    print(f"Wrote {args.output}")
    if args.strict and report["n_alerts"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
