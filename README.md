# ml-forecasting-service

Day-ahead forecast of German daily electricity load.

The series is the Open Power System Data Germany daily file for 2006–2017, in gigawatt-hours: national load, wind generation, and solar generation. Training builds leakage-safe lag, rolling, and calendar features, tunes XGBoost and LightGBM with Optuna on expanding windows, and scores the chosen model on a final year that tuning never sees. Seasonal naive forecasts are the baselines. A split-conformal interval is set on a calibration slice that the fit also never sees.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
ruff check src tests && ruff format --check src tests
python -m forecasting.training --config configs/train.yaml
```

`configs/ci.yaml` is the same protocol with a two-trial search. The backtest writes `reports/backtest_results.json` and the champion artifact under `models/champion/`. The daily file is vendored at `data/raw/opsd_germany_daily.csv`; `python -m forecasting.data.download` checks it against the pinned SHA-256.

The published metrics in `reports/` are the output of that command. Serving, MLflow, drift checks, and the longer write-up sit on top of this training path.
