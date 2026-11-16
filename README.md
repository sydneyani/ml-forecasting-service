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

The published metrics in `reports/` are the output of that command.

## Track

`configs/train.yaml` logs the run to a local SQLite-backed MLflow store and registers the model under the alias `champion`. A file-only `mlruns` store cannot host the registry. To record the champion that is already in the tree, without another search:

```bash
python -m forecasting.tracking.register
```

## Serve

The API loads `models/champion`. It does not call MLflow at request time.

```bash
python -m forecasting.serving
curl -s localhost:8000/health
curl -s localhost:8000/ready
```

`POST /predict` takes a target date and daily history. Features are built with the same function the backtest uses. `POST /predict/batch` scores several dates from one history and does not feed predictions back in as later actuals. There is no authentication on this demo.
