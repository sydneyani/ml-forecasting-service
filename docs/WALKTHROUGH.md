# Walkthrough

This is the path through the repository, in the order a design review usually goes. The numbers are the committed 2017 holdout (`reports/backtest_results.json`, model version `5828f2d8d5a4`). A fresh `configs/train.yaml` run can move them. Quote the file you are looking at.

## The decision

The forecast is national electricity consumption in Germany on day `t`, in gigawatt-hours. The origin is the end of day `t-1`. Anything that happens during day `t` is not an input.

That rules out same-day wind and solar. The file stores generation actuals. A transmission operator would have a day-ahead wind forecast, and this dataset does not. Using the actual would make the backtest look better than a real morning-ahead forecast. The feature code therefore keeps only lags of wind and solar.

## The information set

`src/forecasting/features/build.py` is the contract.

- Load lags at 1, 2, 3, 7, 14, 21, 28, and 364 days. Lag 364 is 52 weeks, so the year-ago value falls on the same weekday. Lag 365 would mix weekdays.
- Rolling mean and sample standard deviation of load over 7, 14, and 28 days. The series is shifted by one day before the window, so day `t` is not inside its own features.
- Day-of-week and month as 0/1 columns, plus day of year, plus a German national holiday flag. Those are facts about day `t`, and the calendar is known in advance.
- Missing wind and solar stay missing. A zero would mean "no generation," which is a different event from "this series had not started."

The daily index is filled before any shift. A missing calendar day becomes an empty row, so "yesterday" cannot silently become "the previous row." Supervised training drops rows that still lack a load lag. It does not drop rows only because solar is missing. The first usable day is 364 days after 2006-01-01, which leaves 4,019 rows.

`inference_features` throws away every observation on or after the target date and then calls the same builder. Passing a frame that already contains tomorrow's actual does not change the feature row. `tests/test_features.py` locks that, including a one-day gap in the calendar.

## Why the split is not random

Random k-fold would train on Tuesdays that occur after the Monday being scored. The backtest cuts the supervised frame into three contiguous pieces:

| Piece | Dates | Rows | Used for |
| --- | --- | --- | --- |
| Development | 2006-12-31 to 2016-10-02 | 3,564 | Optuna and the final fit |
| Calibration | 2016-10-03 to 2016-12-31 | 90 | Interval width only |
| Test | 2017-01-01 to 2017-12-31 | 365 | One score, after the champion is fixed |

Inside development, cross-validation is four expanding windows of 90 days. Together they cover 2015-10-09 through 2016-10-02, the year of development data closest to the test. Each fold trains only on rows before its validation block. The final champion is refit on all of development, so those validation rows are inside the served model. The CV mean is therefore a selection score, not an unbiased estimate of 2017. The number to quote is the test MAE.

## Baselines

Three forecasts are read off the calendar, not fit:

- Lag 1: tomorrow equals today. Test MAE 102.34 GWh, MAPE 7.73%.
- Lag 7: tomorrow equals the same weekday last week. Test MAE 54.10 GWh, MAPE 4.04%.
- Lag 364: tomorrow equals the same weekday last year. Test MAE 34.68 GWh, MAPE 2.60%.

The yearly naive is the serious baseline. Weekly seasonality is strong, and so is the annual cycle. A model that only beats lag 7 has not cleared the bar that matters.

## The search

Optuna runs a separate study for XGBoost and LightGBM. The objective is mean MAE on the four development folds. XGBoost searches trees, depth, learning rate, `min_child_weight`, row and column subsampling, L2, and gamma. LightGBM searches trees, learning rate, leaves, `min_child_samples`, subsampling, and L2. Both run with one thread and a fixed seed. There is no early stopping, so the stored tree count is the one the study picked.

Development CV MAE was 15.31 GWh for XGBoost (standard deviation 3.81 across the four folds) and 15.78 for LightGBM. The lower score is the champion. A tie would keep XGBoost. Test MAE happened to rank them the same way (18.16 vs 18.28). That agreement is luck we are allowed to notice and not a second selection step. The code chooses the champion before it scores 2017.

The served XGBoost uses 356 trees, depth 6, learning rate about 0.042. The full search took 61 seconds on the CPU recorded in the report.

## What 2017 actually did

| Model | Test MAE | Test RMSE | Test MAPE |
| --- | ---: | ---: | ---: |
| Lag 1 | 102.34 | 149.16 | 7.73% |
| Lag 7 | 54.10 | 96.70 | 4.04% |
| Lag 364 | 34.68 | 63.83 | 2.60% |
| LightGBM | 18.28 | 26.19 | 1.34% |
| XGBoost | 18.16 | 25.66 | 1.33% |

Champion test MAE is 0.524 times the yearly naive, and 0.336 times the weekly naive. Mean error is -3.50 GWh: on average the forecast is a few gigawatt-hours low, on a typical day near 1,400 GWh.

The weekly naive breaks where the week is not a copy of last week. Monday MAE is 26.62 GWh for the champion and 90.54 for lag 7. December is 20.30 versus 91.68. July is 10.51 versus 20.79, a smaller gap, on a calmer month. Gain share lines up with that story: `consumption_lag_7` is 28.1% of total gain, `is_holiday` is 18.6%, `consumption_lag_364` is 15.3%, and `month_12` is 5.2%. The four wind and solar lags together are about 0.7%. Lagged generation actuals are in the matrix and the trees mostly ignore them.

CV MAE (15.31) is lower than test MAE (18.16). That is the usual direction. The folds sit in late 2015 and 2016, and the final model has seen those rows.

## The interval

After the champion is fit on development, absolute residuals on the 90 calibration days are sorted. With `n = 90` and `alpha = 0.1`, the rank is `ceil(91 * 0.9) = 82`. The 82nd smallest absolute residual is 47.74 GWh. Every test prediction uses the band `[yhat - 47.74, yhat + 47.74]`. Mean width is 95.48 GWh.

Under exchangeable residuals this band has finite-sample coverage of at least 90%. Load residuals are not exchangeable: a holiday week does not look like a random draw from October. The empirical coverage on 2017 is 94.5%, a bit conservative. The band is the same width in July and in December, so summer intervals are looser than they need to be. A next step, if someone asks, is a normalized interval (scale the residual by a seasonal width) or quantile regression. This repository ships the constant-width version because the assumption is easy to state and the coverage is measured.

The calibration rows are not in the fit. Giving them to the model would make the point forecast slightly stronger and would void the interval. Ninety days out of a ten-year history is the trade.

## Tracking

`python -m forecasting.tracking.register` loads `models/champion` and writes a run to SQLite at `sqlite:///mlruns/mlflow.db`. A directory of `mlruns` files can store experiments. It cannot store a model registry. SQLite is the smallest store that can. Artifacts go under `mlruns/artifacts`, which is gitignored along with the database.

The run logs the test metrics, the three baseline MAEs, the interval width, the 2017 coverage, and the parameter set. The model is registered as `germany-daily-load`, and the alias `champion` points at that version. The HTTP process never opens this database. It reads `models/champion`. Promoting a new fit means writing that directory and, separately, logging it. The alias is the record of which run you claim is current.

## Serving

`POST /predict` takes `target_date` and a list of daily observations. Pydantic rejects a negative load. The feature function rejects a history that cannot build the 364-day lag, and it rejects duplicate dates. The response is the point forecast in GWh, the conformal band, and the model version.

`POST /predict/batch` scores several dates from one history. It does not append a prediction and treat it as tomorrow's actual. Multi-step forecasts would need a different evaluation. `/health` answers when the process is up. `/ready` answers when the champion loaded, and includes the stored test MAE so a caller can see which artifact is live.

A request for 2017-12-31 through the live server returns 1141.81 GWh, the same value as `reports/test_predictions.csv` for that day. The band is about 1094 to 1190 GWh. The Dockerfile runs this process with `MODEL_DIR` pointed at the copied champion. It does not install MLflow.

## Drift

`python -m forecasting.monitoring.drift` builds the same supervised frame and computes a population stability index for every model input, development versus 2017. The usual reading is: under 0.1 is a small move, 0.1 to 0.2 is worth a look, 0.2 or more means open the histogram. The script's default threshold is 0.2. CI runs the script and does not fail the job when something alerts. Solar capacity growth is a real change, and a red build would train people to ignore the check.

What the committed report shows:

| Feature | PSI | At or above 0.2 |
| --- | ---: | --- |
| `consumption_roll_mean_14` | 2.487 | yes |
| `consumption_roll_mean_28` | 2.460 | yes |
| `consumption_roll_mean_7` | 1.456 | yes |
| `wind_lag_1` | 0.756 | yes |
| `wind_lag_7` | 0.748 | yes |
| `consumption_lag_364` | 0.258 | yes |
| `consumption` (the target, not a feature) | 0.144 | no |
| `consumption_lag_1` | 0.143 | no |
| `solar_lag_1` | 0.110 | no |

Wind is the clean exogenous story. On days where the lag exists, development wind averages about 144 GWh and 2017 averages about 280 GWh. The trees give that column almost no gain, so the drift matters as monitoring context more than as an explanation of the 2017 error.

The rolling-mean PSI looks alarming next to a daily-load PSI of 0.14. Smoothing changes the shape. The bottom decile of 2006–2016 seven-day means is everything below about 1,203 GWh, and 2017 has zero days in that bin. Daily load still reaches low weekends and holidays, which is why `consumption_lag_1` does not alert. An empty bin is also clipped at `1e-6` before the log, and that clip inflates PSI. The right sentence is: 2017's weekly averages sit higher than the lowest decile of the prior decade, the metric is sensitive to that empty bin, and the holdout error is still about half of the yearly naive. PSI is an investigation flag. It is not a license to refit on the test year.

## What CI does

`.github/workflows/ci.yml` installs the dev extra, runs Ruff, runs `pytest`, trains `configs/ci.yaml` (two trials, XGBoost only, smaller windows), scores 2017-12-31 with that tiny artifact, and writes a drift report under `reports/ci/`. The tiny model is a smoke test of the entrypoint. Its MAE is not the number in the table above. `tests/test_readme.py` checks that the README, this file, and the model card still contain the champion's test MAE formatted from the JSON.

## How this would sit on Databricks or Azure

The local pieces map onto jobs, not onto a new design.

- The CSV lands in a Delta table. The feature function stays one import, used by the training job and by the scoring path, so the two cannot drift apart. A feature store is justified once more than one job needs the same offline and online definition.
- The backtest becomes a job. Optuna can stay, or the same search can be Hyperopt. The point of the search, mean MAE on expanding windows before a final year, does not change.
- SQLite MLflow becomes the workspace tracking server. The `champion` alias becomes a Unity Catalog alias, or a stage, with the same rule: the alias moves when a run earns it, and the endpoint loads that version.
- The FastAPI process becomes a serving endpoint. The request still carries history, or the endpoint reads the latest rows from the feature table. The second version is what you want once the history is not the caller's problem.
- The drift script becomes a scheduled job that writes PSI by feature into a table and pages a person when wind or the smoothed load crosses 0.2. It still should not block a deploy by itself.

On Azure ML the same roles are a pipeline job, the model registry, and a managed endpoint. Nothing in this repository calls those services. The local path is what is tested.

## Limits you will be asked about

- The file stops at the end of 2017. There is no evidence here about 2024 load, rooftop solar, or a demand shock.
- One series. A retail problem in the M5 style adds a series id and computes lags inside each series. The split logic does not change. The code here does not do that.
- No weather forecast, no price, no holiday bridge-day feature beyond the national flag.
- The interval is constant width and slightly conservative on this particular year.
- Two models were scored on the test set. Only CV chose between them. A third model family would be another look at the same year.
- The API trusts its caller and has no auth.

If the review asks for prediction intervals that vary by season, say the constant band is the shipped method, coverage on 2017 was 94.5% against a 90% target, and a scaled residual is the change you would try on the calibration window before touching 2017 again.
