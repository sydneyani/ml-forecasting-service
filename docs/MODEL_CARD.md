# Model card: Germany daily load

## Details

| | |
| --- | --- |
| Name | `germany-daily-load` |
| Version | `5828f2d8d5a4` (SHA-256 prefix of `models/champion/model.json`) |
| Type | XGBoost regressor, `tree_method=hist`, 356 trees, max depth 6 |
| Target | Consumption on day `t`, gigawatt-hours |
| Horizon | 1 day |
| Interval | Split conformal, absolute residual, nominal 90%, half-width 47.74 GWh |
| Training stack | pandas 3.1.1, numpy 2.5.5, scikit-learn 1.9.1, XGBoost 3.4.1, LightGBM 4.7.0, Optuna 5.1.0 |

LightGBM was tuned in the same search and lost on development CV (15.78 vs 15.31 GWh). It is not served.

## Intended use

A worked example of a day-ahead load forecast that a reviewer can run locally: feature contract, backtest, registry, and HTTP scoring. The decision the model imitates is "what is tomorrow's national consumption, given history through today."

## Out of scope

Dispatch, trading, or any live grid decision. The series ends on 2017-12-31. The model is one national total, not a zone, a meter, or a customer. It does not take a weather forecast. Same-day wind and solar in the file are generation actuals, and they are not inputs.

## Data

Open Power System Data, Germany, daily, 2006-01-01 through 2017-12-31. Consumption, wind, and solar are daily energy in GWh. Wind is missing early in the file and solar is missing until the early 2010s. Those gaps stay missing in the matrix.

Rows after a 364-day warmup: 4,019. Development fit: 2006-12-31 through 2016-10-02 (3,564 days). Calibration, used only for the interval: 2016-10-03 through 2016-12-31 (90 days). Test: 2017-01-01 through 2017-12-31 (365 days).

## Evaluation

Expanding-window CV inside the development period, four blocks of 90 days covering 2015-10-09 through 2016-10-02. Optuna minimized mean validation MAE. The champion is the lower development CV MAE, with ties broken toward XGBoost. Test metrics were computed after that choice.

| Model | CV MAE | Test MAE | Test RMSE | Test MAPE |
| --- | ---: | ---: | ---: | ---: |
| Naive lag-1 |  | 102.34 | 149.16 | 7.73% |
| Seasonal naive, lag 7 |  | 54.10 | 96.70 | 4.04% |
| Seasonal naive, lag 364 |  | 34.68 | 63.83 | 2.60% |
| LightGBM | 15.78 | 18.28 | 26.19 | 1.34% |
| XGBoost (champion) | 15.31 | 18.16 | 25.66 | 1.33% |

Test MAE is 0.524 times the lag-364 seasonal naive. Mean error is -3.50 GWh (the model forecasts slightly low). The 90% interval covered 94.5% of 2017. Monday MAE is 26.62 GWh against 90.54 for the weekly naive. December MAE is 20.30 GWh against 91.68.

The largest gain shares are `consumption_lag_7` (28.1%), `is_holiday` (18.6%), and `consumption_lag_364` (15.3%). The four wind and solar lags together are about 0.7% of total gain.

## Ethical considerations and caveats

The series is a national aggregate. It has no personal data. Errors are in public energy totals, not in a decision about an individual.

Residuals are not exchangeable across seasons, so the conformal coverage guarantee is a statement about the calibration window, not a promise about the next winter. The empirical 2017 coverage is the number to quote next to the nominal 90%. The band is a constant width, so it is wider than it needs to be on quiet summer days and can still miss a holiday week.

2017 wind generation is higher than the development years (PSI 0.76 on `wind_lag_1`). The model barely uses that feature. A later year with a different load shape is outside the evidence in this repository.

Reproduce the table with `python -m forecasting.training --config configs/train.yaml`. The command overwrites `models/champion` and `reports/backtest_results.json`.
