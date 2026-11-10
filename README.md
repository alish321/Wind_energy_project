# Short-Term Wind Generation Forecasting for Great Britain

Hourly wind generation forecasting using weather variables, lagged generation, and time-aware machine learning, with a strict leakage audit and chronological evaluation.

**Notebook:** `code.ipynb` (Python 3.11, scikit-learn)

## Research questions

1. How accurately can hourly GB wind generation be forecast using weather variables, lagged generation features, and time-aware ML models?
2. Which model gives the best balance of accuracy, robustness, and interpretability?

---

## Data

Place these three files in the same folder as the notebook:

| File | Contents |
|------|----------|
| `time_series_60min_singleindex.csv` | Hourly electricity generation and load variables, including the target `GB_UKM_wind_generation_actual` |
| `weather_data.csv` | Country-level weather (temperature, direct and diffuse radiation) |
| `london_nwp_data.csv` | Extra weather (wind speed/direction at 10 m and 100 m, temperature, pressure, humidity, radiation) saved from the Open-Meteo archive API |

Notes:
- The notebook only *reads* `london_nwp_data.csv`. The code that downloads it from Open-Meteo is not included, so you need to supply that file (or regenerate it from the Open-Meteo archive API).
- All timestamps use the `utc_timestamp` column, converted to timezone-naive UTC.

## Method

1. **Data inspection:** duplicates, missing values, target summary and plots.
2. **Leakage audit:** same-time onshore + offshore generation has a correlation of 1.0 with the target, so those columns (and actual load and solar generation) are barred from being predictors.
3. **Feature engineering (36 features):**
   - Cyclical calendar features (hour, day of week, month as sin/cos)
   - Target lags: 1, 2, 3, 6, 12, 24, 48, 168 hours
   - Rolling mean/std computed on the **shifted** target (past values only): 3, 6, 24, 168-hour windows
   - Wind direction encoded as sin/cos
   - Capacity, ENTSO-E load forecast, and weather variables
4. **Chronological split (no shuffling):** 70% train / 15% validation / 15% test.
   - Train: 2015-01-09 → 2018-06-24
   - Validation: 2018-06-24 → 2019-03-19
   - Test: 2019-03-19 → 2019-12-30
   - 40,494 rows after dropping rows with missing lag/rolling values
5. **Baselines:** training mean, persistence (lag 1), daily persistence (lag 24), weekly persistence (lag 168).
6. **Models compared:** Ridge, ElasticNet, Linear SVR, MLP, Random Forest, Extra Trees, Gradient Boosting, HistGradientBoosting, AdaBoost. Imputation and scaling are done inside scikit-learn pipelines to avoid preprocessing leakage.
7. **Tuning:** `GridSearchCV` with `TimeSeriesSplit` on the training set only, for Extra Trees, Gradient Boosting and Random Forest.
8. **Final evaluation:** the chosen model is refit on train + validation, then scored once on the untouched test set.
9. **Extras:** Wilcoxon signed-rank test, impurity and permutation feature importance, error analysis.

Metrics: MAE, RMSE, nRMSE (by mean and by range), R², and % RMSE improvement over the best persistence baseline.

## Results (as recorded in the notebook)

**Validation (top models)**

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| Extra Trees (tuned) | 437.7 | 817.4 | 0.932 |
| Extra Trees (default) | 439.3 | 818.4 | 0.932 |
| Gradient Boosting (default) | 458.0 | 833.5 | 0.930 |
| Persistence lag 1 (best baseline) | 476.3 | 942.0 | – |

**Final test set: tuned Extra Trees**

| Metric | Value |
|--------|-------|
| MAE | 389.4 |
| RMSE | 747.2 |
| R² | 0.930 |
| nRMSE (mean / range) | 0.142 / 0.054 |
| RMSE improvement vs persistence (lag 1) | 14.9% |

- **Statistical test:** Extra Trees beats Gradient Boosting on paired validation absolute errors (Wilcoxon p ≈ 8e-26 on hourly errors, p ≈ 4e-13 on daily-aggregated errors).
- **Feature importance:** recent history dominates: `target_lag_1`, `target_rolling_mean_3`, `target_lag_2`, `target_rolling_mean_6`, `target_lag_3`. Weather (wind speed) and calendar features contribute much less.

Units of generation are not stated in the notebook (likely MW).

## Requirements

```
python >= 3.10
numpy
pandas
matplotlib
scikit-learn
scipy
jupyter
```

Install: `pip install numpy pandas matplotlib scikit-learn scipy jupyter`

## How to run

1. Put the three CSV files next to `code.ipynb`.
2. Launch Jupyter and run all cells in order. The random seed is fixed to 42.
3. Grid search over the three tree models is the slowest step.

## Outputs

Saved to the working directory:
- `fig_final_test_forecast_vs_actual.png`
- `fig_feature_importance_extra_trees.png`
- `fig_permutation_importance_extra_trees.png`

## Limitations

- Forecasts are one step ahead and lean heavily on the previous hours' actual generation. Performance over longer horizons would be worse and is not tested here.
- Weather inputs come from a single London-based archive series plus country-level weather, not site-level data for wind farms.
- Open-Meteo archive data is observed or reanalysis weather, not a true forecast, so results may be optimistic compared with live use.
- The final model is chosen on validation results, so the validation set is not fully independent for model selection.

## Project structure

```
code.ipynb                           # full workflow
time_series_60min_singleindex.csv    # input (not included)
weather_data.csv                     # input (not included)
london_nwp_data.csv                  # input (not included)
```
