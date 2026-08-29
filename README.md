# Pearls AQI Predictor — Final Report

## Overview

An end-to-end machine learning system that forecasts Air Quality Index (AQI) up
to 7 days ahead for 8 major Pakistani cities: Lahore, Faisalabad, Karachi,
Islamabad, Multan, Peshawar, Rawalpindi, and Gujranwala.

**Live dashboard:** [add your Streamlit Cloud link here after deploying]
**Repository:** https://github.com/alijawad480/pearls-aqi-predictor

## Architecture

```
OpenWeather API  -->  Feature Pipeline (hourly)  -->  data/aqi_history.csv (feature store)
                                                              |
                                                              v
                                              Training Pipeline (daily)
                                                              |
                                                              v
                                                   models/ (model registry)
                                                              |
                                                              v
                                              Streamlit Dashboard (live)
```

Both pipelines run automatically via GitHub Actions (feature pipeline hourly,
training pipeline daily) and commit their outputs directly back into this
repo, which acts as a lightweight, fully version-controlled feature store and
model registry.

## A significant engineering detour: the Hopsworks issue

The project originally used Hopsworks (as suggested in the project brief) as
the feature store and model registry. During implementation, every attempt to
write data to Hopsworks' offline feature store failed with the same low-level
error:

```
OSError: Kernel error -> Generic HdfsObjectStore error
  IO error occurred while communicating with HDFS
  RPC listener disconnected
```

This was investigated systematically:

- Reproduced identically on Windows and Linux (WSL)
- Reproduced identically on home WiFi and mobile cellular data
- Reproduced identically across two different Hopsworks client versions (4.2, 5.0)
- Reproduced identically across two separate, freshly-created Hopsworks projects
- Reproduced identically when run from GitHub Actions' cloud servers (which
  have unrestricted public network access, ruling out any local network/NAT
  cause)

Since the failure was consistent across every operating system, network, and
account variable that could be controlled, it was concluded this was a bug or
infrastructure issue specific to the Hopsworks `eu-west` serverless cluster's
offline write path, not something fixable from the client side.

**Resolution:** rather than block the project on an external dependency issue,
the feature store and model registry were re-implemented using CSV files and
joblib model files committed directly to this Git repository, with GitHub
Actions handling the automated updates. This is a simpler architecture that
still satisfies the core requirement (a versioned, queryable historical
dataset and model registry), while remaining fully reliable.

## Data pipeline

- **Source:** OpenWeather Air Pollution API (pollutant concentrations) and
  Current Weather API (temperature, humidity, wind speed/direction)
- **AQI calculation:** OpenWeather's own AQI scale (1-5) is not the standard
  scale most people expect. Instead, the real US EPA AQI (0-500) is calculated
  directly from the PM2.5 reading using the standard EPA breakpoint formula,
  matching what services like IQAir report.
- **Feature engineering:** time-based features (hour, day, month, weekday),
  AQI change rate, and lag features (AQI on each of the past 7 days) used as
  model inputs
- **Historical backfill:** 30 days of real historical data pulled via
  OpenWeather's history endpoint, giving ~700 hourly readings per city

## Modeling approach

For each of the 7 forecast horizons (day 1 through day 7), four model types
are trained and compared: Ridge Regression, Random Forest, a neural network
(scikit-learn MLPRegressor), and XGBoost. Whichever performs best on a
time-based holdout set (most recent 20% of data, to avoid lookahead bias) is
kept and saved.

**Note on dataset size:** with ~30 days of history, only a limited number of
usable 7-day-lookback / 7-day-forecast training examples exist per city.
Accuracy is expected to improve steadily as the automated hourly pipeline
continues accumulating more historical data over time.

### Current results (see models/training_summary.json for latest)

| Horizon | Best model | RMSE | MAE | R² |
|---|---|---|---|---|
| Day 1 | Ridge | 15.2 | 12.6 | 0.72 |
| Day 2 | Random Forest | 13.1 | 11.4 | 0.78 |
| Day 3 | Random Forest | 18.5 | 14.2 | 0.60 |
| Day 4 | Neural Net | 21.1 | 17.7 | 0.48 |
| Day 5 | Neural Net | 25.1 | 19.3 | 0.28 |
| Day 6 | Neural Net | 22.1 | 18.2 | 0.46 |
| Day 7 | Ridge | 19.1 | 16.0 | 0.63 |

As expected, accuracy is strongest for near-term forecasts and degrades for
longer horizons -- a normal pattern in time series forecasting, most
pronounced here due to the limited training data currently available.

## Explainability

SHAP (SHapley Additive exPlanations) values are computed for the day-1
forecast model, showing which features (which lag days, which time-of-year
signals) most influence each prediction. Available in the dashboard under
"Why the model predicts this."

## Dashboard features

- City selector with live current AQI and category (Good/Moderate/Unhealthy/etc.)
- National map overview, color-coded by current AQI across all 8 cities
- 7-day forecast chart, color-coded by AQI severity
- 30-day historical trend chart
- Hazardous AQI alerts
- SHAP-based feature importance panel
- Green/gray theme matching Pearls branding

## Future improvements

- Incorporate weather features (temperature, humidity, wind) into the
  training pipeline once enough weather history has accumulated
- Add per-city models as data volume grows, rather than one pooled model
- Add prediction intervals (uncertainty bounds) alongside point forecasts
- Hyperparameter tuning via RandomizedSearchCV
- Migrate from CSV to SQLite as the dataset grows larger

## Tech stack

Python, pandas, scikit-learn, XGBoost, SHAP, Streamlit, Plotly, GitHub
Actions, OpenWeather API.