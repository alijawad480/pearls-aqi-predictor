import os
import json
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")  # no display needed, we just save the plot to a file
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DATA_FILE = os.path.join(DATA_DIR, "aqi_history.csv")

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

FORECAST_HORIZON_DAYS = 7   # how many days ahead we predict
LOOKBACK_DAYS = 7           # how many past days of AQI we use as input features


def load_daily_data():
    """Load the raw hourly data and collapse it down to one row per city per day."""

    df = pd.read_csv(DATA_FILE, parse_dates=["timestamp"])
    df["date"] = df["timestamp"].dt.date

    daily = (
        df.groupby(["city", "date"])
        .agg(
            aqi=("aqi", "mean"),
            pm2_5=("pm2_5", "mean"),
            pm10=("pm10", "mean"),
            co=("co", "mean"),
            no2=("no2", "mean"),
            o3=("o3", "mean"),
            so2=("so2", "mean"),
        )
        .reset_index()
    )
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values(["city", "date"])
    return daily


def build_supervised_dataset(daily):

    rows = []

    for city, city_df in daily.groupby("city"):
        city_df = city_df.reset_index(drop=True)
        aqi_values = city_df["aqi"].values
        dates = city_df["date"].values
        n = len(city_df)

        # we need LOOKBACK_DAYS of history before, and FORECAST_HORIZON_DAYS after,
        # so we can only build a row for the days in between
        for i in range(LOOKBACK_DAYS, n - FORECAST_HORIZON_DAYS):
            row = {"city": city, "date": dates[i]}

            # input features: AQI on each of the past LOOKBACK_DAYS days
            for lag in range(1, LOOKBACK_DAYS + 1):
                row[f"aqi_lag_{lag}"] = aqi_values[i - lag]

            row["month"] = pd.Timestamp(dates[i]).month
            row["weekday"] = pd.Timestamp(dates[i]).weekday()

            # targets: AQI on each of the next FORECAST_HORIZON_DAYS days
            for horizon in range(1, FORECAST_HORIZON_DAYS + 1):
                row[f"aqi_target_day_{horizon}"] = aqi_values[i + horizon]

            rows.append(row)

    return pd.DataFrame(rows)


def train_and_evaluate(X_train, y_train, X_test, y_test):
    """Train Ridge, Random Forest, and a small neural net; return whichever scores best."""

    candidates = {
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42),
        "neural_net": MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=2000, random_state=42),
        "xgboost": XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42),
    }

    results = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        results[name] = {"model": model, "rmse": rmse, "mae": mae, "r2": r2}

    best_name = min(results, key=lambda name: results[name]["rmse"])
    return best_name, results


def main():
    print("Loading and preparing data...")
    daily = load_daily_data()
    dataset = build_supervised_dataset(daily)

    if len(dataset) < 20:
        print(f"Only {len(dataset)} training rows available -- this is thin, but we'll")
        print("proceed anyway. Accuracy will improve as more historical data accumulates.")

    # one-hot encode the city so the model can distinguish between cities
    dataset = pd.get_dummies(dataset, columns=["city"], prefix="city")

    lag_cols = [f"aqi_lag_{i}" for i in range(1, LOOKBACK_DAYS + 1)]
    city_cols = [c for c in dataset.columns if c.startswith("city_")]
    feature_cols = lag_cols + ["month", "weekday"] + city_cols

    # time-based split: oldest 80% of rows for training, most recent 20% for testing
    dataset = dataset.sort_values("date")
    split_idx = int(len(dataset) * 0.8)
    train_df = dataset.iloc[:split_idx]
    test_df = dataset.iloc[split_idx:]

    os.makedirs(MODELS_DIR, exist_ok=True)

    scaler = StandardScaler()
    X_train_all = scaler.fit_transform(train_df[feature_cols])
    X_test_all = scaler.transform(test_df[feature_cols])

    summary = []
    day_1_model = None
    day_1_model_name = None

    for horizon in range(1, FORECAST_HORIZON_DAYS + 1):
        target_col = f"aqi_target_day_{horizon}"
        y_train = train_df[target_col].values
        y_test = test_df[target_col].values

        print(f"\nTraining models for day {horizon} forecast...")
        best_name, results = train_and_evaluate(X_train_all, y_train, X_test_all, y_test)
        best = results[best_name]

        print(f"  Best model: {best_name} (RMSE={best['rmse']:.2f}, MAE={best['mae']:.2f}, R2={best['r2']:.2f})")

        model_path = os.path.join(MODELS_DIR, f"day_{horizon}_model.joblib")
        joblib.dump(best["model"], model_path)

        if horizon == 1:
            day_1_model = best["model"]
            day_1_model_name = best_name

        summary.append({
            "horizon_days": horizon,
            "best_model": best_name,
            "rmse": round(best["rmse"], 3),
            "mae": round(best["mae"], 3),
            "r2": round(best["r2"], 3),
        })

    # SHAP explainability for the day-1 model (most immediately useful forecast)
    print("\nComputing SHAP feature importance for the day-1 model...")
    try:
        X_train_df = pd.DataFrame(X_train_all, columns=feature_cols)
        explainer = shap.Explainer(day_1_model.predict, X_train_df)
        shap_values = explainer(X_train_df)

        mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
        importance = sorted(
            zip(feature_cols, mean_abs_shap.tolist()),
            key=lambda pair: pair[1],
            reverse=True,
        )
        with open(os.path.join(MODELS_DIR, "shap_importance.json"), "w") as f:
            json.dump({"model_used": day_1_model_name, "importance": importance}, f, indent=2)

        plt.figure()
        shap.summary_plot(shap_values, X_train_df, show=False, plot_type="bar")
        plt.tight_layout()
        plt.savefig(os.path.join(MODELS_DIR, "shap_summary.png"), dpi=120)
        plt.close()
        print("  SHAP importance saved to models/shap_importance.json and shap_summary.png")
    except Exception as e:
        print(f"  Could not compute SHAP values: {e}")

    # save the scaler and feature column order so the dashboard can reuse them exactly
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))
    with open(os.path.join(MODELS_DIR, "feature_columns.json"), "w") as f:
        json.dump(feature_cols, f, indent=2)
    with open(os.path.join(MODELS_DIR, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\nTraining complete. Models saved to models/")
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()