"""
Feature Pipeline for Pearls AQI Predictor

What this script does:
1. Fetches current weather + pollution data for each city from OpenWeather
2. Calculates the real AQI from PM2.5, plus some extra features
3. Appends everything to data/aqi_history.csv, which acts as our feature store

Note: we originally planned to use Hopsworks as the feature store, but hit a
persistent, unfixable bug in Hopsworks' offline write path (confirmed to fail
identically across Windows, Linux, home wifi, mobile data, and even GitHub's
own cloud servers -- ruling out any issue on our end). A CSV file committed
back to this repo is a simple, reliable substitute that still satisfies the
"feature store" role: a growing, versioned historical dataset.

Run this manually with: python feature_pipeline.py
GitHub Actions runs it automatically every hour and commits the updated CSV.
"""

import os
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

from cities import CITIES
from aqi_utils import pm25_to_aqi

load_dotenv()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DATA_FILE = os.path.join(DATA_DIR, "aqi_history.csv")


def fetch_pollution_data(lat, lon):
    """Call OpenWeather's air pollution endpoint for one city and return the raw JSON."""

    url = "http://api.openweathermap.org/data/2.5/air_pollution"
    params = {"lat": lat, "lon": lon, "appid": OPENWEATHER_API_KEY}

    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()


def build_feature_row(city_name, raw_data):
    """Turn the raw API response into one clean row of features for a single city."""

    components = raw_data["list"][0]["components"]
    now = datetime.utcnow()

    pm25 = components["pm2_5"]
    aqi = pm25_to_aqi(pm25)

    row = {
        "city": city_name,
        "timestamp": now,
        "date": now.strftime("%Y-%m-%d"),
        "hour": now.hour,
        "day": now.day,
        "month": now.month,
        "weekday": now.weekday(),

        "pm2_5": pm25,
        "pm10": components["pm10"],
        "co": components["co"],
        "no2": components["no2"],
        "o3": components["o3"],
        "so2": components["so2"],

        "aqi": aqi,
    }
    return row


def add_change_rate(df):
    """Add a column showing how much AQI moved since the previous row, per city."""

    df = df.sort_values(["city", "timestamp"])
    df["aqi_change_rate"] = df.groupby("city")["aqi"].diff().fillna(0)
    return df


def save_to_csv(new_rows_df):
    """Append the new rows to our CSV feature store, creating it if it doesn't exist yet."""

    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(DATA_FILE):
        existing_df = pd.read_csv(DATA_FILE, parse_dates=["timestamp"])
        combined_df = pd.concat([existing_df, new_rows_df], ignore_index=True)
    else:
        combined_df = new_rows_df

    combined_df = add_change_rate(combined_df)

    combined_df.to_csv(DATA_FILE, index=False)
    print(f"Saved {len(new_rows_df)} new rows. Feature store now has {len(combined_df)} total rows.")


def main():
    print("Fetching current AQI data for all cities...")

    all_rows = []
    for city_name, coords in CITIES.items():
        try:
            raw_data = fetch_pollution_data(coords["lat"], coords["lon"])
            row = build_feature_row(city_name, raw_data)
            all_rows.append(row)
            print(f"  {city_name}: AQI = {row['aqi']}")
        except Exception as e:
            print(f"  Failed to fetch data for {city_name}: {e}")

    df = pd.DataFrame(all_rows)
    save_to_csv(df)


if __name__ == "__main__":
    main()
