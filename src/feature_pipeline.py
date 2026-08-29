"""
Feature Pipeline for Pearls AQI Predictor

What this script does:
1. Fetches current weather + pollution data for each city from OpenWeather
2. Calculates the real AQI from PM2.5, plus some extra features
3. Pushes everything to the Hopsworks Feature Store

Run this manually for now with: python feature_pipeline.py
Later, GitHub Actions will run this automatically every hour.
"""

import os
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import hopsworks

from cities import CITIES
from aqi_utils import pm25_to_aqi

load_dotenv()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")


def fetch_pollution_data(lat, lon):
    """Call OpenWeather's air pollution endpoint for one city and return the raw JSON."""

    url = "http://api.openweathermap.org/data/2.5/air_pollution"
    params = {"lat": lat, "lon": lon, "appid": OPENWEATHER_API_KEY}

    response = requests.get(url, params=params)
    response.raise_for_status()  # crashes loudly if the API call fails, so we notice
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


def save_to_hopsworks(df):
    """Connect to Hopsworks and insert the features into (or create) the feature group."""

    project = hopsworks.login(host="eu-west.cloud.hopsworks.ai", api_key_value=HOPSWORKS_API_KEY)
    feature_store = project.get_feature_store()

    feature_group = feature_store.get_or_create_feature_group(
        name="aqi_features",
        version=1,
        description="Hourly AQI features per city for Pearls AQI Predictor",
        primary_key=["city", "timestamp"],
        event_time="timestamp",
        online_enabled=True,
    )

    feature_group.insert(df, write_options={"start_offline_materialization": False})
    print(f"Saved {len(df)} rows to Hopsworks feature group 'aqi_features'.")


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
    df = add_change_rate(df)

    save_to_hopsworks(df)


if __name__ == "__main__":
    main()
