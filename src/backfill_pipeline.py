"""
Historical Backfill for Pearls AQI Predictor

Fetches real historical AQI data for the past N days (default 30) for all 8 cities
using OpenWeather's Air Pollution History API, and merges it into data/aqi_history.csv.
This gives us enough historical data to actually train a model, instead of waiting
hour by hour for the live pipeline to build up data naturally.

Run this manually, once, with: python backfill_pipeline.py
Pass a different number of days like: python backfill_pipeline.py 60
"""

import os
import sys
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

from cities import CITIES
from aqi_utils import pm25_to_aqi

load_dotenv()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DATA_FILE = os.path.join(DATA_DIR, "aqi_history.csv")

DAYS_BACK = int(sys.argv[1]) if len(sys.argv) > 1 else 30


def fetch_history(lat, lon, start_unix, end_unix):
    """Call OpenWeather's air pollution history endpoint for one city and date range."""

    url = "http://api.openweathermap.org/data/2.5/air_pollution/history"
    params = {
        "lat": lat,
        "lon": lon,
        "start": start_unix,
        "end": end_unix,
        "appid": OPENWEATHER_API_KEY,
    }

    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()


def build_feature_row(city_name, entry):
    """Turn one historical entry into a feature row, same shape as the live pipeline."""

    components = entry["components"]
    timestamp = datetime.utcfromtimestamp(entry["dt"])

    pm25 = components["pm2_5"]
    aqi = pm25_to_aqi(pm25)

    row = {
        "city": city_name,
        "timestamp": timestamp,
        "date": timestamp.strftime("%Y-%m-%d"),
        "hour": timestamp.hour,
        "day": timestamp.day,
        "month": timestamp.month,
        "weekday": timestamp.weekday(),
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


def main():
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=DAYS_BACK)
    start_unix = int(start_time.timestamp())
    end_unix = int(end_time.timestamp())

    print(f"Backfilling the last {DAYS_BACK} days of AQI data for all cities...")

    all_rows = []
    for city_name, coords in CITIES.items():
        try:
            data = fetch_history(coords["lat"], coords["lon"], start_unix, end_unix)
            entries = data.get("list", [])
            for entry in entries:
                all_rows.append(build_feature_row(city_name, entry))
            print(f"  {city_name}: fetched {len(entries)} historical readings")
        except Exception as e:
            print(f"  Failed to fetch history for {city_name}: {e}")

        time.sleep(1)  # be polite to the free-tier rate limit

    new_df = pd.DataFrame(all_rows)

    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(DATA_FILE):
        existing_df = pd.read_csv(DATA_FILE, parse_dates=["timestamp"])
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        # remove exact duplicate readings (same city + same timestamp)
        combined_df = combined_df.drop_duplicates(subset=["city", "timestamp"])
    else:
        combined_df = new_df

    combined_df = add_change_rate(combined_df)
    combined_df.to_csv(DATA_FILE, index=False)

    print(f"Backfill complete. Feature store now has {len(combined_df)} total rows.")


if __name__ == "__main__":
    main()