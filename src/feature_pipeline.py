import os
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

from cities import CITIES
from aqi_utils import pm25_to_aqi

load_dotenv()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

# Path to our CSV-based feature store, relative to this script's location
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DATA_FILE = os.path.join(DATA_DIR, "aqi_history.csv")


def fetch_pollution_data(lat, lon):
    """Call OpenWeather's air pollution endpoint for one city and return the raw JSON."""

    url = "http://api.openweathermap.org/data/2.5/air_pollution"
    params = {"lat": lat, "lon": lon, "appid": OPENWEATHER_API_KEY}

    response = requests.get(url, params=params)
    response.raise_for_status()  # crashes loudly if the API call fails, so we notice
    return response.json()


def fetch_weather_data(lat, lon):
    """Call OpenWeather's current weather endpoint (temp, humidity, wind) for one city.

    Note: this only gives us weather going forward, not historical weather -- OpenWeather's
    free tier doesn't include historical weather, only historical pollution. So weather
    features will only be usable in training once enough new data accumulates.
    """

    url = "http://api.openweathermap.org/data/2.5/weather"
    params = {"lat": lat, "lon": lon, "appid": OPENWEATHER_API_KEY, "units": "metric"}

    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()


def build_feature_row(city_name, raw_data, weather_data=None):
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
        "weekday": now.weekday(),  # 0 = Monday, 6 = Sunday

        # raw pollutant readings
        "pm2_5": pm25,
        "pm10": components["pm10"],
        "co": components["co"],
        "no2": components["no2"],
        "o3": components["o3"],
        "so2": components["so2"],

        # calculated target
        "aqi": aqi,
    }

    # weather features, when available wind especially affects how pollutants disperse
    if weather_data is not None:
        row["temperature"] = weather_data["main"]["temp"]
        row["humidity"] = weather_data["main"]["humidity"]
        row["wind_speed"] = weather_data["wind"]["speed"]
        row["wind_deg"] = weather_data["wind"].get("deg", 0)
    else:
        row["temperature"] = None
        row["humidity"] = None
        row["wind_speed"] = None
        row["wind_deg"] = None

    return row


def add_change_rate(df):

    df = df.sort_values(["city", "timestamp"])
    df["aqi_change_rate"] = df.groupby("city")["aqi"].diff().fillna(0)
    return df


def save_to_csv(new_rows_df):

    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(DATA_FILE):
        existing_df = pd.read_csv(DATA_FILE, parse_dates=["timestamp"])
        combined_df = pd.concat([existing_df, new_rows_df], ignore_index=True)
    else:
        combined_df = new_rows_df

    # recalculate change rate across the full history
    combined_df = add_change_rate(combined_df)

    combined_df.to_csv(DATA_FILE, index=False)
    print(f"Saved {len(new_rows_df)} new rows. Feature store now has {len(combined_df)} total rows.")


def main():
    print("Fetching current AQI data for all cities...")

    all_rows = []
    for city_name, coords in CITIES.items():
        try:
            raw_data = fetch_pollution_data(coords["lat"], coords["lon"])
            try:
                weather_data = fetch_weather_data(coords["lat"], coords["lon"])
            except Exception:
                weather_data = None  # don't let a weather API hiccup break the pollution data
            row = build_feature_row(city_name, raw_data, weather_data)
            all_rows.append(row)
            print(f"  {city_name}: AQI = {row['aqi']}")
        except Exception as e:
            print(f"  Failed to fetch data for {city_name}: {e}")

    df = pd.DataFrame(all_rows)
    save_to_csv(df)


if __name__ == "__main__":
    main()