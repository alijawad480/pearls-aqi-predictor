# Pearls AQI Predictor

End-to-end AQI forecasting system for Pakistani cities. Built as a serverless pipeline:
OpenWeather (data) -> Hopsworks (feature store + model registry) -> Streamlit (dashboard).

## Setup (do this once)

1. Install Python 3.10+ if you don't have it.
2. Open this folder in VS Code.
3. Create a virtual environment and install dependencies:

   ```
   python -m venv venv
   venv\Scripts\activate        (on Windows)
   source venv/bin/activate     (on Mac/Linux)
   pip install -r requirements.txt
   ```

4. Copy `.env.example` to a new file called `.env`
5. Fill in your two API keys inside `.env`:
   - `OPENWEATHER_API_KEY` — from openweathermap.org
   - `HOPSWORKS_API_KEY` — from your Hopsworks project settings

## Run the feature pipeline

```
cd src
python feature_pipeline.py
```

This fetches live AQI data for 8 cities and pushes it to your Hopsworks feature store.
Run it a few times over the next days/weeks to start building up history — or better,
run the backfill script (coming in Phase 2) to generate historical data faster.

## Project status

- [x] Phase 1: Feature pipeline (this script)
- [ ] Phase 2: Historical backfill
- [ ] Phase 3: Training pipeline (Ridge, Random Forest, LSTM for 7-day forecast)
- [ ] Phase 4: GitHub Actions automation
- [ ] Phase 5: Streamlit dashboard (green/gray Pearls theme)
