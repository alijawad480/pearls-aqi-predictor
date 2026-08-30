import os
import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from cities import CITIES
from aqi_utils import aqi_category

DATA_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "aqi_history.csv")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

LOOKBACK_DAYS = 7
FORECAST_HORIZON_DAYS = 7

CATEGORY_COLORS = {
    "green": "#4CAF50",
    "yellow": "#FBC02D",
    "orange": "#FF9800",
    "red": "#E53935",
    "purple": "#8E24AA",
    "maroon": "#6D1B1B",
}

st.set_page_config(page_title="Pearls AQI Predictor", page_icon=":cloud:", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }

.pearls-header {
    background: linear-gradient(120deg, #0B3D2E 0%, #1B5E20 35%, #2E7D32 60%, #66BB6A 100%);
    background-size: 200% 200%;
    animation: shine 6s ease infinite;
    padding: 28px 32px;
    border-radius: 16px;
    margin-bottom: 24px;
    box-shadow: 0 8px 24px rgba(46,125,50,0.25);
}
@keyframes shine {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
.pearls-header h1 {
    color: white;
    margin: 0;
    font-weight: 700;
    letter-spacing: 0.5px;
}
.pearls-header p {
    color: #E8F5E9;
    margin: 4px 0 0 0;
    font-size: 15px;
}

div[data-testid="stMetric"] {
    background: linear-gradient(145deg, #ffffff, #ECEFF1);
    border: 1px solid #CFD8DC;
    border-radius: 14px;
    padding: 16px 20px;
    box-shadow: 0 4px 14px rgba(38,50,56,0.08);
}

.stTabs [data-baseweb="tab-list"] { gap: 4px; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=3600)
def load_daily_data():
    """Load the raw hourly CSV and collapse it to one AQI value per city per day."""

    df = pd.read_csv(DATA_FILE, parse_dates=["timestamp"])
    df["date"] = df["timestamp"].dt.date
    daily = df.groupby(["city", "date"]).agg(aqi=("aqi", "mean")).reset_index()
    daily["date"] = pd.to_datetime(daily["date"])
    return daily.sort_values(["city", "date"])


@st.cache_data(ttl=3600)
def load_raw_data():
    """Load the full hourly CSV, unaggregated -- used for EDA (correlations, patterns)."""

    df = pd.read_csv(DATA_FILE, parse_dates=["timestamp"])
    return df


@st.cache_resource
def load_models():
    """Load the 7 trained models, the scaler, and the feature column order."""

    models = {}
    for horizon in range(1, FORECAST_HORIZON_DAYS + 1):
        path = os.path.join(MODELS_DIR, f"day_{horizon}_model.joblib")
        if os.path.exists(path):
            models[horizon] = joblib.load(path)

    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))

    with open(os.path.join(MODELS_DIR, "feature_columns.json")) as f:
        feature_cols = json.load(f)

    summary = []
    summary_path = os.path.join(MODELS_DIR, "training_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)

    return models, scaler, feature_cols, summary


def build_input_row(city, daily_df, feature_cols):
    """Build one input row (most recent 7 days of AQI + date info) for a given city."""

    city_df = daily_df[daily_df["city"] == city].sort_values("date")
    if len(city_df) < LOOKBACK_DAYS:
        return None

    last_values = city_df["aqi"].values[-LOOKBACK_DAYS:][::-1]  # most recent first

    row = {}
    for lag in range(1, LOOKBACK_DAYS + 1):
        row[f"aqi_lag_{lag}"] = last_values[lag - 1]

    today = datetime.utcnow()
    row["month"] = today.month
    row["weekday"] = today.weekday()

    for col in feature_cols:
        if col.startswith("city_"):
            row[col] = 1 if col == f"city_{city}" else 0

    ordered = [row.get(col, 0) for col in feature_cols]
    return np.array(ordered).reshape(1, -1)


st.markdown("""
<div class="pearls-header">
    <h1>Pearls AQI Predictor</h1>
    <p>7-day Air Quality Index forecasts for major Pakistani cities</p>
</div>
""", unsafe_allow_html=True)

daily_df = load_daily_data()
models, scaler, feature_cols, summary = load_models()

# --- National overview map ---
latest_per_city = daily_df.sort_values("date").groupby("city").last().reset_index()
if not latest_per_city.empty:
    latest_per_city["lat"] = latest_per_city["city"].map(lambda c: CITIES[c]["lat"])
    latest_per_city["lon"] = latest_per_city["city"].map(lambda c: CITIES[c]["lon"])
    latest_per_city["category"] = latest_per_city["aqi"].apply(lambda v: aqi_category(int(v))[0])

    st.subheader("National Overview")

    legend_items = [
        ("Good (0-50)", "#4CAF50"),
        ("Moderate (51-100)", "#FBC02D"),
        ("Unhealthy for Sensitive (101-150)", "#FF9800"),
        ("Unhealthy (151-200)", "#E53935"),
        ("Very Unhealthy (201-300)", "#8E24AA"),
        ("Hazardous (300+)", "#6D1B1B"),
    ]
    legend_html = '<div style="display:flex; flex-wrap:wrap; gap:16px; padding:10px 4px 18px 4px;">'
    for label, color in legend_items:
        legend_html += (
            f'<div style="display:flex; align-items:center; gap:6px;">'
            f'<span style="width:14px; height:14px; border-radius:4px; background:{color}; '
            f'display:inline-block; box-shadow:0 1px 3px rgba(0,0,0,0.2);"></span>'
            f'<span style="font-size:13px; color:#37474F; font-family:Poppins;">{label}</span>'
            f'</div>'
        )
    legend_html += "</div>"
    st.markdown(legend_html, unsafe_allow_html=True)

    marker_style = st.radio("Marker style", ["Circles", "Location pins"], horizontal=True, label_visibility="collapsed")

    map_fig = go.Figure()

    def aqi_to_color(v):
        _, key = aqi_category(int(v))
        return CATEGORY_COLORS.get(key, "#9E9E9E")

    latest_per_city["marker_color"] = latest_per_city["aqi"].apply(aqi_to_color)

    if marker_style == "Circles":
        # soft glow layer (larger, semi-transparent, behind the real marker)
        map_fig.add_trace(go.Scattergeo(
            lat=latest_per_city["lat"], lon=latest_per_city["lon"],
            mode="markers",
            marker=dict(
                size=latest_per_city["aqi"] / 5 + 6,
                color=latest_per_city["marker_color"],
                opacity=0.25,
                line=dict(width=0),
            ),
            hoverinfo="skip",
            showlegend=False,
        ))

        # crisp marker on top, with city labels
        map_fig.add_trace(go.Scattergeo(
            lat=latest_per_city["lat"], lon=latest_per_city["lon"],
            mode="markers+text",
            text=latest_per_city["city"],
            textposition="top center",
            textfont=dict(size=11, color="#263238", family="Poppins"),
            marker=dict(
                size=latest_per_city["aqi"] / 10 + 6,
                color=latest_per_city["marker_color"],
                line=dict(width=2, color="white"),
                opacity=0.95,
            ),
            customdata=latest_per_city[["aqi", "category"]],
            hovertemplate="<b>%{text}</b><br>AQI: %{customdata[0]:.0f}<br>%{customdata[1]}<extra></extra>",
            showlegend=False,
        ))
    else:
        # pin-style: a location emoji sized/colored by severity, city name below it
        pin_sizes = latest_per_city["aqi"].apply(lambda v: 18 + min(int(v) // 20, 14))

        map_fig.add_trace(go.Scattergeo(
            lat=latest_per_city["lat"], lon=latest_per_city["lon"],
            mode="text",
            text=["📍"] * len(latest_per_city),
            textfont=dict(size=pin_sizes, color=latest_per_city["marker_color"]),
            customdata=latest_per_city[["city", "aqi", "category"]],
            hovertemplate="<b>%{customdata[0]}</b><br>AQI: %{customdata[1]:.0f}<br>%{customdata[2]}<extra></extra>",
            showlegend=False,
        ))

        map_fig.add_trace(go.Scattergeo(
            lat=latest_per_city["lat"], lon=latest_per_city["lon"],
            mode="text",
            text=latest_per_city["city"],
            textposition="bottom center",
            textfont=dict(size=11, color="#263238", family="Poppins"),
            hoverinfo="skip",
            showlegend=False,
        ))

    map_fig.update_geos(
        center=dict(lat=30.3753, lon=69.3451),
        projection_scale=9,
        showcountries=True, countrycolor="#90A4AE",
        showland=True, landcolor="#F4F6F5",
        showocean=True, oceancolor="#E3EDEA",
        showlakes=False,
        bgcolor="rgba(0,0,0,0)",
    )
    map_fig.update_layout(
        height=460,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(map_fig, use_container_width=True)

city = st.selectbox("Select a city", list(CITIES.keys()))

city_daily = daily_df[daily_df["city"] == city].sort_values("date")

if city_daily.empty:
    st.warning("No data available yet for this city. Check back once the pipeline has run a few times.")
else:
    latest_aqi = city_daily["aqi"].iloc[-1]
    category, color_name = aqi_category(int(latest_aqi))

    col1, col2 = st.columns([1, 2])

    with col1:
        st.metric("Current AQI", f"{latest_aqi:.0f}", category)

        if latest_aqi > 150:
            st.error(f"Hazardous alert: AQI is **{category}** in {city}. Limiting outdoor activity is advised.")
        elif latest_aqi > 100:
            st.warning(f"AQI is **{category}** in {city}.")
        else:
            st.success(f"AQI is **{category}** in {city}.")

    input_row = build_input_row(city, daily_df, feature_cols)

    if input_row is not None:
        scaled_input = scaler.transform(input_row)

        forecast_values = []
        for horizon in range(1, FORECAST_HORIZON_DAYS + 1):
            if horizon in models:
                prediction = models[horizon].predict(scaled_input)[0]
                forecast_values.append(prediction)
            else:
                forecast_values.append(None)

        forecast_dates = [
            (datetime.utcnow() + timedelta(days=h)).strftime("%a %d %b")
            for h in range(1, FORECAST_HORIZON_DAYS + 1)
        ]

        bar_colors = []
        for value in forecast_values:
            if value is not None:
                _, color_key = aqi_category(int(value))
                bar_colors.append(CATEGORY_COLORS.get(color_key, "#9E9E9E"))
            else:
                bar_colors.append("#9E9E9E")

        fig = go.Figure(
            data=[
                go.Bar(
                    x=forecast_dates,
                    y=forecast_values,
                    marker=dict(
                        color=bar_colors,
                        line=dict(width=1.5, color="white"),
                    ),
                    text=[f"{v:.0f}" if v is not None else "-" for v in forecast_values],
                    textposition="outside",
                    textfont=dict(size=13, color="#263238", family="Poppins"),
                )
            ]
        )
        fig.update_layout(
            title=dict(text=f"7-Day AQI Forecast for {city}", font=dict(size=18, family="Poppins", color="#1B5E20")),
            yaxis_title="Predicted AQI",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#37474F", family="Poppins"),
            bargap=0.35,
            yaxis=dict(gridcolor="#ECEFF1"),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough historical data yet to generate a 7-day forecast for this city.")

    # --- Historical trend chart ---
    st.subheader(f"Past 30 Days — {city}")
    recent_history = city_daily.tail(30)
    trend_fig = go.Figure(data=[go.Scatter(
        x=recent_history["date"], y=recent_history["aqi"],
        mode="lines+markers",
        line=dict(color="#2E7D32", width=3),
        marker=dict(size=6, color="#2E7D32"),
        fill="tozeroy",
        fillcolor="rgba(46,125,50,0.08)",
    )])
    trend_fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#37474F", family="Poppins"),
        yaxis=dict(title="AQI", gridcolor="#ECEFF1"),
        xaxis=dict(gridcolor="#ECEFF1"),
    )
    st.plotly_chart(trend_fig, use_container_width=True)

    with st.expander("Model performance details"):
        if summary:
            st.dataframe(pd.DataFrame(summary), use_container_width=True)
        st.caption(
            "Models are retrained daily as more historical data accumulates via GitHub Actions. "
            "Forecast accuracy improves over time as the dataset grows."
        )

    with st.expander("Why the model predicts this (feature importance)"):
        shap_path = os.path.join(MODELS_DIR, "shap_importance.json")
        shap_plot_path = os.path.join(MODELS_DIR, "shap_summary.png")
        if os.path.exists(shap_path):
            with open(shap_path) as f:
                shap_data = json.load(f)
            st.caption(f"Based on the day-1 forecast model ({shap_data['model_used']})")
            importance_df = pd.DataFrame(shap_data["importance"], columns=["feature", "mean_abs_shap_value"])
            st.bar_chart(importance_df.set_index("feature").head(10))
        else:
            st.caption("SHAP explainability data not yet available -- run the training pipeline to generate it.")

st.divider()
st.subheader("Exploratory Data Analysis")

raw_df = load_raw_data()

eda_tab1, eda_tab2, eda_tab3 = st.tabs(["Pollutant Correlations", "Time Patterns", "City Comparison"])

with eda_tab1:
    st.caption("How strongly each pollutant relates to overall AQI and to each other")
    corr_cols = ["aqi", "pm2_5", "pm10", "co", "no2", "o3", "so2"]
    available_cols = [c for c in corr_cols if c in raw_df.columns]
    corr_matrix = raw_df[available_cols].corr()

    heatmap_fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale=[[0, "#ECEFF1"], [0.5, "#81C784"], [1, "#1B5E20"]],
        text=corr_matrix.round(2).values,
        texttemplate="%{text}",
        textfont=dict(size=11),
    ))
    heatmap_fig.update_layout(height=420, paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Poppins", color="#37474F"))
    st.plotly_chart(heatmap_fig, use_container_width=True)

with eda_tab2:
    col_a, col_b = st.columns(2)

    with col_a:
        st.caption("Average AQI by hour of day")
        hourly_pattern = raw_df.groupby("hour")["aqi"].mean().reset_index()
        hour_fig = go.Figure(data=[go.Scatter(x=hourly_pattern["hour"], y=hourly_pattern["aqi"], mode="lines+markers", line=dict(color="#2E7D32", width=3))])
        hour_fig.update_layout(xaxis_title="Hour (UTC)", yaxis_title="Avg AQI", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Poppins", color="#37474F"))
        st.plotly_chart(hour_fig, use_container_width=True)

    with col_b:
        st.caption("Average AQI by day of week")
        weekday_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        weekday_pattern = raw_df.groupby("weekday")["aqi"].mean().reindex(range(7)).reset_index()
        weekday_pattern["weekday_name"] = weekday_names
        weekday_fig = go.Figure(data=[go.Bar(x=weekday_pattern["weekday_name"], y=weekday_pattern["aqi"], marker_color="#66BB6A")])
        weekday_fig.update_layout(yaxis_title="Avg AQI", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Poppins", color="#37474F"))
        st.plotly_chart(weekday_fig, use_container_width=True)

with eda_tab3:
    st.caption("AQI distribution across all cities (box plot shows spread, median, and outliers)")
    box_fig = go.Figure()
    for c in CITIES.keys():
        city_values = raw_df[raw_df["city"] == c]["aqi"]
        box_fig.add_trace(go.Box(y=city_values, name=c, marker_color="#2E7D32"))
    box_fig.update_layout(yaxis_title="AQI", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Poppins", color="#37474F"), showlegend=False)
    st.plotly_chart(box_fig, use_container_width=True)

st.divider()
st.caption("Pearls AQI Predictor — internship project. Data source: OpenWeather Air Pollution API.")