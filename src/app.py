import os
import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go

from cities import CITIES
from aqi_utils import aqi_category

DATA_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "aqi_history.csv")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

LOOKBACK_DAYS = 7
FORECAST_HORIZON_DAYS = 7

CATEGORY_COLORS = {
    "green": "#00E676",
    "yellow": "#FFD54F",
    "orange": "#FFB74D",
    "red": "#FF5252",
    "purple": "#CE93D8",
    "maroon": "#B71C1C",
}

st.set_page_config(page_title="Pearls AQI Predictor", page_icon=":cloud:", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }

.stApp {
    background: radial-gradient(circle at 12% 8%, rgba(0,230,118,0.10), transparent 40%),
                radial-gradient(circle at 88% 78%, rgba(0,150,136,0.10), transparent 45%),
                #0B0F19;
}

/* glass card look, used for every content block */
.glass {
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 20px 22px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.35);
    backdrop-filter: blur(6px);
    margin-bottom: 18px;
}

.pearls-badge {
    display: inline-block;
    background: rgba(0,230,118,0.12);
    color: #00E676;
    border: 1px solid rgba(0,230,118,0.3);
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 10px;
}

.pearls-title { color: #F0F3F5; font-size: 34px; font-weight: 700; margin: 0; }
.pearls-subtitle { color: #8B96A5; font-size: 14px; margin-top: 4px; }

.metric-label { color: #8B96A5; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; }
.metric-value { color: #F0F3F5; font-size: 22px; font-weight: 700; margin-top: 2px; }
.metric-sub { color: #8B96A5; font-size: 11px; margin-top: 2px; }

.category-pill {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    margin-top: 6px;
}

div[data-testid="stMetric"] { background: transparent; }
h2, h3 { color: #F0F3F5 !important; }
p, span, label { color: #C7CFD6; }
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
    """Load the full hourly CSV, unaggregated -- used for pollutant breakdown, trends, EDA."""

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


def category_pill_html(category, color_key):
    color = CATEGORY_COLORS.get(color_key, "#8B96A5")
    return f'<span class="category-pill" style="background:{color}22; color:{color}; border:1px solid {color}55;">{category}</span>'


#  Header
last_refresh = datetime.utcnow().strftime("%b %d, %Y  %H:%M UTC")
header_col1, header_col2 = st.columns([3, 1])
with header_col1:
    st.markdown("""
    <div class="pearls-badge">Live AQI Intelligence</div>
    <div class="pearls-title">Pearls AQI Predictor</div>
    <div class="pearls-subtitle">7-day air quality forecasts for major Pakistani cities, with live conditions and model explainability</div>
    """, unsafe_allow_html=True)
with header_col2:
    st.markdown(f"""
    <div class="glass" style="text-align:right; padding:14px 18px;">
        <div class="metric-label">Last Refresh</div>
        <div class="metric-value" style="font-size:15px;">{last_refresh}</div>
    </div>
    """, unsafe_allow_html=True)

daily_df = load_daily_data()
raw_df = load_raw_data()
models, scaler, feature_cols, summary = load_models()

city = st.selectbox("City", list(CITIES.keys()), label_visibility="collapsed")

city_daily = daily_df[daily_df["city"] == city].sort_values("date")
city_raw = raw_df[raw_df["city"] == city].sort_values("timestamp")

if city_daily.empty:
    st.warning("No data available yet for this city. Check back once the pipeline has run a few times.")
    st.stop()

latest_aqi = city_daily["aqi"].iloc[-1]
category, color_key = aqi_category(int(latest_aqi))
gauge_color = CATEGORY_COLORS.get(color_key, "#8B96A5")

prev_aqi = city_daily["aqi"].iloc[-2] if len(city_daily) > 1 else latest_aqi
change_24h = latest_aqi - prev_aqi

# Current conditions: gauge + side cards
gauge_col, side_col = st.columns([2, 1])

with gauge_col:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Current Air Quality</div>', unsafe_allow_html=True)

    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=latest_aqi,
        domain=dict(x=[0, 1], y=[0, 1]),
        number=dict(font=dict(size=42, color="#F0F3F5", family="Poppins"), valueformat=".0f"),
        gauge=dict(
            axis=dict(range=[0, 300], tickcolor="#8B96A5", tickfont=dict(color="#8B96A5", size=10)),
            bar=dict(color=gauge_color, thickness=0.28),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            steps=[
                {"range": [0, 50], "color": "rgba(0,230,118,0.15)"},
                {"range": [50, 100], "color": "rgba(255,213,79,0.15)"},
                {"range": [100, 150], "color": "rgba(255,183,77,0.15)"},
                {"range": [150, 200], "color": "rgba(255,82,82,0.15)"},
                {"range": [200, 300], "color": "rgba(183,28,28,0.2)"},
            ],
        ),
    ))
    gauge_fig.update_layout(height=300, margin=dict(l=30, r=30, t=55, b=10), paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Poppins"))
    st.plotly_chart(gauge_fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with side_col:
    st.markdown(f"""
    <div class="glass">
        <div class="metric-label">Current Status</div>
        {category_pill_html(category, color_key)}
        <div class="metric-sub" style="margin-top:10px;">{'▲' if change_24h >= 0 else '▼'} {abs(change_24h):.0f} vs 24h ago</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="glass">
        <div class="metric-label">Daily AQI</div>
        <div class="metric-value">{latest_aqi:.0f}</div>
        <div class="metric-sub">Live snapshot</div>
    </div>
    """, unsafe_allow_html=True)

# Pollutant grid
if not city_raw.empty:
    latest_row = city_raw.iloc[-1]
    pollutants = [
        ("PM2.5", latest_row.get("pm2_5"), "µg/m³"),
        ("PM10", latest_row.get("pm10"), "µg/m³"),
        ("O3", latest_row.get("o3"), "µg/m³"),
        ("NO2", latest_row.get("no2"), "µg/m³"),
        ("SO2", latest_row.get("so2"), "µg/m³"),
        ("CO", latest_row.get("co"), "µg/m³"),
    ]
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label" style="margin-bottom:10px;">Pollutant Breakdown</div>', unsafe_allow_html=True)
    pollutant_cols = st.columns(6)
    for col, (name, value, unit) in zip(pollutant_cols, pollutants):
        with col:
            val_display = f"{value:.1f}" if value is not None and not pd.isna(value) else "-"
            st.markdown(f"""
            <div style="text-align:center;">
                <div class="metric-label">{name}</div>
                <div class="metric-value" style="font-size:18px;">{val_display}</div>
                <div class="metric-sub">{unit}</div>
            </div>
            """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# 24-hour trend 
st.markdown('<div class="glass">', unsafe_allow_html=True)
st.markdown('<div class="metric-label" style="margin-bottom:10px;">24-Hour AQI Trend</div>', unsafe_allow_html=True)
last_24h = city_raw.tail(24)
if not last_24h.empty:
    trend_fig = go.Figure(data=[go.Scatter(
        x=last_24h["timestamp"], y=last_24h["aqi"],
        mode="lines", line=dict(color="#00E676", width=3),
        fill="tozeroy", fillcolor="rgba(0,230,118,0.08)",
    )])
    trend_fig.update_layout(
        height=260, margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8B96A5", family="Poppins"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
    )
    st.plotly_chart(trend_fig, use_container_width=True)
else:
    st.caption("Not enough hourly data yet.")
st.markdown('</div>', unsafe_allow_html=True)

# Forecast 
input_row = build_input_row(city, daily_df, feature_cols)

if input_row is not None:
    scaled_input = scaler.transform(input_row)

    forecast_values = []
    for horizon in range(1, FORECAST_HORIZON_DAYS + 1):
        if horizon in models:
            forecast_values.append(models[horizon].predict(scaled_input)[0])
        else:
            forecast_values.append(None)

    forecast_dates_short = [(datetime.utcnow() + timedelta(days=h)).strftime("%a %d") for h in range(1, FORECAST_HORIZON_DAYS + 1)]

    st.markdown('<div class="metric-label" style="margin: 6px 0 10px 2px;">Forecast</div>', unsafe_allow_html=True)
    forecast_cols = st.columns(min(4, FORECAST_HORIZON_DAYS))
    for i, col in enumerate(forecast_cols):
        value = forecast_values[i]
        if value is not None:
            cat, ckey = aqi_category(int(value))
            with col:
                st.markdown(f"""
                <div class="glass" style="text-align:center;">
                    <div class="metric-label">+{i+1} Day</div>
                    {category_pill_html(cat, ckey)}
                    <div class="metric-value" style="margin-top:8px;">{value:.1f}</div>
                    <div class="metric-sub">{forecast_dates_short[i]}</div>
                </div>
                """, unsafe_allow_html=True)

    bar_colors = []
    for value in forecast_values:
        if value is not None:
            _, ckey = aqi_category(int(value))
            bar_colors.append(CATEGORY_COLORS.get(ckey, "#8B96A5"))
        else:
            bar_colors.append("#8B96A5")

    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label" style="margin-bottom:10px;">Predicted AQI Trend (7 Days)</div>', unsafe_allow_html=True)
    forecast_fig = go.Figure(data=[go.Scatter(
        x=forecast_dates_short, y=forecast_values,
        mode="lines+markers",
        line=dict(color="#00E676", width=3),
        marker=dict(size=9, color=bar_colors, line=dict(width=2, color="#0B0F19")),
    )])
    forecast_fig.update_layout(
        height=280, margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8B96A5", family="Poppins"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(title="AQI", gridcolor="rgba(255,255,255,0.05)"),
    )
    st.plotly_chart(forecast_fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("Not enough historical data yet to generate a 7-day forecast for this city.")

# Explainability
st.markdown('<div class="glass">', unsafe_allow_html=True)
st.markdown('<div class="metric-label" style="margin-bottom:6px;">Explainability</div>', unsafe_allow_html=True)
shap_path = os.path.join(MODELS_DIR, "shap_importance.json")
if os.path.exists(shap_path):
    with open(shap_path) as f:
        shap_data = json.load(f)
    st.caption(f"Based on the day-1 forecast model ({shap_data['model_used']})")
    importance_df = pd.DataFrame(shap_data["importance"], columns=["feature", "mean_abs_shap_value"]).head(8)

    imp_fig = go.Figure(data=[go.Bar(
        x=importance_df["mean_abs_shap_value"], y=importance_df["feature"],
        orientation="h",
        marker=dict(
            color=importance_df["mean_abs_shap_value"],
            colorscale=[[0, "#26C6DA"], [0.5, "#B388FF"], [1, "#FF6E6E"]],
            line=dict(width=0),
        ),
    )])
    imp_fig.update_layout(
        height=300, margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8B96A5", family="Poppins"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(imp_fig, use_container_width=True)
else:
    st.caption("SHAP explainability data not yet available -- run the training pipeline to generate it.")
st.markdown('</div>', unsafe_allow_html=True)

# Model performance
with st.expander("Model performance details"):
    if summary:
        st.dataframe(pd.DataFrame(summary), use_container_width=True)
    st.caption("Models are retrained daily as more historical data accumulates via GitHub Actions.")

# National map
st.markdown('<div class="glass">', unsafe_allow_html=True)
st.markdown('<div class="metric-label" style="margin-bottom:10px;">National Overview</div>', unsafe_allow_html=True)

latest_per_city = daily_df.sort_values("date").groupby("city").last().reset_index()
if not latest_per_city.empty:
    latest_per_city["lat"] = latest_per_city["city"].map(lambda c: CITIES[c]["lat"])
    latest_per_city["lon"] = latest_per_city["city"].map(lambda c: CITIES[c]["lon"])
    latest_per_city["category"] = latest_per_city["aqi"].apply(lambda v: aqi_category(int(v))[0])
    latest_per_city["marker_color"] = latest_per_city["aqi"].apply(lambda v: CATEGORY_COLORS.get(aqi_category(int(v))[1], "#8B96A5"))

    map_fig = go.Figure()
    map_fig.add_trace(go.Scattergeo(
        lat=latest_per_city["lat"], lon=latest_per_city["lon"],
        mode="markers",
        marker=dict(size=latest_per_city["aqi"] / 5 + 6, color=latest_per_city["marker_color"], opacity=0.25, line=dict(width=0)),
        hoverinfo="skip", showlegend=False,
    ))
    map_fig.add_trace(go.Scattergeo(
        lat=latest_per_city["lat"], lon=latest_per_city["lon"],
        mode="markers+text",
        text=latest_per_city["city"],
        textposition="top center",
        textfont=dict(size=11, color="#C7CFD6", family="Poppins"),
        marker=dict(size=latest_per_city["aqi"] / 10 + 6, color=latest_per_city["marker_color"], line=dict(width=2, color="#0B0F19"), opacity=0.95),
        customdata=latest_per_city[["aqi", "category"]],
        hovertemplate="<b>%{text}</b><br>AQI: %{customdata[0]:.0f}<br>%{customdata[1]}<extra></extra>",
        showlegend=False,
    ))
    map_fig.update_geos(
        center=dict(lat=30.3753, lon=69.3451), projection_scale=9,
        showcountries=True, countrycolor="rgba(255,255,255,0.15)",
        showland=True, landcolor="rgba(255,255,255,0.03)",
        showocean=True, oceancolor="rgba(255,255,255,0.01)",
        showlakes=False, bgcolor="rgba(0,0,0,0)",
    )
    map_fig.update_layout(height=440, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(map_fig, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# EDA

st.markdown('<div class="glass">', unsafe_allow_html=True)
st.markdown('<div class="metric-label" style="margin-bottom:10px;">Exploratory Data Analysis</div>', unsafe_allow_html=True)

eda_tab1, eda_tab2, eda_tab3 = st.tabs(["Pollutant Correlations", "Time Patterns", "City Comparison"])

with eda_tab1:
    corr_cols = ["aqi", "pm2_5", "pm10", "co", "no2", "o3", "so2"]
    available_cols = [c for c in corr_cols if c in raw_df.columns]
    corr_matrix = raw_df[available_cols].corr()
    heatmap_fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.columns,
        colorscale=[
            [0.0, "#3949AB"],   # low/negative correlation - indigo
            [0.35, "#26C6DA"],  # teal
            [0.55, "#8B96A5"],  # neutral grey, near zero
            [0.75, "#FFB74D"],  # amber
            [1.0, "#FF6E6E"],   # strong positive - coral
        ],
        text=corr_matrix.round(2).values, texttemplate="%{text}", textfont=dict(size=11, color="#0B0F19"),
        colorbar=dict(tickfont=dict(color="#8B96A5")),
    ))
    heatmap_fig.update_layout(height=420, paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Poppins", color="#8B96A5"))
    st.plotly_chart(heatmap_fig, use_container_width=True)

with eda_tab2:
    col_a, col_b = st.columns(2)
    with col_a:
        st.caption("Average AQI by hour of day")
        hourly_pattern = raw_df.groupby("hour")["aqi"].mean().reset_index()
        hour_fig = go.Figure(data=[go.Scatter(
            x=hourly_pattern["hour"], y=hourly_pattern["aqi"],
            mode="lines+markers", line=dict(color="#26C6DA", width=3),
            fill="tozeroy", fillcolor="rgba(38,198,218,0.12)",
            marker=dict(size=6, color="#26C6DA"),
        )])
        hour_fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Poppins", color="#8B96A5"), xaxis=dict(gridcolor="rgba(255,255,255,0.05)"), yaxis=dict(gridcolor="rgba(255,255,255,0.05)"))
        st.plotly_chart(hour_fig, use_container_width=True)
    with col_b:
        st.caption("Average AQI by day of week")
        weekday_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        weekday_pattern = raw_df.groupby("weekday")["aqi"].mean().reindex(range(7)).reset_index()
        weekday_pattern["weekday_name"] = weekday_names
        weekday_fig = go.Figure(data=[go.Bar(
            x=weekday_pattern["weekday_name"], y=weekday_pattern["aqi"],
            marker=dict(color=weekday_pattern["aqi"], colorscale=[[0, "#FFD54F"], [1, "#FF6E6E"]]),
        )])
        weekday_fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Poppins", color="#8B96A5"), xaxis=dict(gridcolor="rgba(255,255,255,0.05)"), yaxis=dict(gridcolor="rgba(255,255,255,0.05)"))
        st.plotly_chart(weekday_fig, use_container_width=True)

with eda_tab3:
    city_palette = ["#00E676", "#26C6DA", "#FFB74D", "#F06292", "#B388FF", "#FF6E6E", "#AEEA00", "#64B5F6"]
    box_fig = go.Figure()
    for i, c in enumerate(CITIES.keys()):
        city_values = raw_df[raw_df["city"] == c]["aqi"]
        box_fig.add_trace(go.Box(y=city_values, name=c, marker_color=city_palette[i % len(city_palette)]))
    box_fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Poppins", color="#8B96A5"), yaxis=dict(gridcolor="rgba(255,255,255,0.05)"), showlegend=False)
    st.plotly_chart(box_fig, use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

st.caption("Pearls AQI Predictor — internship project. Data source: OpenWeather Air Pollution API.")