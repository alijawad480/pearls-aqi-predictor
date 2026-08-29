# OpenWeather gives its own AQI scale (1 to 5), which is NOT the standard AQI
# people are used to seeing (0 to 500, the EPA scale used by IQAir, AQICN, etc).
# So instead of using OpenWeather's own scale, we calculate the real US EPA AQI
# ourselves from the raw PM2.5 value they give us. PM2.5 is the pollutant that
# usually drives AQI in Pakistani cities, so this is the standard approach.

# Each row below is: (low_concentration, high_concentration, low_aqi, high_aqi)
PM25_BREAKPOINTS = [
    (0.0, 12.0, 0, 50),
    (12.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200),
    (150.5, 250.4, 201, 300),
    (250.5, 350.4, 301, 400),
    (350.5, 500.4, 401, 500),
]


def pm25_to_aqi(pm25_value):
    """Convert a PM2.5 concentration (µg/m³) into a standard US AQI value (0-500)."""

    # cap extreme values so the formula doesn't break
    if pm25_value > 500.4:
        pm25_value = 500.4

    for low_c, high_c, low_aqi, high_aqi in PM25_BREAKPOINTS:
        if low_c <= pm25_value <= high_c:
            # standard linear interpolation formula used by the EPA
            aqi = ((high_aqi - low_aqi) / (high_c - low_c)) * (pm25_value - low_c) + low_aqi
            return round(aqi)

    return 500  # anything above the table is treated as max


def aqi_category(aqi_value):
    """Return the human-readable AQI category and a color, used for alerts and the dashboard."""

    if aqi_value <= 50:
        return "Good", "green"
    elif aqi_value <= 100:
        return "Moderate", "yellow"
    elif aqi_value <= 150:
        return "Unhealthy for Sensitive Groups", "orange"
    elif aqi_value <= 200:
        return "Unhealthy", "red"
    elif aqi_value <= 300:
        return "Very Unhealthy", "purple"
    else:
        return "Hazardous", "maroon"
