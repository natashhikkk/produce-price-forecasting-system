from datetime import date, timedelta

import pandas as pd
import requests

from config.settings import (
    OPEN_METEO_URL,
    WEATHER_START_DATE,
    WEATHER_LOCATIONS,
    WEATHER_FIELDS,
    REQUEST_TIMEOUT,
)


def load_weather(end_date=None):
    """Получает ежедневные погодные данные через Open-Meteo API."""

    if end_date is None:
        end_date = (
            date.today() - timedelta(days=5)
        ).isoformat()

    weather_parts = []

    for location, (lat, lon) in WEATHER_LOCATIONS.items():

        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": WEATHER_START_DATE,
            "end_date": end_date,
            "daily": WEATHER_FIELDS,
            "timezone": "Europe/London",
        }

        response = requests.get(
            OPEN_METEO_URL,
            params=params,
            timeout=REQUEST_TIMEOUT,
        )

        response.raise_for_status()

        data = response.json()

        if "daily" not in data:
            raise ValueError(
                f"Open-Meteo не вернул daily-данные для {location}."
            )

        df = pd.DataFrame(data["daily"])
        df["location"] = location

        weather_parts.append(df)

    return pd.concat(
        weather_parts,
        ignore_index=True,
    )