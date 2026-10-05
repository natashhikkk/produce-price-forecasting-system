import pandas as pd

from config.settings import (
    WEATHER_RAW_FILE,
    PERIOD_ORIGIN_DATE,
    WEATHER_PROCESSED_FILE,
)


def prepare_weather():
    df = pd.read_csv(WEATHER_RAW_FILE)

    df["time"] = pd.to_datetime(df["time"], errors="coerce")

    weather_columns = [
        "temperature_2m_mean",
        "temperature_2m_min",
        "temperature_2m_max",
        "precipitation_sum",
    ]

    df[weather_columns] = df[weather_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    df = (
        df.dropna(subset=["time", "location"] + weather_columns)
        .drop_duplicates(subset=["time", "location"])
    )

    regional = (
        df.groupby("time", as_index=False)
        .agg(
            temperature_mean=("temperature_2m_mean", "mean"),
            temperature_min=("temperature_2m_min", "mean"),
            temperature_max=("temperature_2m_max", "mean"),
            precipitation=("precipitation_sum", "mean"),
        )
    )

    origin = pd.Timestamp(PERIOD_ORIGIN_DATE)

    regional["period_start"] = (
        origin
        + pd.to_timedelta(
            ((regional["time"] - origin).dt.days // 14) * 14,
            unit="D",
        )
    )

    weather_14d = (
        regional.groupby("period_start", as_index=False)
        .agg(
            temperature_mean=("temperature_mean", "mean"),
            temperature_min=("temperature_min", "mean"),
            temperature_max=("temperature_max", "mean"),
            precipitation_sum=("precipitation", "sum"),
            days_count=("time", "count"),
        )
    )

    weather_14d = (
        weather_14d[weather_14d["days_count"] == 14]
        .drop(columns="days_count")
        .sort_values("period_start")
        .reset_index(drop=True)
    )

    return weather_14d


if __name__ == "__main__":
    weather = prepare_weather()

    WEATHER_PROCESSED_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    weather.to_csv(
        WEATHER_PROCESSED_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print("Размер:", weather.shape)
    print(
        "Период:",
        weather["period_start"].min(),
        "—",
        weather["period_start"].max(),
    )
    print("Пропуски:", weather.isna().sum().sum())
    print(
        "Дубликаты:",
        weather["period_start"].duplicated().sum(),
    )
    print("Данные сохранены:", WEATHER_PROCESSED_FILE)