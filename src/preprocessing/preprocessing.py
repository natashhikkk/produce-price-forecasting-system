import pandas as pd

from config.settings import (
    SELECTED_PRODUCTS,
    PERIOD_ORIGIN_DATE,
    WEATHER_FIELDS,
    WEATHER_LOCATIONS,
    FUEL_REQUIRED_COLUMNS,
    FERTILISER_PUBLICATION_LAG_MONTHS,
)


ORIGIN = pd.Timestamp(PERIOD_ORIGIN_DATE)
PRODUCT_KEY = ["item", "variety", "unit"]


def get_period_start(dates):
    days = (dates - ORIGIN).dt.days
    return ORIGIN + pd.to_timedelta((days // 14) * 14, unit="D")


def prepare_prices(df):
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    for column in ["category", "item", "variety", "unit"]:
        df[column] = df[column].astype("string").str.strip().str.lower()

    selected = pd.DataFrame(
        SELECTED_PRODUCTS,
        columns=PRODUCT_KEY,
    )

    df = df.merge(selected, on=PRODUCT_KEY, how="inner")
    df = df.dropna(subset=["date", "price", "category"])
    df = df[df["price"] > 0].drop_duplicates()

    available = set(
        df[PRODUCT_KEY]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )

    missing = set(SELECTED_PRODUCTS) - available

    if missing:
        raise ValueError(
            f"Нет данных для товарных позиций: {sorted(missing)}"
        )

    df["period_start"] = get_period_start(df["date"])

    return (
        df.groupby(
            ["category", *PRODUCT_KEY, "period_start"],
            as_index=False,
        )
        .agg(
            price=("price", "mean"),
            source_observations=("price", "size"),
        )
    )


def prepare_weather(df):
    df = df.copy()

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df[WEATHER_FIELDS] = df[WEATHER_FIELDS].apply(
        pd.to_numeric,
        errors="coerce",
    )

    df = (
        df.dropna(subset=["time", "location", *WEATHER_FIELDS])
        .drop_duplicates(["time", "location"], keep="last")
    )

    daily = (
        df.groupby("time", as_index=False)
        .agg(
            temperature_mean=("temperature_2m_mean", "mean"),
            temperature_min=("temperature_2m_min", "mean"),
            temperature_max=("temperature_2m_max", "mean"),
            precipitation=("precipitation_sum", "mean"),
            locations=("location", "nunique"),
        )
    )

    daily = daily[
        daily["locations"] == len(WEATHER_LOCATIONS)
    ].copy()

    daily["period_start"] = get_period_start(daily["time"])

    weather = (
        daily.groupby("period_start", as_index=False)
        .agg(
            temperature_mean=("temperature_mean", "mean"),
            temperature_min=("temperature_min", "mean"),
            temperature_max=("temperature_max", "mean"),
            precipitation_sum=("precipitation", "sum"),
            days=("time", "count"),
        )
    )

    return (
        weather[weather["days"] == 14]
        .drop(columns="days")
        .reset_index(drop=True)
    )


def prepare_diesel(df):
    df = df.copy()
    date_column, diesel_column = FUEL_REQUIRED_COLUMNS

    df[date_column] = pd.to_datetime(
        df[date_column],
        dayfirst=True,
        errors="coerce",
    )

    df[diesel_column] = pd.to_numeric(
        df[diesel_column],
        errors="coerce",
    )

    df = (
        df.dropna(subset=[date_column, diesel_column])
        .drop_duplicates(date_column, keep="last")
    )

    df = df[df[date_column] >= ORIGIN].copy()
    df["period_start"] = get_period_start(df[date_column])

    return (
        df.groupby("period_start", as_index=False)
        .agg(diesel_price=(diesel_column, "mean"))
    )


def prepare_fertiliser(df):
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["index"] = pd.to_numeric(df["index"], errors="coerce")

    for column in ["type", "category"]:
        df[column] = df[column].astype("string").str.strip().str.lower()

    df = df[
        (df["type"] == "input")
        & (df["category"] == "fertilisers_and_soil_improvers")
    ]

    df = (
        df.dropna(subset=["date", "index"])
        .drop_duplicates("date", keep="last")
        .sort_values("date")
    )

    df["available_date"] = (
        df["date"]
        + pd.DateOffset(months=FERTILISER_PUBLICATION_LAG_MONTHS)
    )

    periods = pd.DataFrame({
        "period_start": pd.date_range(
            ORIGIN,
            pd.Timestamp.today().normalize(),
            freq="14D",
        )
    })

    result = pd.merge_asof(
        periods,
        df[["available_date", "index"]],
        left_on="period_start",
        right_on="available_date",
        direction="backward",
    )

    return (
        result.rename(columns={"index": "fertiliser_index"})
        [["period_start", "fertiliser_index"]]
        .dropna()
        .reset_index(drop=True)
    )


def check_external_gaps(df, columns, name):
    complete = df[columns].notna().all(axis=1)

    if not complete.any():
        raise ValueError(f"Нет данных внешнего фактора: {name}")

    last_available = df.loc[
        complete,
        "period_start",
    ].max()

    internal = df.loc[
        df["period_start"] <= last_available,
        columns,
    ]

    if internal.isna().any().any():
        raise ValueError(
            f"Во внешнем факторе '{name}' есть внутренние пропуски."
        )


def merge_processed_data(prices, weather, diesel, fertiliser):
    periods = pd.DataFrame({
        "period_start": pd.date_range(
            prices["period_start"].min(),
            prices["period_start"].max(),
            freq="14D",
        )
    })

    products = (
        prices[["category", *PRODUCT_KEY]]
        .drop_duplicates(PRODUCT_KEY)
    )

    if len(products) != len(SELECTED_PRODUCTS):
        raise ValueError(
            "Количество товарных позиций не соответствует настройкам."
        )

    external = (
        periods
        .merge(weather, on="period_start", how="left")
        .merge(diesel, on="period_start", how="left")
        .merge(fertiliser, on="period_start", how="left")
    )

    check_external_gaps(
        external,
        [
            "temperature_mean",
            "temperature_min",
            "temperature_max",
            "precipitation_sum",
        ],
        "погода",
    )

    check_external_gaps(
        external,
        ["diesel_price"],
        "дизель",
    )

    check_external_gaps(
        external,
        ["fertiliser_index"],
        "удобрения",
    )

    result = (
        products
        .merge(periods, how="cross")
        .merge(
            prices,
            on=["category", *PRODUCT_KEY, "period_start"],
            how="left",
        )
        .merge(
            external,
            on="period_start",
            how="left",
        )
    )

    return result.sort_values(
        ["item", "period_start"]
    ).reset_index(drop=True)