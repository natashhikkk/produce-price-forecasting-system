import pandas as pd

from config.settings import (
    SELECTED_PRODUCTS,
    PERIOD_ORIGIN_DATE,
    FUEL_REQUIRED_COLUMNS,
    FERTILISER_PUBLICATION_LAG_MONTHS,
)


def prepare_prices(df):
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    for column in ["category", "item", "variety", "unit"]:
        df[column] = (
            df[column]
            .astype("string")
            .str.strip()
            .str.lower()
        )

    selected = pd.DataFrame(
        SELECTED_PRODUCTS,
        columns=["item", "variety", "unit"],
    )

    df = df.merge(
        selected,
        on=["item", "variety", "unit"],
        how="inner",
    )

    df = (
        df.dropna(subset=["date", "price"])
        .query("price > 0")
        .drop_duplicates()
    )

    origin = pd.Timestamp(PERIOD_ORIGIN_DATE)

    df["period_start"] = (
        origin
        + pd.to_timedelta(
            ((df["date"] - origin).dt.days // 14) * 14,
            unit="D",
        )
    )

    return (
        df.groupby(
            [
                "category",
                "item",
                "variety",
                "unit",
                "period_start",
            ],
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
        df.dropna(
            subset=["time", "location"] + weather_columns
        )
        .drop_duplicates(
            subset=["time", "location"]
        )
    )

    daily = (
        df.groupby("time", as_index=False)
        .agg(
            temperature_mean=("temperature_2m_mean", "mean"),
            temperature_min=("temperature_2m_min", "mean"),
            temperature_max=("temperature_2m_max", "mean"),
            precipitation=("precipitation_sum", "mean"),
        )
    )

    origin = pd.Timestamp(PERIOD_ORIGIN_DATE)

    daily["period_start"] = (
        origin
        + pd.to_timedelta(
            ((daily["time"] - origin).dt.days // 14) * 14,
            unit="D",
        )
    )

    weather = (
        daily.groupby("period_start", as_index=False)
        .agg(
            temperature_mean=("temperature_mean", "mean"),
            temperature_min=("temperature_min", "mean"),
            temperature_max=("temperature_max", "mean"),
            precipitation_sum=("precipitation", "sum"),
            days_count=("time", "count"),
        )
    )

    return (
        weather[weather["days_count"] == 14]
        .drop(columns="days_count")
        .reset_index(drop=True)
    )


def prepare_diesel(df):
    df = df.copy()

    date_column = "Date"
    diesel_column = FUEL_REQUIRED_COLUMNS[1]

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
        .drop_duplicates(subset=[date_column])
    )

    origin = pd.Timestamp(PERIOD_ORIGIN_DATE)

    df = df[df[date_column] >= origin].copy()

    df["period_start"] = (
        origin
        + pd.to_timedelta(
            ((df[date_column] - origin).dt.days // 14) * 14,
            unit="D",
        )
    )

    return (
        df.groupby("period_start", as_index=False)
        .agg(
            diesel_price=(diesel_column, "mean"),
        )
    )


def prepare_fertiliser(df):
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["index"] = pd.to_numeric(df["index"], errors="coerce")

    df["type"] = (
        df["type"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    df["category"] = (
        df["category"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    df = df[
        (df["type"] == "input")
        & (df["category"] == "fertilisers_and_soil_improvers")
    ].copy()

    df = (
        df.dropna(subset=["date", "index"])
        .drop_duplicates(subset=["date"])
        .sort_values("date")
    )

    df["available_date"] = (
        df["date"]
        + pd.DateOffset(
            months=FERTILISER_PUBLICATION_LAG_MONTHS
        )
    )

    periods = pd.DataFrame({
        "period_start": pd.date_range(
            start=PERIOD_ORIGIN_DATE,
            end=pd.Timestamp.today().normalize(),
            freq="14D",
        )
    })

    fertiliser = pd.merge_asof(
        periods,
        df[["available_date", "index"]],
        left_on="period_start",
        right_on="available_date",
        direction="backward",
    )

    return (
        fertiliser
        .rename(columns={"index": "fertiliser_index"})
        [["period_start", "fertiliser_index"]]
        .dropna()
        .reset_index(drop=True)
    )


def merge_processed_data(
    prices,
    weather,
    diesel,
    fertiliser,
):
    sources = [
        prices,
        weather,
        diesel,
        fertiliser,
    ]

    start = max(
        df["period_start"].min()
        for df in sources
    )

    end = min(
        df["period_start"].max()
        for df in sources
    )

    periods = pd.DataFrame({
        "period_start": pd.date_range(
            start=start,
            end=end,
            freq="14D",
        )
    })

    product_columns = [
        "category",
        "item",
        "variety",
        "unit",
    ]

    products = prices[
        product_columns
    ].drop_duplicates()

    combined = products.merge(
        periods,
        how="cross",
    )

    combined = combined.merge(
        prices,
        on=product_columns + ["period_start"],
        how="left",
    )

    for source in [weather, diesel, fertiliser]:
        combined = combined.merge(
            source,
            on="period_start",
            how="left",
        )

    external_columns = [
        "temperature_mean",
        "temperature_min",
        "temperature_max",
        "precipitation_sum",
        "diesel_price",
        "fertiliser_index",
    ]

    if combined[external_columns].isna().any().any():
        raise ValueError(
            "Во внешних факторах обнаружены пропуски."
        )

    return (
        combined
        .sort_values(["item", "period_start"])
        .reset_index(drop=True)
    )