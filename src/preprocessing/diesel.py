import pandas as pd

from config.settings import (
    DIESEL_RAW_FILE,
    PERIOD_ORIGIN_DATE,
    DIESEL_PROCESSED_FILE,
    FUEL_REQUIRED_COLUMNS,
)


def prepare_diesel():
    df = pd.read_csv(DIESEL_RAW_FILE)

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

    diesel_14d = (
        df.groupby("period_start", as_index=False)
        .agg(
            diesel_price=(diesel_column, "mean"),
        )
        .sort_values("period_start")
        .reset_index(drop=True)
    )

    return diesel_14d


if __name__ == "__main__":
    diesel = prepare_diesel()

    DIESEL_PROCESSED_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    diesel.to_csv(
        DIESEL_PROCESSED_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print("Размер:", diesel.shape)
    print(
        "Период:",
        diesel["period_start"].min(),
        "—",
        diesel["period_start"].max(),
    )
    print("Пропуски:", diesel.isna().sum().sum())
    print(
        "Дубликаты:",
        diesel["period_start"].duplicated().sum(),
    )
    print("Данные сохранены:", DIESEL_PROCESSED_FILE)