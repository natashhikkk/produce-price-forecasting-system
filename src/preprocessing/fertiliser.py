import pandas as pd

from config.settings import (
    FERTILISER_RAW_FILE,
    PERIOD_ORIGIN_DATE,
    FERTILISER_PROCESSED_FILE,
    FERTILISER_PUBLICATION_LAG_MONTHS,
)


def prepare_fertiliser():
    df = pd.read_csv(FERTILISER_RAW_FILE)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["index"] = pd.to_numeric(df["index"], errors="coerce")

    df["type"] = df["type"].astype("string").str.strip().str.lower()
    df["category"] = df["category"].astype("string").str.strip().str.lower()

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

    origin = pd.Timestamp(PERIOD_ORIGIN_DATE)
    last_date = pd.Timestamp.today().normalize()

    periods = pd.DataFrame({
        "period_start": pd.date_range(
            start=origin,
            end=last_date,
            freq="14D",
        )
    })

    fertiliser_14d = pd.merge_asof(
        periods.sort_values("period_start"),
        df[["available_date", "index"]].sort_values("available_date"),
        left_on="period_start",
        right_on="available_date",
        direction="backward",
    )

    fertiliser_14d = (
        fertiliser_14d
        .rename(columns={"index": "fertiliser_index"})
        [["period_start", "fertiliser_index"]]
        .dropna()
        .reset_index(drop=True)
    )

    return fertiliser_14d


if __name__ == "__main__":
    fertiliser = prepare_fertiliser()

    FERTILISER_PROCESSED_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fertiliser.to_csv(
        FERTILISER_PROCESSED_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print("Размер:", fertiliser.shape)
    print(
        "Период:",
        fertiliser["period_start"].min(),
        "—",
        fertiliser["period_start"].max(),
    )
    print("Пропуски:", fertiliser.isna().sum().sum())
    print(
        "Дубликаты:",
        fertiliser["period_start"].duplicated().sum(),
    )
    print("Данные сохранены:", FERTILISER_PROCESSED_FILE)