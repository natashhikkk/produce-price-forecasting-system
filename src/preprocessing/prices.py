import pandas as pd

from config.settings import (
    DEFRA_RAW_FILE,
    SELECTED_PRODUCTS,
    PERIOD_ORIGIN_DATE,
    PRICES_PROCESSED_FILE,
)


def prepare_prices():
    df = pd.read_csv(DEFRA_RAW_FILE)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    text_columns = ["category", "item", "variety", "unit"]

    for column in text_columns:
        df[column] = df[column].astype("string").str.strip().str.lower()

    selected = pd.DataFrame(
        SELECTED_PRODUCTS,
        columns=["item", "variety", "unit"],
    )

    df = df.merge(
        selected,
        on=["item", "variety", "unit"],
        how="inner",
    )

    df = df.dropna(subset=["date", "price"])
    df = df[df["price"] > 0]
    df = df.drop_duplicates()

    key_columns = ["item", "variety", "unit", "date"]

    if df.duplicated(subset=key_columns).any():
        raise ValueError(
            "Обнаружено несколько наблюдений "
            "для одного товара и одной даты."
        )

    origin = pd.Timestamp(PERIOD_ORIGIN_DATE)

    df["period_start"] = (
        origin
        + pd.to_timedelta(
            ((df["date"] - origin).dt.days // 14) * 14,
            unit="D",
        )
    )

    prices_14d = (
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
        .sort_values(["item", "period_start"])
        .reset_index(drop=True)
    )

    return prices_14d


if __name__ == "__main__":
    prices = prepare_prices()

    PRICES_PROCESSED_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    prices.to_csv(
        PRICES_PROCESSED_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print("Размер:", prices.shape)
    print(
        "Период:",
        prices["period_start"].min(),
        "—",
        prices["period_start"].max(),
    )
    print("Данные сохранены:", PRICES_PROCESSED_FILE)