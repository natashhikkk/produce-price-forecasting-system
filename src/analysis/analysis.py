import pandas as pd

from config.settings import ANALYSIS_DATASET_FILE


EXTERNAL_FACTORS = [
    "temperature_mean",
    "temperature_min",
    "temperature_max",
    "precipitation_sum",
    "diesel_price",
    "fertiliser_index",
]


def analyze_missing_values(df):
    result = (
        df.groupby("item")["price"]
        .agg(total="size", available="count")
    )

    result["missing"] = (
        result["total"] - result["available"]
    )

    result["missing_pct"] = (
        result["missing"]
        / result["total"]
        * 100
    ).round(2)

    common_missing = (
        df.groupby("period_start")["price"]
        .apply(lambda x: x.isna().all())
    )

    common_missing = common_missing[
        common_missing
    ].index

    return result, common_missing


def descriptive_statistics(df):
    return (
        df.groupby("item")["price"]
        .agg(
            count="count",
            mean="mean",
            median="median",
            std="std",
            min="min",
            max="max",
        )
        .round(3)
    )


def analyze_trend(df):
    rows = []

    for item, data in df.groupby("item"):
        prices = (
            data["price"]
            .dropna()
            .reset_index(drop=True)
        )

        x = pd.Series(
            range(len(prices)),
            dtype=float,
        )

        slope = (
            x.cov(prices) / x.var()
            if len(prices) > 1
            else 0
        )

        annual_change = (
            slope * 26 / prices.mean() * 100
            if prices.mean() != 0
            else 0
        )

        rows.append({
            "item": item,
            "trend_per_period": slope,
            "annual_trend_pct": annual_change,
        })

    return (
        pd.DataFrame(rows)
        .set_index("item")
        .round(2)
    )


def analyze_seasonality(df):
    data = df.dropna(
        subset=["price"]
    ).copy()

    data["month"] = (
        data["period_start"].dt.month
    )

    monthly = (
        data.groupby(
            ["item", "month"],
            as_index=False,
        )["price"]
        .mean()
    )

    max_month = monthly.loc[
        monthly.groupby("item")["price"].idxmax()
    ][["item", "month"]]

    min_month = monthly.loc[
        monthly.groupby("item")["price"].idxmin()
    ][["item", "month"]]

    return (
        max_month
        .rename(columns={"month": "highest_price_month"})
        .merge(
            min_month.rename(
                columns={"month": "lowest_price_month"}
            ),
            on="item",
        )
        .set_index("item")
    )


def detect_anomalies(df):
    rows = []

    for item, data in df.groupby("item"):
        prices = data["price"].dropna()

        q1 = prices.quantile(0.25)
        q3 = prices.quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        anomalies = (
            (prices < lower)
            | (prices > upper)
        ).sum()

        rows.append({
            "item": item,
            "potential_anomalies": anomalies,
        })

    return pd.DataFrame(rows).set_index("item")


def analyze_external_factors(df):
    rows = []

    for item, data in df.groupby("item"):
        data = (
            data
            .sort_values("period_start")
            .copy()
        )

        row = {"item": item}

        for factor in EXTERNAL_FACTORS:
            row[factor] = (
                data["price"]
                .corr(data[factor].shift(1))
            )

        rows.append(row)

    return (
        pd.DataFrame(rows)
        .set_index("item")
        .round(3)
    )


def run_analysis(df):
    df = df.copy()

    df["period_start"] = pd.to_datetime(
        df["period_start"]
    )

    print("АНАЛИЗ ДАННЫХ")

    print("\nОбщая информация")
    print("Размер:", df.shape)
    print(
        "Период:",
        df["period_start"].min(),
        "—",
        df["period_start"].max(),
    )
    print(
        "Товарных позиций:",
        df["item"].nunique(),
    )

    missing, common_missing = (
        analyze_missing_values(df)
    )

    print("\nПропуски цены")
    print(missing)

    print(
        "\nПериодов без цен по всем товарам:",
        len(common_missing),
    )

    if len(common_missing) > 0:
        print(
            pd.Series(common_missing)
            .dt.strftime("%Y-%m-%d")
            .to_string(index=False)
        )

    print("\nОписательная статистика")
    print(descriptive_statistics(df))

    print("\nОценка тренда")
    print(analyze_trend(df))

    print("\nСезонность")
    print(analyze_seasonality(df))

    print("\nПотенциальные аномалии")
    print(detect_anomalies(df))

    print(
        "\nКорреляция цены с внешними "
        "факторами предыдущего периода"
    )
    print(
        analyze_external_factors(df).to_string()
    )


if __name__ == "__main__":
    data = pd.read_csv(
        ANALYSIS_DATASET_FILE,
        parse_dates=["period_start"],
    )

    run_analysis(data)