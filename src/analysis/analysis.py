import numpy as np
import pandas as pd

from config.settings import ANALYSIS_DATASET_FILE


EXTERNAL_FACTORS = {
    "temperature_mean": 2,
    "precipitation_sum": 2,
    "diesel_price": 2,
    "fertiliser_index": 1,
}


def analyze_missing_values(df):
    result = (
        df.groupby("item")["price"]
        .agg(total="size", available="count")
    )

    result["missing"] = result["total"] - result["available"]
    result["missing_pct"] = (
        result["missing"] / result["total"] * 100
    ).round(2)

    missing_periods = (
        df.groupby("period_start")["price"]
        .apply(lambda x: x.isna().all())
    )

    common_missing = pd.DataFrame({
        "period_start": missing_periods[
            missing_periods
        ].index
    })

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
        data = (
            data.dropna(subset=["price"])
            .sort_values("period_start")
        )

        if len(data) < 2:
            slope = np.nan
            annual_change = np.nan
        else:
            periods = (
                (data["period_start"] - data["period_start"].iloc[0])
                .dt.days
                / 14
            )

            slope = np.polyfit(
                periods,
                data["price"],
                1,
            )[0]

            annual_change = (
                slope * 26 / data["price"].mean() * 100
            )

        rows.append({
            "item": item,
            "trend_per_period": slope,
            "annual_trend_pct": annual_change,
        })

    return (
        pd.DataFrame(rows)
        .set_index("item")
        .round({
            "trend_per_period": 4,
            "annual_trend_pct": 2,
        })
    )


def analyze_seasonality(df):
    data = df.dropna(subset=["price"]).copy()
    data["month"] = data["period_start"].dt.month

    monthly = (
        data.groupby(["item", "month"])["price"]
        .mean()
        .reset_index()
    )

    highest = monthly.loc[
        monthly.groupby("item")["price"].idxmax(),
        ["item", "month"],
    ]

    lowest = monthly.loc[
        monthly.groupby("item")["price"].idxmin(),
        ["item", "month"],
    ]

    return (
        highest
        .rename(columns={"month": "highest_price_month"})
        .merge(
            lowest.rename(
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

        anomalies = (
            (prices < q1 - 1.5 * iqr)
            | (prices > q3 + 1.5 * iqr)
        ).sum()

        rows.append({
            "item": item,
            "potential_anomalies": anomalies,
        })

    return pd.DataFrame(rows).set_index("item")


def analyze_external_factors(df):
    rows = []

    for item, data in df.groupby("item"):
        data = data.sort_values("period_start")

        row = {"item": item}

        for factor, lag in EXTERNAL_FACTORS.items():
            row[f"{factor}_lag_{lag}"] = (
                data["price"].corr(
                    data[factor].shift(lag)
                )
            )

        rows.append(row)

    return (
        pd.DataFrame(rows)
        .set_index("item")
        .round(3)
    )


def run_analysis(df, verbose=True):
    df = df.copy()
    df["period_start"] = pd.to_datetime(df["period_start"])

    missing, common_missing = analyze_missing_values(df)
    statistics = descriptive_statistics(df)
    trend = analyze_trend(df)
    seasonality = analyze_seasonality(df)
    anomalies = detect_anomalies(df)
    correlations = analyze_external_factors(df)

    results = {
        "missing": missing,
        "common_missing": common_missing,
        "statistics": statistics,
        "trend": trend,
        "seasonality": seasonality,
        "anomalies": anomalies,
        "correlations": correlations,
    }

    if verbose:
        print("АНАЛИЗ ДАННЫХ")

        print("\nПропуски:")
        print(missing.to_string())

        print("\nОбщие пропущенные периоды:")
        print(common_missing.to_string(index=False))

        print("\nОписательная статистика:")
        print(statistics.to_string())

        print("\nТренд:")
        print(trend.to_string())

        print("\nСезонность:")
        print(seasonality.to_string())

        print("\nПотенциальные аномалии:")
        print(anomalies.to_string())

        print("\nКорреляции с внешними факторами:")
        print(correlations.to_string())

    return results


if __name__ == "__main__":
    data = pd.read_csv(
        ANALYSIS_DATASET_FILE,
        parse_dates=["period_start"],
    )

    run_analysis(data)