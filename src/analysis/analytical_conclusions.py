import pandas as pd

from src.analysis.analysis import analyze_external_factors


CORRELATION_COLUMNS = {
    "temperature_mean_lag_2": "температурой",
    "precipitation_sum_lag_2": "осадками",
    "diesel_price_lag_2": "ценой дизельного топлива",
    "fertiliser_index_lag_1": "индексом удобрений",
}


def describe_change(change):
    if change <= -5:
        return "значительное снижение"
    if change < -1:
        return "умеренное снижение"
    if change <= 1:
        return "сохранение относительно стабильного уровня"
    if change < 5:
        return "умеренный рост"
    return "значительный рост"


def describe_correlation(value):
    value = abs(value)

    if value < 0.3:
        return "слабая"
    if value < 0.5:
        return "умеренная"
    if value < 0.7:
        return "сильная"
    return "очень сильная"


def get_anomaly_summary(prices):
    prices = prices.dropna()

    if prices.empty:
        return {
            "anomaly_count": 0,
            "anomaly_pct": 0,
            "high_anomaly_count": 0,
            "low_anomaly_count": 0,
            "anomaly_lower_bound": None,
            "anomaly_upper_bound": None,
        }

    q1 = prices.quantile(0.25)
    q3 = prices.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    low_count = int((prices < lower).sum())
    high_count = int((prices > upper).sum())
    count = low_count + high_count

    return {
        "anomaly_count": count,
        "anomaly_pct": count / len(prices) * 100,
        "high_anomaly_count": high_count,
        "low_anomaly_count": low_count,
        "anomaly_lower_bound": lower,
        "anomaly_upper_bound": upper,
    }


def build_anomaly_text(anomalies):
    count = anomalies["anomaly_count"]
    high = anomalies["high_anomaly_count"]
    low = anomalies["low_anomaly_count"]

    if count == 0:
        return (
            "Аномальные значения цены по критерию IQR "
            "в историческом ряду не выявлены."
        )

    text = (
        f"По критерию IQR в историческом ряду выявлено "
        f"{count} аномальных периодов "
        f"({anomalies['anomaly_pct']:.1f}% наблюдений)"
    )

    if high and low:
        return (
            text
            + f": {high} с необычно высокой ценой "
            + f"и {low} с необычно низкой ценой."
        )

    if high:
        return text + ", все с необычно высокой ценой."

    return text + ", все с необычно низкой ценой."


def build_conclusion(
    item,
    unit,
    forecast,
    latest_price,
    change,
    change_description,
    factor,
    correlation,
    correlation_strength,
    anomaly_text,
):
    start = forecast["forecast_period_start"].strftime("%d.%m.%Y")
    end = forecast["forecast_period_end"].strftime("%d.%m.%Y")

    text = (
        f"На прогнозном горизонте {start}–{end} ожидается "
        f"{change_description} оптовой цены на {item}: "
        f"изменение составляет {change:+.1f}% — "
        f"с {latest_price:.3f} до {forecast['forecast']:.3f} £/{unit}. "
        f"Прогнозный интервал составляет "
        f"{forecast['lower_bound']:.3f}–"
        f"{forecast['upper_bound']:.3f} £/{unit}. "
    )

    if factor is not None:
        direction = (
            "положительной"
            if correlation > 0
            else "отрицательной"
        )

        text += (
            f"Среди учтённых внешних факторов наиболее выраженная "
            f"связь цены наблюдается с {factor} "
            f"(r = {correlation:.2f}); связь является "
            f"{direction}, а её сила оценивается как "
            f"{correlation_strength}. "
        )
    else:
        text += (
            "Недостаточно данных для оценки связи цены "
            "с внешними факторами. "
        )

    return text + anomaly_text


def create_analytical_conclusions(data, forecasts):
    correlations = analyze_external_factors(data)
    results = []

    for item in sorted(data["item"].unique()):
        product = (
            data[data["item"] == item]
            .sort_values("period_start")
        )

        forecast_rows = forecasts[
            forecasts["item"] == item
        ]

        if forecast_rows.empty:
            continue

        forecast = forecast_rows.iloc[0]
        prices = product["price"].dropna()

        if prices.empty or pd.isna(forecast["forecast"]):
            continue

        latest_price = prices.iloc[-1]
        latest = product.dropna(subset=["price"]).iloc[-1]

        variety = latest["variety"]
        unit = latest["unit"]

        change = (
            (forecast["forecast"] - latest_price)
            / latest_price
            * 100
        )

        change_description = describe_change(change)

        factor = None
        correlation = None
        correlation_strength = None

        if item in correlations.index:
            row = correlations.loc[item]

            available = row[
                [
                    column
                    for column in CORRELATION_COLUMNS
                    if column in row.index
                ]
            ].dropna()

            if not available.empty:
                column = available.abs().idxmax()

                correlation = available[column]
                factor = CORRELATION_COLUMNS[column]
                correlation_strength = describe_correlation(
                    correlation
                )

        anomalies = get_anomaly_summary(
            product["price"]
        )

        anomaly_text = build_anomaly_text(
            anomalies
        )

        conclusion = build_conclusion(
            item,
            unit,
            forecast,
            latest_price,
            change,
            change_description,
            factor,
            correlation,
            correlation_strength,
            anomaly_text,
        )

        results.append({
            "item": item,
            "variety": variety,
            "unit": unit,
            "forecast_period_start":
                forecast["forecast_period_start"],
            "forecast_period_end":
                forecast["forecast_period_end"],
            "latest_price": latest_price,
            "forecast": forecast["forecast"],
            "lower_bound": forecast["lower_bound"],
            "upper_bound": forecast["upper_bound"],
            "change_pct": change,
            "change_description": change_description,
            "strongest_factor": factor,
            "correlation": correlation,
            "correlation_strength": correlation_strength,
            **anomalies,
            "conclusion": conclusion,
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    from config.settings import (
        ANALYSIS_DATASET_FILE,
        FORECASTS_FILE,
    )

    data = pd.read_csv(
        ANALYSIS_DATASET_FILE,
        parse_dates=["period_start"],
    )

    forecasts = pd.read_csv(
        FORECASTS_FILE,
        parse_dates=[
            "forecast_period_start",
            "forecast_period_end",
        ],
    )

    conclusions = create_analytical_conclusions(
        data,
        forecasts,
    )

    print(
        conclusions[
            [
                "item",
                "change_pct",
                "strongest_factor",
                "correlation",
                "anomaly_count",
                "anomaly_pct",
            ]
        ].to_string(index=False)
    )

    for _, row in conclusions.iterrows():
        print(f"\n{row['item']}:")
        print(row["conclusion"])

    print("\nВсего выводов:", len(conclusions))