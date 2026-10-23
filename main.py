import pandas as pd

from config.settings import (
    DEFRA_RAW_FILE,
    WEATHER_RAW_FILE,
    DIESEL_RAW_FILE,
    FERTILISER_RAW_FILE,
    ANALYSIS_DATASET_FILE,
    MODEL_METRICS_FILE,
    FORECASTS_FILE,
    ANALYTICAL_CONCLUSIONS_FILE,
)
from src.data_loading.defra_prices import load_defra_prices
from src.data_loading.weather_api import load_weather
from src.data_loading.diesel_prices import load_diesel_prices
from src.data_loading.fertiliser_index import load_agricultural_price_index
from src.preprocessing.preprocessing import (
    prepare_prices,
    prepare_weather,
    prepare_diesel,
    prepare_fertiliser,
    merge_processed_data,
)
from src.analysis.analysis import run_analysis
from src.analysis.analytical_conclusions import create_analytical_conclusions
from src.forecasting.forecasting import (
    TEST_START_DATE,
    prepare_data,
    evaluate_models,
    get_best_models,
    create_forecasts,
)
from src.visualization.visualization import create_all_visualizations
from src.utils.storage import update_csv
from src.utils.load_logger import log_load


SOURCES = {
    "prices": (
        "DEFRA",
        load_defra_prices,
        DEFRA_RAW_FILE,
        ["category", "item", "variety", "date", "unit"],
    ),
    "weather": (
        "Open-Meteo",
        load_weather,
        WEATHER_RAW_FILE,
        ["time", "location"],
    ),
    "diesel": (
        "Diesel",
        load_diesel_prices,
        DIESEL_RAW_FILE,
        ["Date"],
    ),
    "fertiliser": (
        "Agricultural Price Index",
        load_agricultural_price_index,
        FERTILISER_RAW_FILE,
        ["type", "category", "date"],
    ),
}

FACTOR_NAMES = {
    "temperature_mean_lag_2": "температура, lag 2",
    "precipitation_sum_lag_2": "осадки, lag 2",
    "diesel_price_lag_2": "дизель, lag 2",
    "fertiliser_index_lag_1": "удобрения, lag 1",
}


def run_source(name, loader, file_path, keys):
    try:
        received = loader()
        data, new_rows = update_csv(received, file_path, keys)

        log_load(
            name,
            "success",
            rows_received=len(received),
            new_rows=new_rows,
        )

        print(
            f"{name}: {len(data)} строк "
            f"(получено {len(received)}, новых {new_rows})"
        )
        return data

    except Exception as error:
        if file_path.exists():
            try:
                data = pd.read_csv(file_path, encoding="utf-8-sig")
                log_load(name, "fallback", message=str(error))

                print(
                    f"{name}: используются локальные данные "
                    f"({len(data)} строк)"
                )
                return data

            except Exception as local_error:
                error = (
                    f"{error}; локальный файл не прочитан: "
                    f"{local_error}"
                )

        log_load(name, "error", message=str(error))
        print(f"{name}: данные недоступны — {error}")
        return None


def load_sources():
    return {
        key: run_source(*source)
        for key, source in SOURCES.items()
    }


def save_csv(df, file_path):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_path, index=False, encoding="utf-8-sig")


def print_analysis_summary(results):
    missing = results["missing"]["missing_pct"]
    worst_item = missing.idxmax()

    anomalies = int(
        results["anomalies"]["potential_anomalies"].sum()
    )

    trends = results["trend"]["annual_trend_pct"].dropna()
    correlations = results["correlations"].stack().dropna()

    print("\nАнализ данных")
    print(
        "Общих пропущенных периодов:",
        len(results["common_missing"]),
    )
    print(
        f"Наибольшая доля пропусков: "
        f"{worst_item} — {missing[worst_item]:.2f}%"
    )
    print("Потенциальных аномалий по IQR:", anomalies)

    if not trends.empty:
        print(
            f"Диапазон годового тренда: "
            f"{trends.min():.2f}% — {trends.max():.2f}%"
        )

    if not correlations.empty:
        item, factor = correlations.abs().idxmax()
        value = correlations.loc[(item, factor)]

        print(
            f"Наиболее сильная корреляция: "
            f"{item} / {FACTOR_NAMES.get(factor, factor)} "
            f"— {value:.3f}"
        )


def print_forecast_summary(metrics, forecasts, model_start):
    best = get_best_models(metrics)

    print("Начало моделирования:", model_start.date())
    print("Начало тестового периода:", TEST_START_DATE.date())

    print("\nЛучшие модели:")
    for model, count in best["model"].value_counts().items():
        print(f"{model}: {count}")

    ridge = (
        metrics[
            (metrics["status"] == "ok")
            & metrics["model"].isin(["Ridge", "Ridge + external"])
        ]
        .pivot(index="item", columns="model", values="MAE")
        .dropna()
    )

    improved = int(
        (ridge["Ridge + external"] < ridge["Ridge"]).sum()
    )

    successful = forecasts[forecasts["status"] == "ok"]

    print(
        f"\nВнешние факторы улучшили Ridge "
        f"для {improved} из {len(ridge)} товаров"
    )
    print(
        f"Успешных прогнозов: "
        f"{len(successful)} из {len(forecasts)}"
    )

    if not successful.empty:
        start = successful["forecast_period_start"].min()
        end = successful["forecast_period_end"].max()
        print("Прогнозный период:", start.date(), "—", end.date())


def main():
    print("Запуск системы")

    print("\nЗагрузка данных")
    raw = load_sources()

    if any(df is None for df in raw.values()):
        print("\nРабота остановлена: отсутствуют необходимые данные")
        return

    print("\nПредобработка данных")

    dataset = merge_processed_data(
        prepare_prices(raw["prices"]),
        prepare_weather(raw["weather"]),
        prepare_diesel(raw["diesel"]),
        prepare_fertiliser(raw["fertiliser"]),
    )

    save_csv(dataset, ANALYSIS_DATASET_FILE)

    print("Товаров:", dataset["item"].nunique())
    print("Периодов:", dataset["period_start"].nunique())
    print("Строк:", len(dataset))
    print("Пропущенных цен:", dataset["price"].isna().sum())
    print(
        "Период:",
        dataset["period_start"].min().date(),
        "—",
        dataset["period_start"].max().date(),
    )

    analysis = run_analysis(dataset, verbose=False)
    print_analysis_summary(analysis)

    print("\nПрогнозирование")

    model_data, model_start = prepare_data(dataset)
    metrics = evaluate_models(model_data)
    forecasts = create_forecasts(model_data, metrics)

    save_csv(metrics, MODEL_METRICS_FILE)
    save_csv(forecasts, FORECASTS_FILE)

    print_forecast_summary(metrics, forecasts, model_start)

    print("\nАналитические выводы")

    conclusions = create_analytical_conclusions(
        dataset,
        forecasts,
    )
    save_csv(conclusions, ANALYTICAL_CONCLUSIONS_FILE)

    print("Сформировано выводов:", len(conclusions))

    print("\nВизуализация")

    general, products = create_all_visualizations(
        dataset,
        metrics,
        forecasts,
    )

    print("Сформировано графиков:", len(general) + len(products))
    print("\nРабота системы завершена успешно")


if __name__ == "__main__":
    main()