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

from src.forecasting.forecasting import (
    prepare_data,
    evaluate_models,
    get_best_models,
    create_forecasts,
)

from src.analysis.analytical_conclusions import (
    create_analytical_conclusions,
)

from src.visualization.visualization import (
    create_all_visualizations,
)

from src.utils.storage import update_csv
from src.utils.load_logger import log_load


SOURCES = {
    "prices": {
        "name": "DEFRA",
        "loader": load_defra_prices,
        "file": DEFRA_RAW_FILE,
        "keys": [
            "category",
            "item",
            "variety",
            "date",
            "unit",
        ],
    },
    "weather": {
        "name": "Open-Meteo",
        "loader": load_weather,
        "file": WEATHER_RAW_FILE,
        "keys": ["time", "location"],
    },
    "diesel": {
        "name": "Diesel",
        "loader": load_diesel_prices,
        "file": DIESEL_RAW_FILE,
        "keys": ["Date"],
    },
    "fertiliser": {
        "name": "Agricultural Price Index",
        "loader": load_agricultural_price_index,
        "file": FERTILISER_RAW_FILE,
        "keys": ["type", "category", "date"],
    },
}


def run_source(source):
    name = source["name"]
    file_path = source["file"]

    try:
        received = source["loader"]()

        data, new_rows = update_csv(
            received,
            file_path,
            source["keys"],
        )

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
            data = pd.read_csv(
                file_path,
                encoding="utf-8-sig",
            )

            log_load(
                name,
                "fallback",
                message=str(error),
            )

            print(
                f"{name}: ошибка загрузки, "
                f"используются локальные данные "
                f"({len(data)} строк)"
            )

            return data

        log_load(
            name,
            "error",
            message=str(error),
        )

        print(f"{name}: ошибка — {error}")
        return None


def load_sources():
    return {
        key: run_source(source)
        for key, source in SOURCES.items()
    }


def save_csv(df, file_path):
    file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        file_path,
        index=False,
        encoding="utf-8-sig",
    )


def main():
    print("Загрузка данных")

    sources = load_sources()

    if any(df is None for df in sources.values()):
        print(
            "\nРабота остановлена: "
            "отсутствуют необходимые данные"
        )
        return

    print("\nПредобработка данных")

    prices = prepare_prices(
        sources["prices"]
    )

    weather = prepare_weather(
        sources["weather"]
    )

    diesel = prepare_diesel(
        sources["diesel"]
    )

    fertiliser = prepare_fertiliser(
        sources["fertiliser"]
    )

    dataset = merge_processed_data(
        prices,
        weather,
        diesel,
        fertiliser,
    )

    save_csv(
        dataset,
        ANALYSIS_DATASET_FILE,
    )

    print(
        "Товаров:",
        dataset["item"].nunique(),
    )

    print(
        "Периодов:",
        dataset["period_start"].nunique(),
    )

    print(
        "Строк:",
        len(dataset),
    )

    print(
        "Пропущенных цен:",
        dataset["price"].isna().sum(),
    )

    print(
        "Период:",
        dataset["period_start"].min().date(),
        "—",
        dataset["period_start"].max().date(),
    )

    print("\nПрогнозирование")

    prepared = prepare_data(dataset)

    if isinstance(prepared, tuple):
        model_data = prepared[0]

        if len(prepared) > 1:
            model_start = prepared[1]
            print(
                "Начало моделирования:",
                model_start,
            )
    else:
        model_data = prepared

    metrics = evaluate_models(
        model_data
    )

    forecasts = create_forecasts(
        model_data,
        metrics,
    )

    save_csv(
        metrics,
        MODEL_METRICS_FILE,
    )

    save_csv(
        forecasts,
        FORECASTS_FILE,
    )

    best_models = get_best_models(
        metrics
    )

    print("Выбранные модели:")

    for model, count in (
        best_models["model"]
        .value_counts()
        .items()
    ):
        print(
            f"{model}: {count}"
        )

    success = (
        forecasts["status"] == "ok"
    ).sum()

    print(
        "Прогнозов:",
        len(forecasts),
    )

    print(
        "Успешно:",
        success,
    )

    print(
        "Ошибок:",
        len(forecasts) - success,
    )

    successful_forecasts = forecasts[
        forecasts["status"] == "ok"
    ]

    if not successful_forecasts.empty:
        print(
            "Прогнозный период:",
            successful_forecasts[
                "forecast_period_start"
            ].min(),
            "—",
            successful_forecasts[
                "forecast_period_end"
            ].max(),
        )

    print("\nАналитические выводы")

    conclusions = create_analytical_conclusions(
        dataset,
        forecasts,
    )

    save_csv(
        conclusions,
        ANALYTICAL_CONCLUSIONS_FILE,
    )

    print(
        "Сформировано выводов:",
        len(conclusions),
    )

    print("\nВизуализация")

    general_figures, product_figures = (
        create_all_visualizations(
            dataset,
            metrics,
            forecasts,
        )
    )

    print(
        "Общих графиков:",
        len(general_figures),
    )

    print(
        "Графиков по товарам:",
        len(product_figures),
    )

    print(
        "Всего графиков:",
        len(general_figures)
        + len(product_figures),
    )

    print(
        "\nРабота системы завершена успешно"
    )


if __name__ == "__main__":
    main()