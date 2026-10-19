import pandas as pd

from config.settings import (
    DEFRA_RAW_FILE,
    WEATHER_RAW_FILE,
    DIESEL_RAW_FILE,
    FERTILISER_RAW_FILE,
    ANALYSIS_DATASET_FILE,
    MODEL_METRICS_FILE,
    FORECASTS_FILE,
)

from src.data_loading.defra_prices import load_defra_prices
from src.data_loading.weather_api import load_weather
from src.data_loading.diesel_prices import load_diesel_prices
from src.data_loading.fertiliser_index import (
    load_agricultural_price_index,
)

from src.preprocessing.preprocessing import (
    prepare_prices,
    prepare_weather,
    prepare_diesel,
    prepare_fertiliser,
    merge_processed_data,
)

from src.forecasting.forecasting import (
    TEST_START_DATE,
    prepare_data,
    evaluate_models,
    get_best_models,
    create_forecasts,
)

from src.visualization.visualization import (
    create_all_visualizations,
    FIGURES_DIR,
)

from src.utils.storage import update_csv
from src.utils.load_logger import log_load


SOURCES = {
    "prices": (
        "DEFRA prices",
        load_defra_prices,
        DEFRA_RAW_FILE,
        ["category", "item", "variety", "date", "unit"],
    ),
    "weather": (
        "Open-Meteo weather",
        load_weather,
        WEATHER_RAW_FILE,
        ["time", "location"],
    ),
    "diesel": (
        "Diesel prices",
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


def run_source(name, loader, file_path, key_columns):
    try:
        received = loader()

        data, new_rows = update_csv(
            received,
            file_path,
            key_columns,
        )

        log_load(
            name,
            "success",
            len(received),
            new_rows,
        )

        print(
            f"{name}: получено {len(received)}, "
            f"в хранилище {len(data)}, новых {new_rows}"
        )

        return data

    except Exception as error:
        print(f"{name}: ошибка загрузки — {error}")

        if file_path.exists():
            try:
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
                    f"{name}: использованы локальные данные, "
                    f"{len(data)} строк"
                )

                return data

            except Exception as local_error:
                error = (
                    f"{error}; локальный файл "
                    f"не удалось прочитать: {local_error}"
                )

        log_load(
            name,
            "error",
            message=str(error),
        )

        print(f"{name}: данные недоступны")

        return None


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


def load_sources():
    data = {}

    for key, source in SOURCES.items():
        data[key] = run_source(*source)

    return data


def main():
    print("Запуск системы")

    print("\nЗагрузка данных")
    raw = load_sources()

    if any(data is None for data in raw.values()):
        print(
            "\nНе удалось получить все необходимые данные"
        )
        return

    print("\nПредобработка данных")

    dataset = merge_processed_data(
        prepare_prices(raw["prices"]),
        prepare_weather(raw["weather"]),
        prepare_diesel(raw["diesel"]),
        prepare_fertiliser(raw["fertiliser"]),
    )

    save_csv(
        dataset,
        ANALYSIS_DATASET_FILE,
    )

    print("Предобработка завершена")
    print(
        "Товарных позиций:",
        dataset[
            ["item", "variety", "unit"]
        ].drop_duplicates().shape[0],
    )
    print(
        "Двухнедельных периодов:",
        dataset["period_start"].nunique(),
    )
    print("Размер набора:", dataset.shape)
    print(
        "Пропущенных цен:",
        dataset["price"].isna().sum(),
    )
    print(
        "Период:",
        dataset["period_start"].min(),
        "—",
        dataset["period_start"].max(),
    )
    print(
        "Данные сохранены:",
        ANALYSIS_DATASET_FILE,
    )

    print("\nПрогнозирование")

    model_data, model_start = prepare_data(
        dataset
    )

    metrics = evaluate_models(
        model_data
    )

    forecasts = create_forecasts(
        model_data,
        metrics,
    )

    best_models = get_best_models(
        metrics
    )

    save_csv(
        metrics,
        MODEL_METRICS_FILE,
    )

    save_csv(
        forecasts,
        FORECASTS_FILE,
    )

    print("Прогнозирование завершено")
    print(
        "Начало моделирования:",
        model_start.date(),
    )
    print(
        "Начало тестового периода:",
        TEST_START_DATE.date(),
    )

    print("\nЛучшие модели:")

    for model, count in (
        best_models["model"]
        .value_counts()
        .items()
    ):
        print(f"{model}: {count}")

    successful = (
        forecasts["status"] == "ok"
    ).sum()

    print(
        "\nИтоговых прогнозов:",
        len(forecasts),
    )
    print(
        "Успешных:",
        successful,
    )
    print(
        "Ошибок:",
        len(forecasts) - successful,
    )
    print(
        "Прогнозный период:",
        forecasts["forecast_period_start"].min().date(),
        "—",
        forecasts["forecast_period_end"].max().date(),
    )

    print(
        "\nМетрики сохранены:",
        MODEL_METRICS_FILE,
    )
    print(
        "Прогнозы сохранены:",
        FORECASTS_FILE,
    )

    print("\nВизуализация")

    general, products = create_all_visualizations(
        dataset,
        metrics,
        forecasts,
    )

    print("Визуализация завершена")
    print("Общих графиков:", len(general))
    print("Графиков по товарам:", len(products))
    print(
        "Всего графиков:",
        len(general) + len(products),
    )
    print(
        "Графики сохранены:",
        FIGURES_DIR,
    )

    print(
        "\nРабота системы завершена успешно"
    )


if __name__ == "__main__":
    main()