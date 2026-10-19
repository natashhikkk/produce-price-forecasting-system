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

from src.analysis.analysis import run_analysis

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


def run_source(name, loader, file_path, key_columns):
    try:
        df = loader()

        new_rows = update_csv(
            df,
            file_path,
            key_columns,
        )

        log_load(
            name,
            "success",
            len(df),
            new_rows,
            "",
        )

        print(
            f"{name}: {len(df)} строк, "
            f"новых {new_rows}"
        )

        return df

    except Exception as error:
        log_load(
            name,
            "error",
            0,
            0,
            str(error),
        )

        print(
            f"{name}: ошибка загрузки — {error}"
        )

        if file_path.exists():
            df = pd.read_csv(
                file_path
            )

            print(
                f"{name}: использованы "
                f"локальные данные, "
                f"{len(df)} строк"
            )

            return df

        return None


def save_analysis_dataset(df):
    ANALYSIS_DATASET_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        ANALYSIS_DATASET_FILE,
        index=False,
        encoding="utf-8-sig",
    )


def main():
    print("Запуск системы")

    print("\nЗагрузка данных")

    prices_raw = run_source(
        "DEFRA prices",
        load_defra_prices,
        DEFRA_RAW_FILE,
        [
            "category",
            "item",
            "variety",
            "date",
            "unit",
        ],
    )

    weather_raw = run_source(
        "Open-Meteo weather",
        load_weather,
        WEATHER_RAW_FILE,
        [
            "time",
            "location",
        ],
    )

    diesel_raw = run_source(
        "Diesel prices",
        load_diesel_prices,
        DIESEL_RAW_FILE,
        ["Date"],
    )

    fertiliser_raw = run_source(
        "Agricultural Price Index",
        load_agricultural_price_index,
        FERTILISER_RAW_FILE,
        [
            "type",
            "category",
            "date",
        ],
    )

    if any(
        df is None
        for df in [
            prices_raw,
            weather_raw,
            diesel_raw,
            fertiliser_raw,
        ]
    ):
        print(
            "\nНе удалось получить "
            "все необходимые данные"
        )
        return

    print("\nПредобработка данных")

    prices = prepare_prices(
        prices_raw
    )

    weather = prepare_weather(
        weather_raw
    )

    diesel = prepare_diesel(
        diesel_raw
    )

    fertiliser = prepare_fertiliser(
        fertiliser_raw
    )

    analysis_dataset = merge_processed_data(
        prices,
        weather,
        diesel,
        fertiliser,
    )

    save_analysis_dataset(
        analysis_dataset
    )

    print("Предобработка завершена")

    print(
        "Товарных позиций:",
        analysis_dataset[
            ["item", "variety", "unit"]
        ].drop_duplicates().shape[0],
    )

    print(
        "Двухнедельных периодов:",
        analysis_dataset[
            "period_start"
        ].nunique(),
    )

    print(
        "Размер набора:",
        analysis_dataset.shape,
    )

    print(
        "Пропущенных цен:",
        analysis_dataset[
            "price"
        ].isna().sum(),
    )

    print(
        "Период:",
        analysis_dataset[
            "period_start"
        ].min(),
        "—",
        analysis_dataset[
            "period_start"
        ].max(),
    )

    print(
        "Данные сохранены:",
        ANALYSIS_DATASET_FILE,
    )

    print("\nАнализ данных")

    run_analysis(
        analysis_dataset,
        verbose=False,
    )

    print("Анализ завершён")

    print("\nПрогнозирование")

    model_data, model_start = (
        prepare_data(
            analysis_dataset
        )
    )

    metrics = evaluate_models(
        model_data
    )

    best_models = get_best_models(
        metrics
    )

    forecasts = create_forecasts(
        model_data,
        metrics,
    )

    MODEL_METRICS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics.to_csv(
        MODEL_METRICS_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    forecasts.to_csv(
        FORECASTS_FILE,
        index=False,
        encoding="utf-8-sig",
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
        print(
            f"{model}: {count}"
        )

    successful = (
        forecasts["status"] == "ok"
    ).sum()

    failed = (
        forecasts["status"] != "ok"
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
        failed,
    )

    print(
        "Прогнозный период:",
        forecasts[
            "forecast_period_start"
        ].min().date(),
        "—",
        forecasts[
            "forecast_period_end"
        ].max().date(),
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

    general_paths, product_paths = (
        create_all_visualizations(
            analysis_dataset,
            metrics,
            forecasts,
        )
    )

    print("Визуализация завершена")

    print(
        "Общих графиков:",
        len(general_paths),
    )

    print(
        "Графиков по товарам:",
        len(product_paths),
    )

    print(
        "Всего графиков:",
        len(general_paths)
        + len(product_paths),
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