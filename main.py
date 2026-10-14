import pandas as pd

from config.settings import (
    DEFRA_RAW_FILE,
    WEATHER_RAW_FILE,
    DIESEL_RAW_FILE,
    FERTILISER_RAW_FILE,
    ANALYSIS_DATASET_FILE,
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

from src.utils.storage import update_csv
from src.utils.load_logger import log_load


def run_source(source_name, loader, file_path, key_columns):
    try:
        df = loader()

        new_rows = update_csv(
            df=df,
            file_path=file_path,
            key_columns=key_columns,
        )

        log_load(
            source=source_name,
            status="success",
            rows_received=len(df),
            new_rows=new_rows,
            message="OK",
        )

        print(
            f"{source_name}: "
            f"{len(df)} строк, новых {new_rows}"
        )

        return df

    except Exception as error:
        log_load(
            source=source_name,
            status="error",
            message=str(error),
        )

        print(f"{source_name}: ошибка — {error}")

        if file_path.exists():
            print("Используется последняя локальная версия.")
            return pd.read_csv(file_path)

        return None


def save_processed(df):
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

    raw_prices = run_source(
        source_name="DEFRA prices",
        loader=load_defra_prices,
        file_path=DEFRA_RAW_FILE,
        key_columns=[
            "category",
            "item",
            "variety",
            "unit",
            "date",
        ],
    )

    raw_weather = run_source(
        source_name="Open-Meteo weather",
        loader=load_weather,
        file_path=WEATHER_RAW_FILE,
        key_columns=[
            "time",
            "location",
        ],
    )

    raw_diesel = run_source(
        source_name="Diesel prices",
        loader=load_diesel_prices,
        file_path=DIESEL_RAW_FILE,
        key_columns=[
            "Date",
        ],
    )

    raw_fertiliser = run_source(
        source_name="Agricultural Price Index",
        loader=load_agricultural_price_index,
        file_path=FERTILISER_RAW_FILE,
        key_columns=[
            "type",
            "category",
            "date",
        ],
    )

    raw_data = [
        raw_prices,
        raw_weather,
        raw_diesel,
        raw_fertiliser,
    ]

    if any(df is None for df in raw_data):
        print(
            "\nПредобработка не выполнена: "
            "не удалось получить все необходимые данные."
        )
        return

    print("\nПредобработка данных")

    prices = prepare_prices(raw_prices)
    weather = prepare_weather(raw_weather)
    diesel = prepare_diesel(raw_diesel)
    fertiliser = prepare_fertiliser(raw_fertiliser)

    dataset = merge_processed_data(
        prices,
        weather,
        diesel,
        fertiliser,
    )

    save_processed(dataset)

    product_count = (
        dataset[
            ["item", "variety", "unit"]
        ]
        .drop_duplicates()
        .shape[0]
    )

    print("Предобработка завершена")
    print("Товарных позиций:", product_count)
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
        "\nДанные сохранены:",
        ANALYSIS_DATASET_FILE,
    )

    print("\nРабота системы завершена успешно")


if __name__ == "__main__":
    main()