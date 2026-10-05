from src.data_loading.defra_prices import load_defra_prices
from src.data_loading.weather_api import load_weather
from src.data_loading.diesel_prices import load_diesel_prices
from src.data_loading.fertiliser_index import (
    load_agricultural_price_index,
)

from src.utils.storage import update_csv
from src.utils.load_logger import log_load

from config.settings import (
    DEFRA_RAW_FILE,
    WEATHER_RAW_FILE,
    DIESEL_RAW_FILE,
    FERTILISER_RAW_FILE,
)


def run_source(source_name, loader, file_path, key_columns):
    """
    Загружает данные одного источника,
    сохраняет их и записывает результат в журнал.
    """

    print(f"\nЗагрузка: {source_name}")

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

        print("Статус: успешно")
        print("Получено строк:", len(df))
        print("Новых строк:", new_rows)

    except Exception as error:

        log_load(
            source=source_name,
            status="error",
            message=str(error),
        )

        print("Статус: ошибка")
        print(error)


def main():

    print("Начало загрузки данных")

    run_source(
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

    run_source(
        source_name="Open-Meteo weather",
        loader=load_weather,
        file_path=WEATHER_RAW_FILE,
        key_columns=[
            "time",
            "location",
        ],
    )

    run_source(
        source_name="Diesel prices",
        loader=load_diesel_prices,
        file_path=DIESEL_RAW_FILE,
        key_columns=[
            "Date",
        ],
    )

    run_source(
        source_name="Agricultural Price Index",
        loader=load_agricultural_price_index,
        file_path=FERTILISER_RAW_FILE,
        key_columns=[
            "type",
            "category",
            "date",
        ],
    )

    print("\nЗагрузка данных завершена")


if __name__ == "__main__":
    main()
