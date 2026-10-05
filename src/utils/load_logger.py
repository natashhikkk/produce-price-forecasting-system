from datetime import datetime
from pathlib import Path

import pandas as pd

from config.settings import LOAD_LOG_FILE


def log_load(
    source,
    status,
    rows_received=0,
    new_rows=0,
    message="",
):
    """
    Записывает результат загрузки источника в журнал.
    """

    log_file = Path(LOAD_LOG_FILE)

    # Создаём папку logs, если её нет
    log_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    record = pd.DataFrame(
        [{
            "datetime": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "source": source,
            "status": status,
            "rows_received": rows_received,
            "new_rows": new_rows,
            "message": message,
        }]
    )

    # Если журнал уже есть — добавляем строку
    if log_file.exists():

        record.to_csv(
            log_file,
            mode="a",
            header=False,
            index=False,
            encoding="utf-8-sig"
        )

    # Первый запуск — создаём файл с заголовками
    else:

        record.to_csv(
            log_file,
            index=False,
            encoding="utf-8-sig"
        )

if __name__ == "__main__":

    log_load(
        source="test_source",
        status="success",
        rows_received=100,
        new_rows=10,
        message="Тестовая запись"
    )

    print("Запись добавлена в журнал.")