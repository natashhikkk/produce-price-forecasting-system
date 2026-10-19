from datetime import datetime

import pandas as pd

from config.settings import LOAD_LOG_FILE


def log_load(
    source,
    status,
    rows_received=0,
    new_rows=0,
    message="",
):
    """Записывает результат загрузки источника в журнал."""

    LOAD_LOG_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    record = pd.DataFrame([{
        "datetime": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "source": source,
        "status": status,
        "rows_received": rows_received,
        "new_rows": new_rows,
        "message": message,
    }])

    record.to_csv(
        LOAD_LOG_FILE,
        mode="a",
        header=not LOAD_LOG_FILE.exists(),
        index=False,
        encoding="utf-8-sig",
    )