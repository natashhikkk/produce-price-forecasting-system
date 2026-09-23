from io import BytesIO
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

from config.settings import (
    DEFRA_PRICES_PAGE_URL,
    DEFRA_REQUIRED_COLUMNS,
    REQUEST_TIMEOUT,
)


def load_defra_prices():
    csv_url = find_csv_url()

    response = requests.get(
        csv_url,
        timeout=REQUEST_TIMEOUT
    )
    response.raise_for_status()

    df = pd.read_csv(
        BytesIO(response.content)
    )

    # Проверяем только структуру
    missing = (
        set(DEFRA_REQUIRED_COLUMNS)
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"Отсутствуют обязательные столбцы: {missing}"
        )

    if df.empty:
        raise ValueError(
            "Получен пустой файл DEFRA."
        )

    return df
