from io import BytesIO
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

from config.settings import (
    AGRICULTURAL_PRICE_INDEX_PAGE_URL,
    AGRICULTURAL_PRICE_INDEX_REQUIRED_COLUMNS,
    REQUEST_TIMEOUT,
)


def find_index_csv_url(session):
    """Находит актуальный CSV Agricultural Price Index."""

    response = session.get(
        AGRICULTURAL_PRICE_INDEX_PAGE_URL,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()

    soup = BeautifulSoup(
        response.text,
        "html.parser",
    )

    for link in soup.find_all("a", href=True):
        text = " ".join(link.stripped_strings).lower()

        if (
            "agricultural price index" in text
            and "2020 =100" in text
            and "csv format" in text
        ):
            return urljoin(
                AGRICULTURAL_PRICE_INDEX_PAGE_URL,
                link["href"],
            )

    raise RuntimeError(
        "Не найден актуальный CSV Agricultural Price Index."
    )


def load_agricultural_price_index():
    """Получает данные Agricultural Price Index."""

    session = requests.Session()

    # Не используем системные настройки прокси
    session.trust_env = False

    csv_url = find_index_csv_url(session)

    response = session.get(
        csv_url,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()

    df = pd.read_csv(
        BytesIO(response.content),
    )

    missing = (
        set(AGRICULTURAL_PRICE_INDEX_REQUIRED_COLUMNS)
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"Отсутствуют обязательные столбцы: {missing}"
        )

    if df.empty:
        raise ValueError(
            "Получен пустой файл Agricultural Price Index."
        )

    return df


if __name__ == "__main__":
    df = load_agricultural_price_index()

    print(df.head())
    print()
    print("Размер:", df.shape)
    print("Столбцы:", df.columns.tolist())