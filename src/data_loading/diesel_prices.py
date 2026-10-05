from io import BytesIO
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

from config.settings import (
    FUEL_PRICES_PAGE_URL,
    FUEL_REQUIRED_COLUMNS,
    REQUEST_TIMEOUT,
)


def find_fuel_csv_urls():
    """Находит ссылки на оба CSV с ценами топлива."""

    response = requests.get(
        FUEL_PRICES_PAGE_URL,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()

    soup = BeautifulSoup(
        response.text,
        "html.parser",
    )

    urls = {}

    for link in soup.find_all("a", href=True):
        text = " ".join(link.stripped_strings).lower()

        if (
            "weekly road fuel prices (csv)" in text
            and "2003 to 2017" in text
        ):
            urls["archive"] = urljoin(
                FUEL_PRICES_PAGE_URL,
                link["href"],
            )

        if (
            "weekly road fuel prices (csv)" in text
            and "2018 to" in text
        ):
            urls["current"] = urljoin(
                FUEL_PRICES_PAGE_URL,
                link["href"],
            )

    if "archive" not in urls or "current" not in urls:
        raise RuntimeError(
            "Не удалось найти оба CSV с ценами топлива."
        )

    return urls


def load_diesel_prices():
    """Получает архивные и актуальные данные о ценах топлива."""

    urls = find_fuel_csv_urls()

    parts = []

    for source, url in urls.items():

        response = requests.get(
            url,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()

        df = pd.read_csv(
            BytesIO(response.content),
        )

        # Удаляем возможный BOM из названия первого столбца
        df.columns = df.columns.str.replace(
            "\ufeff",
            "",
            regex=False,
        )

        missing = (
            set(FUEL_REQUIRED_COLUMNS)
            - set(df.columns)
        )

        if missing:
            raise ValueError(
                f"В данных топлива отсутствуют столбцы: {missing}"
            )

        if df.empty:
            raise ValueError(
                f"Получен пустой файл: {source}"
            )

        parts.append(df)

    return pd.concat(
        parts,
        ignore_index=True,
    )


if __name__ == "__main__":
    fuel = load_diesel_prices()

    print(fuel.head())
    print()
    print("Размер:", fuel.shape)
    print("Столбцы:", fuel.columns.tolist())