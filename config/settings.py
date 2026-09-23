from pathlib import Path


# Корневая папка проекта
BASE_DIR = Path(__file__).resolve().parents[1]

# Папки с данными
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
LOGS_DIR = BASE_DIR / "logs"


# DEFRA Wholesale fruit and vegetable prices

DEFRA_PRICES_PAGE_URL = (
    "https://www.gov.uk/government/statistical-data-sets/"
    "wholesale-fruit-and-vegetable-prices-weekly-average"
)

# Последняя исходная версия файла
DEFRA_RAW_FILE = RAW_DATA_DIR / "defra_prices_latest.csv"

# Накопленная история
DEFRA_HISTORY_FILE = PROCESSED_DATA_DIR / "defra_prices_history.csv"

# Таймаут сетевого запроса
REQUEST_TIMEOUT = 60


# Обязательные столбцы источника
DEFRA_REQUIRED_COLUMNS = [
    "category",
    "item",
    "variety",
    "date",
    "price",
    "unit",
]

# Ключ, однозначно определяющий наблюдение
DEFRA_KEY_COLUMNS = [
    "category",
    "item",
    "variety",
    "unit",
    "date",
]

SELECTED_PRODUCTS = [
    ("apples", "bramleys_seedling", "kg"),
    ("beetroot", "beetroot", "kg"),
    ("cabbage", "red", "kg"),
    ("carrots", "topped_washed", "kg"),
    ("cauliflower", "all", "head"),
    ("celeriac", "celeriac", "kg"),
    ("curly_kale", "curly_kale", "kg"),
    ("leeks", "trimmed", "kg"),
    ("lettuce", "butterhead_indoor", "head"),
    ("onion", "bulb_brown", "kg"),
    ("pak_choi", "pak_choi", "kg"),
    ("parsnips", "all_varieties", "kg"),
    ("spring_greens", "prepacked", "kg"),
    ("swede", "swede", "kg"),
    ("turnip", "turnip", "kg"),
]


# Open-Meteo Historical Weather API

OPEN_METEO_URL = (
    "https://archive-api.open-meteo.com/v1/archive"
)

WEATHER_START_DATE = "2017-11-03"

WEATHER_LOCATIONS = {
    "Norwich": (52.6309, 1.2974),
    "Ely": (52.3995, 0.2624),
    "Colchester": (51.8959, 0.8919),
}

WEATHER_FIELDS = [
    "temperature_2m_mean",
    "temperature_2m_min",
    "temperature_2m_max",
    "precipitation_sum",
]
