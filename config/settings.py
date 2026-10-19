from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = DATA_DIR / "results"
LOGS_DIR = BASE_DIR / "logs"


DEFRA_RAW_FILE = RAW_DATA_DIR / "defra_prices_latest.csv"
WEATHER_RAW_FILE = RAW_DATA_DIR / "weather.csv"
DIESEL_RAW_FILE = RAW_DATA_DIR / "diesel_prices.csv"
FERTILISER_RAW_FILE = RAW_DATA_DIR / "fertiliser_index.csv"

ANALYSIS_DATASET_FILE = (
    PROCESSED_DATA_DIR / "analysis_dataset.csv"
)

MODEL_METRICS_FILE = (
    RESULTS_DIR / "model_metrics.csv"
)

FORECASTS_FILE = (
    RESULTS_DIR / "forecasts.csv"
)

LOAD_LOG_FILE = (
    LOGS_DIR / "load_log.csv"
)


REQUEST_TIMEOUT = 60

PERIOD_ORIGIN_DATE = "2017-11-03"


DEFRA_PRICES_PAGE_URL = (
    "https://www.gov.uk/government/statistical-data-sets/"
    "wholesale-fruit-and-vegetable-prices-weekly-average"
)

DEFRA_REQUIRED_COLUMNS = [
    "category",
    "item",
    "variety",
    "date",
    "price",
    "unit",
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


OPEN_METEO_URL = (
    "https://archive-api.open-meteo.com/v1/archive"
)

WEATHER_START_DATE = PERIOD_ORIGIN_DATE

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


FUEL_PRICES_PAGE_URL = (
    "https://www.gov.uk/government/statistics/"
    "weekly-road-fuel-prices"
)

FUEL_REQUIRED_COLUMNS = [
    "Date",
    "ULSD (Ultra low sulphur diesel) Pump price in pence/litre",
]


AGRICULTURAL_PRICE_INDEX_PAGE_URL = (
    "https://www.gov.uk/government/statistics/"
    "agricultural-price-indices"
)

AGRICULTURAL_PRICE_INDEX_REQUIRED_COLUMNS = [
    "type",
    "category",
    "date",
    "index",
]

FERTILISER_PUBLICATION_LAG_MONTHS = 2