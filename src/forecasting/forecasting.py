import warnings

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX

from config.settings import (
    ANALYSIS_DATASET_FILE,
    MODEL_METRICS_FILE,
    FORECASTS_FILE,
)


SEASON_LENGTH = 26
TEST_START_DATE = pd.Timestamp("2024-12-20")
INTERVAL_Z = 1.96

PRODUCT_COLUMNS = ["item", "variety", "unit"]

BASIC_FEATURES = [
    "price_lag_1",
    "price_lag_2",
    "price_lag_26",
    "rolling_mean_2",
    "rolling_mean_4",
    "month",
]

EXTERNAL_FEATURES = [
    "temperature_lag_1",
    "precipitation_lag_1",
    "diesel_lag_1",
    "fertiliser_lag_1",
]

MODEL_NAMES = [
    "Seasonal Naive",
    "SARIMAX",
    "Ridge",
    "Ridge + external",
]


def prepare_data(df):
    df = df.copy()
    df["period_start"] = pd.to_datetime(df["period_start"])
    df = df.sort_values(PRODUCT_COLUMNS + ["period_start"])

    grouped = df.groupby(PRODUCT_COLUMNS)

    df["price_lag_1"] = grouped["price"].shift(1)
    df["price_lag_2"] = grouped["price"].shift(2)
    df["price_lag_26"] = grouped["price"].shift(26)

    df["rolling_mean_2"] = grouped["price"].transform(
        lambda x: x.shift(1).rolling(2).mean()
    )
    df["rolling_mean_4"] = grouped["price"].transform(
        lambda x: x.shift(1).rolling(4).mean()
    )

    df["temperature_lag_1"] = grouped["temperature_mean"].shift(1)
    df["precipitation_lag_1"] = grouped["precipitation_sum"].shift(1)
    df["diesel_lag_1"] = grouped["diesel_price"].shift(1)
    df["fertiliser_lag_1"] = grouped["fertiliser_index"].shift(1)

    df["month"] = df["period_start"].dt.month

    periods = sorted(df["period_start"].unique())
    model_start = pd.Timestamp(periods[SEASON_LENGTH])

    return df[df["period_start"] >= model_start].copy(), model_start


def fit_sarimax(series):
    values = series.astype(float).to_numpy()

    model = SARIMAX(
        values,
        order=(1, 1, 1),
        seasonal_order=(1, 0, 0, SEASON_LENGTH),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        result = model.fit(disp=False, maxiter=500)

    converged = bool(
        result.mle_retvals.get("converged", False)
    )

    return result, converged


def ridge_predictions(data, features):
    train = data[
        data["period_start"] < TEST_START_DATE
    ].dropna(subset=["price"] + features)

    test = data[
        data["period_start"] >= TEST_START_DATE
    ]

    predictions = np.full(len(test), np.nan)
    valid = test[features].notna().all(axis=1)

    if train.empty or not valid.any():
        return predictions

    model = make_pipeline(
        StandardScaler(),
        Ridge(alpha=1.0),
    )

    model.fit(train[features], train["price"])
    predictions[valid.to_numpy()] = model.predict(
        test.loc[valid, features]
    )

    return predictions


def evaluate_product(data):
    data = data.sort_values("period_start").reset_index(drop=True)

    train = data[data["period_start"] < TEST_START_DATE]
    test = data[data["period_start"] >= TEST_START_DATE].copy()

    comparison = pd.DataFrame({
        "price": test["price"].to_numpy(),
        "Seasonal Naive": test["price_lag_26"].to_numpy(),
    })

    sarimax, converged = fit_sarimax(train["price"])

    sarimax_predictions = []

    for actual in test["price"]:
        sarimax_predictions.append(
            float(sarimax.forecast(1)[0])
        )
        sarimax = sarimax.append([actual], refit=False)

    comparison["SARIMAX"] = sarimax_predictions
    comparison["Ridge"] = ridge_predictions(
        data,
        BASIC_FEATURES,
    )
    comparison["Ridge + external"] = ridge_predictions(
        data,
        BASIC_FEATURES + EXTERNAL_FEATURES,
    )

    common = comparison.dropna(
        subset=["price"] + MODEL_NAMES
    )

    results = []

    for model_name in MODEL_NAMES:
        errors = (
            common["price"]
            - common[model_name]
        )

        results.append({
            "model": model_name,
            "MAE": errors.abs().mean(),
            "RMSE": np.sqrt((errors ** 2).mean()),
            "evaluated_periods": len(common),
            "status": (
                "not_converged"
                if model_name == "SARIMAX" and not converged
                else "ok"
            ),
        })

    return results


def evaluate_models(data):
    results = []

    for product, product_data in data.groupby(PRODUCT_COLUMNS):
        item, variety, unit = product

        for result in evaluate_product(product_data):
            result.update({
                "item": item,
                "variety": variety,
                "unit": unit,
            })
            results.append(result)

    return pd.DataFrame(results)


def get_best_models(metrics):
    valid = metrics[
        (metrics["status"] == "ok")
        & metrics["MAE"].notna()
    ].copy()

    valid = valid.sort_values(
        ["item", "MAE", "RMSE"]
    )

    return (
        valid
        .groupby(PRODUCT_COLUMNS, as_index=False)
        .first()
        .sort_values("item")
        .reset_index(drop=True)
    )


def forecast_next(data, model_name):
    data = data.sort_values("period_start").reset_index(drop=True)

    next_period = (
        data["period_start"].max()
        + pd.Timedelta(days=14)
    )

    if model_name == "Seasonal Naive":
        forecast = data["price"].iloc[-26]

        if pd.isna(forecast):
            return np.nan, next_period, "missing_lag"

        return float(forecast), next_period, "ok"

    if model_name == "SARIMAX":
        result, converged = fit_sarimax(data["price"])

        if not converged:
            return np.nan, next_period, "not_converged"

        return (
            float(result.forecast(1)[0]),
            next_period,
            "ok",
        )

    features = BASIC_FEATURES

    if model_name == "Ridge + external":
        features = BASIC_FEATURES + EXTERNAL_FEATURES

    prices = data["price"]

    future = pd.DataFrame([{
        "price_lag_1": prices.iloc[-1],
        "price_lag_2": prices.iloc[-2],
        "price_lag_26": prices.iloc[-26],
        "rolling_mean_2": (
            prices.iloc[-2:].mean()
            if prices.iloc[-2:].notna().all()
            else np.nan
        ),
        "rolling_mean_4": (
            prices.iloc[-4:].mean()
            if prices.iloc[-4:].notna().all()
            else np.nan
        ),
        "temperature_lag_1": data["temperature_mean"].iloc[-1],
        "precipitation_lag_1": data["precipitation_sum"].iloc[-1],
        "diesel_lag_1": data["diesel_price"].iloc[-1],
        "fertiliser_lag_1": data["fertiliser_index"].iloc[-1],
        "month": next_period.month,
    }])

    if future[features].isna().any(axis=None):
        return np.nan, next_period, "missing_features"

    train = data.dropna(
        subset=["price"] + features
    )

    model = make_pipeline(
        StandardScaler(),
        Ridge(alpha=1.0),
    )

    model.fit(train[features], train["price"])

    forecast = model.predict(
        future[features]
    )[0]

    return float(forecast), next_period, "ok"


def create_forecasts(data, metrics):
    forecasts = []

    for _, best in get_best_models(metrics).iterrows():
        product_data = data[
            (data["item"] == best["item"])
            & (data["variety"] == best["variety"])
            & (data["unit"] == best["unit"])
        ]

        forecast, period, status = forecast_next(
            product_data,
            best["model"],
        )

        if status == "ok":
            margin = INTERVAL_Z * best["RMSE"]
            lower = max(0, forecast - margin)
            upper = forecast + margin
        else:
            lower = np.nan
            upper = np.nan

        forecasts.append({
            "item": best["item"],
            "variety": best["variety"],
            "unit": best["unit"],
            "forecast_period_start": period,
            "forecast_period_end": period + pd.Timedelta(days=13),
            "model": best["model"],
            "forecast": forecast,
            "lower_bound": lower,
            "upper_bound": upper,
            "validation_MAE": best["MAE"],
            "validation_RMSE": best["RMSE"],
            "status": status,
        })

    return pd.DataFrame(forecasts)


def external_factor_comparison(metrics):
    table = metrics[
        metrics["model"].isin(
            ["Ridge", "Ridge + external"]
        )
    ].pivot(
        index="item",
        columns="model",
        values="MAE",
    )

    table["improvement_pct"] = (
        (table["Ridge"] - table["Ridge + external"])
        / table["Ridge"]
        * 100
    )

    return table.round(2)


if __name__ == "__main__":
    source_data = pd.read_csv(
        ANALYSIS_DATASET_FILE,
        parse_dates=["period_start"],
    )

    data, model_start = prepare_data(source_data)

    metrics = evaluate_models(data)
    forecasts = create_forecasts(data, metrics)
    best = get_best_models(metrics)

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

    print("ПРОГНОЗИРОВАНИЕ")
    print("\nНачало моделирования:", model_start.date())
    print("Начало тестового периода:", TEST_START_DATE.date())

    print("\nЛучшие модели:")
    print(
        best[
            [
                "item",
                "model",
                "MAE",
                "RMSE",
                "evaluated_periods",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\nКоличество побед:")
    print(best["model"].value_counts().to_string())

    not_converged = metrics[
        (metrics["model"] == "SARIMAX")
        & (metrics["status"] == "not_converged")
    ]

    if not not_converged.empty:
        print("\nSARIMAX не сошёлся для:")
        print(", ".join(not_converged["item"]))

    print("\nВлияние внешних факторов:")
    print(
        external_factor_comparison(
            metrics
        ).to_string()
    )

    print("\nИтоговый прогноз:")
    print(
        forecasts[
            [
                "item",
                "forecast_period_start",
                "model",
                "forecast",
                "lower_bound",
                "upper_bound",
                "status",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\nМетрики сохранены:", MODEL_METRICS_FILE)
    print("Прогнозы сохранены:", FORECASTS_FILE)