import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config.settings import (
    ANALYSIS_DATASET_FILE,
    MODEL_METRICS_FILE,
    FORECASTS_FILE,
)


FIGURES_DIR = MODEL_METRICS_FILE.parent / "figures"
GENERAL_DIR = FIGURES_DIR / "general"

MODEL_ORDER = [
    "Seasonal Naive",
    "SARIMAX",
    "Ridge",
    "Ridge + external",
]

EXTERNAL_FACTORS = {
    "temperature_mean": "Temperature",
    "precipitation_sum": "Precipitation",
    "diesel_price": "Diesel",
    "fertiliser_index": "Fertiliser",
}


def save_figure(fig, folder, filename):
    folder.mkdir(parents=True, exist_ok=True)

    path = folder / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return path


def get_product(df, item):
    data = df[df["item"] == item].copy()

    if data.empty:
        raise ValueError(f"Товар '{item}' не найден.")

    return data.sort_values("period_start")


def plot_product_forecast(df, forecasts, item):
    data = get_product(df, item).tail(26)

    forecast = forecasts[
        forecasts["item"] == item
    ]

    if forecast.empty:
        raise ValueError(f"Прогноз для '{item}' не найден.")

    forecast = forecast.iloc[0]
    actual = data.dropna(subset=["price"])

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        data["period_start"],
        data["price"],
        marker="o",
        label="Actual price",
    )

    if not actual.empty:
        ax.plot(
            [
                actual["period_start"].iloc[-1],
                forecast["forecast_period_start"],
            ],
            [
                actual["price"].iloc[-1],
                forecast["forecast"],
            ],
            linestyle="--",
        )

    ax.errorbar(
        forecast["forecast_period_start"],
        forecast["forecast"],
        yerr=[
            [forecast["forecast"] - forecast["lower_bound"]],
            [forecast["upper_bound"] - forecast["forecast"]],
        ],
        fmt="o",
        capsize=6,
        label=f"Forecast ({forecast['model']})",
    )

    ax.set_title(f"Price history and forecast: {item}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()

    return save_figure(
        fig,
        FIGURES_DIR / item,
        "forecast.png",
    )


def plot_product_model_metrics(metrics, item):
    data = metrics[
        (metrics["item"] == item)
        & (metrics["status"] == "ok")
    ].copy()

    data["model"] = pd.Categorical(
        data["model"],
        categories=MODEL_ORDER,
        ordered=True,
    )

    data = data.sort_values("model")

    x = np.arange(len(data))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(
        x - width / 2,
        data["MAE"],
        width,
        label="MAE",
    )

    ax.bar(
        x + width / 2,
        data["RMSE"],
        width,
        label="RMSE",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        data["model"],
        rotation=20,
    )

    ax.set_title(f"Model comparison: {item}")
    ax.set_ylabel("Error")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()

    return save_figure(
        fig,
        FIGURES_DIR / item,
        "model_metrics.png",
    )


def plot_product_seasonality(df, item):
    data = get_product(df, item)

    monthly = (
        data
        .dropna(subset=["price"])
        .assign(
            month=lambda x:
            x["period_start"].dt.month
        )
        .groupby("month")["price"]
        .mean()
        .reindex(range(1, 13))
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        monthly.index,
        monthly.values,
        marker="o",
    )

    ax.set_title(f"Seasonality: {item}")
    ax.set_xlabel("Month")
    ax.set_ylabel("Average price")
    ax.set_xticks(range(1, 13))
    ax.grid(alpha=0.3)

    fig.tight_layout()

    return save_figure(
        fig,
        FIGURES_DIR / item,
        "seasonality.png",
    )


def plot_price_indices(df):
    data = df.dropna(subset=["price"]).copy()

    first_prices = (
        data
        .sort_values("period_start")
        .groupby("item")["price"]
        .transform("first")
    )

    data["price_index"] = (
        data["price"] / first_prices * 100
    )

    fig, ax = plt.subplots(figsize=(12, 6))

    for item, product in data.groupby("item"):
        ax.plot(
            product["period_start"],
            product["price_index"],
            label=item,
        )

    ax.set_title(
        "Price indices by product (first value = 100)"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Index")
    ax.legend(ncol=3, fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()

    return save_figure(
        fig,
        GENERAL_DIR,
        "price_indices.png",
    )


def plot_forecast_changes(df, forecasts):
    last_prices = (
        df
        .dropna(subset=["price"])
        .sort_values("period_start")
        .groupby("item", as_index=False)
        .last()[["item", "price"]]
        .rename(columns={"price": "last_price"})
    )

    data = forecasts.merge(
        last_prices,
        on="item",
        how="left",
    )

    data["change_pct"] = (
        (data["forecast"] - data["last_price"])
        / data["last_price"]
        * 100
    )

    data = data.sort_values("change_pct")

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.barh(
        data["item"],
        data["change_pct"],
    )

    ax.axvline(0, linewidth=1)

    ax.set_title(
        "Expected price change for next period"
    )
    ax.set_xlabel("Change, %")
    ax.set_ylabel("Product")
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()

    return save_figure(
        fig,
        GENERAL_DIR,
        "forecast_changes.png",
    )


def plot_heatmap(
    data,
    title,
    filename,
    rotation=0,
    vmin=None,
    vmax=None,
):
    fig, ax = plt.subplots(figsize=(10, 6))

    image = ax.imshow(
        data.values,
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_xticks(
        range(len(data.columns))
    )
    ax.set_xticklabels(
        data.columns,
        rotation=rotation,
    )

    ax.set_yticks(
        range(len(data.index))
    )
    ax.set_yticklabels(
        data.index
    )

    ax.set_title(title)

    fig.colorbar(
        image,
        ax=ax,
    )

    fig.tight_layout()

    return save_figure(
        fig,
        GENERAL_DIR,
        filename,
    )


def plot_model_heatmap(metrics):
    baseline = (
        metrics[
            metrics["model"] == "Seasonal Naive"
        ][["item", "MAE"]]
        .rename(
            columns={"MAE": "baseline_mae"}
        )
    )

    data = metrics.merge(
        baseline,
        on="item",
        how="left",
    )

    data["relative_mae"] = (
        data["MAE"]
        / data["baseline_mae"]
    )

    heatmap = data.pivot(
        index="item",
        columns="model",
        values="relative_mae",
    )

    heatmap = heatmap.reindex(
        columns=MODEL_ORDER
    )

    return plot_heatmap(
        heatmap,
        "Relative model error",
        "model_heatmap.png",
        rotation=20,
    )


def plot_seasonality_heatmap(df):
    data = (
        df
        .dropna(subset=["price"])
        .assign(
            month=lambda x:
            x["period_start"].dt.month
        )
    )

    monthly = (
        data
        .groupby(
            ["item", "month"]
        )["price"]
        .mean()
        .unstack()
        .reindex(columns=range(1, 13))
    )

    normalized = monthly.div(
        monthly.mean(axis=1),
        axis=0,
    )

    return plot_heatmap(
        normalized,
        "Seasonality by product",
        "seasonality_heatmap.png",
    )


def plot_external_factors_heatmap(df):
    rows = []

    for item, data in (
        df
        .sort_values("period_start")
        .groupby("item")
    ):
        row = {"item": item}

        for factor, name in (
            EXTERNAL_FACTORS.items()
        ):
            row[name] = data["price"].corr(
                data[factor].shift(1)
            )

        rows.append(row)

    heatmap = (
        pd.DataFrame(rows)
        .set_index("item")
        .sort_index()
    )

    return plot_heatmap(
        heatmap,
        "Correlation with external factors",
        "external_factors_heatmap.png",
        vmin=-1,
        vmax=1,
    )


def create_product_visualizations(
    df,
    metrics,
    forecasts,
    item,
):
    return [
        plot_product_forecast(
            df,
            forecasts,
            item,
        ),
        plot_product_model_metrics(
            metrics,
            item,
        ),
        plot_product_seasonality(
            df,
            item,
        ),
    ]


def create_general_visualizations(
    df,
    metrics,
    forecasts,
):
    return [
        plot_price_indices(df),
        plot_forecast_changes(
            df,
            forecasts,
        ),
        plot_model_heatmap(metrics),
        plot_seasonality_heatmap(df),
        plot_external_factors_heatmap(df),
    ]


def create_all_visualizations(
    df,
    metrics,
    forecasts,
):
    general_paths = (
        create_general_visualizations(
            df,
            metrics,
            forecasts,
        )
    )

    product_paths = []

    for item in sorted(
        df["item"].dropna().unique()
    ):
        product_paths.extend(
            create_product_visualizations(
                df,
                metrics,
                forecasts,
                item,
            )
        )

    return general_paths, product_paths


if __name__ == "__main__":
    df = pd.read_csv(
        ANALYSIS_DATASET_FILE,
        parse_dates=["period_start"],
    )

    metrics = pd.read_csv(
        MODEL_METRICS_FILE
    )

    forecasts = pd.read_csv(
        FORECASTS_FILE,
        parse_dates=[
            "forecast_period_start",
            "forecast_period_end",
        ],
    )

    if len(sys.argv) > 1:
        item = sys.argv[1]

        paths = create_product_visualizations(
            df,
            metrics,
            forecasts,
            item,
        )

        print(
            f"Графики для '{item}' сохранены:"
        )

        for path in paths:
            print(path)

    else:
        general_paths, product_paths = (
            create_all_visualizations(
                df,
                metrics,
                forecasts,
            )
        )

        print(
            "Общих графиков:",
            len(general_paths),
        )

        print(
            "Графиков по товарам:",
            len(product_paths),
        )

        print(
            "Папка:",
            FIGURES_DIR,
        )