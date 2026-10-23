import matplotlib.pyplot as plt
import pandas as pd

from config.settings import (
    ANALYSIS_DATASET_FILE,
    MODEL_METRICS_FILE,
    FORECASTS_FILE,
)
from src.analysis.analysis import analyze_external_factors


FIGURES_DIR = MODEL_METRICS_FILE.parent / "figures"

MODEL_ORDER = [
    "Seasonal Naive",
    "SARIMAX",
    "Ridge",
    "Ridge + external",
]

FACTOR_NAMES = {
    "temperature_mean_lag_2": "Temperature",
    "precipitation_sum_lag_2": "Precipitation",
    "diesel_price_lag_2": "Diesel",
    "fertiliser_index_lag_1": "Fertiliser",
}


def save_figure(fig, filename):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_heatmap(data, title, filename, rotation=0, vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    image = ax.imshow(
        data.values,
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_xticks(range(len(data.columns)))
    ax.set_xticklabels(data.columns, rotation=rotation)
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels(data.index)
    ax.set_title(title)

    fig.colorbar(image, ax=ax)
    fig.tight_layout()

    return save_figure(fig, filename)


def plot_price_indices(df):
    data = (
        df.dropna(subset=["price"])
        .sort_values(["item", "period_start"])
        .copy()
    )

    data["price_index"] = (
        data["price"]
        / data.groupby("item")["price"].transform("first")
        * 100
    )

    fig, ax = plt.subplots(figsize=(12, 6))

    for item, product in data.groupby("item"):
        ax.plot(
            product["period_start"],
            product["price_index"],
            label=item,
        )

    ax.set(
        title="Price indices by product (first value = 100)",
        xlabel="Date",
        ylabel="Index",
    )
    ax.legend(ncol=3, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    return save_figure(fig, "price_indices.png")


def plot_forecast_changes(df, forecasts):
    latest = (
        df[df["period_start"] == df["period_start"].max()]
        [["item", "price"]]
        .rename(columns={"price": "last_price"})
    )

    data = (
        forecasts[forecasts["status"] == "ok"]
        .merge(latest, on="item", how="left")
        .dropna(subset=["forecast", "last_price"])
    )

    data["change_pct"] = (
        (data["forecast"] - data["last_price"])
        / data["last_price"]
        * 100
    )
    data = data.sort_values("change_pct")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(data["item"], data["change_pct"])
    ax.axvline(0, linewidth=1)

    ax.set(
        title="Expected price change for next period",
        xlabel="Change, %",
        ylabel="Product",
    )
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    return save_figure(fig, "forecast_changes.png")


def plot_model_heatmap(metrics):
    valid = metrics[metrics["status"] == "ok"]

    baseline = (
        valid[valid["model"] == "Seasonal Naive"][["item", "MAE"]]
        .rename(columns={"MAE": "baseline_mae"})
    )

    data = valid.merge(baseline, on="item", how="left")
    data["relative_mae"] = data["MAE"] / data["baseline_mae"]

    heatmap = (
        data.pivot(index="item", columns="model", values="relative_mae")
        .reindex(columns=MODEL_ORDER)
        .sort_index()
    )

    return plot_heatmap(
        heatmap,
        "Relative model error (MAE / Seasonal Naive MAE)",
        "model_heatmap.png",
        rotation=20,
    )


def plot_seasonality_heatmap(df):
    monthly = (
        df.dropna(subset=["price"])
        .assign(month=lambda x: x["period_start"].dt.month)
        .groupby(["item", "month"])["price"]
        .mean()
        .unstack()
        .reindex(columns=range(1, 13))
    )

    normalized = monthly.div(monthly.mean(axis=1), axis=0)

    return plot_heatmap(
        normalized.sort_index(),
        "Seasonality by product",
        "seasonality_heatmap.png",
    )


def plot_external_factors_heatmap(df):
    heatmap = (
        analyze_external_factors(df)
        .rename(columns=FACTOR_NAMES)
        [list(FACTOR_NAMES.values())]
        .sort_index()
    )

    return plot_heatmap(
        heatmap,
        "Correlation with external factors",
        "external_factors_heatmap.png",
        vmin=-1,
        vmax=1,
    )


def create_all_visualizations(df, metrics, forecasts):
    general = [
        plot_price_indices(df),
        plot_forecast_changes(df, forecasts),
        plot_model_heatmap(metrics),
        plot_seasonality_heatmap(df),
        plot_external_factors_heatmap(df),
    ]

    return general, []


if __name__ == "__main__":
    data = pd.read_csv(
        ANALYSIS_DATASET_FILE,
        parse_dates=["period_start"],
    )
    metrics = pd.read_csv(MODEL_METRICS_FILE)
    forecasts = pd.read_csv(FORECASTS_FILE)

    general, products = create_all_visualizations(
        data,
        metrics,
        forecasts,
    )

    print("Общих графиков:", len(general))
    print("Графиков по товарам:", len(products))
    print("Всего графиков:", len(general) + len(products))
    print("Папка:", FIGURES_DIR)