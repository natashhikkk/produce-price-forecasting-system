import subprocess
import sys

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from config.settings import (
    BASE_DIR,
    ANALYSIS_DATASET_FILE,
    MODEL_METRICS_FILE,
    FORECASTS_FILE,
    ANALYTICAL_CONCLUSIONS_FILE,
    LOAD_LOG_FILE,
)
from src.analysis.analysis import analyze_external_factors


st.set_page_config(page_title="Прогноз оптовых цен", layout="wide")

MONTHS = [
    "Янв", "Фев", "Мар", "Апр", "Май", "Июн",
    "Июл", "Авг", "Сен", "Окт", "Ноя", "Дек",
]

PERIODS = {
    "Последний год": 1,
    "Последние 3 года": 3,
    "Последние 5 лет": 5,
}

CORRELATIONS = {
    "temperature_mean_lag_2": "Температура\nlag 2",
    "precipitation_sum_lag_2": "Осадки\nlag 2",
    "diesel_price_lag_2": "Дизель\nlag 2",
    "fertiliser_index_lag_1": "Удобрения\nlag 1",
}


@st.cache_data
def load_results():
    data = pd.read_csv(ANALYSIS_DATASET_FILE, parse_dates=["period_start"])
    metrics = pd.read_csv(MODEL_METRICS_FILE)
    forecasts = pd.read_csv(
        FORECASTS_FILE,
        parse_dates=["forecast_period_start", "forecast_period_end"],
    )
    conclusions = pd.read_csv(ANALYTICAL_CONCLUSIONS_FILE)
    log = (
        pd.read_csv(LOAD_LOG_FILE, encoding="utf-8-sig")
        if LOAD_LOG_FILE.exists()
        else pd.DataFrame()
    )
    return data, metrics, forecasts, conclusions, log


def update_system():
    with st.spinner("Обновление данных и построение прогнозов..."):
        result = subprocess.run(
            [sys.executable, str(BASE_DIR / "main.py")],
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            errors="replace",
        )

    if result.returncode == 0 and "Работа системы завершена успешно" in result.stdout:
        st.cache_data.clear()
        st.session_state["updated"] = True
        st.rerun()

    st.error("Не удалось выполнить обновление.")
    with st.expander("Подробности ошибки"):
        st.code(result.stdout + "\n" + result.stderr)


def plot_forecast(data, forecast, period):
    history = data

    if period != "Вся история":
        start = data["period_start"].max() - pd.DateOffset(years=PERIODS[period])
        history = data[data["period_start"] >= start]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        history["period_start"],
        history["price"],
        marker="o",
        markersize=3,
        label="Фактическая цена",
    )

    if forecast["status"] == "ok":
        latest = data.dropna(subset=["price"]).iloc[-1]

        ax.plot(
            [latest["period_start"], forecast["forecast_period_start"]],
            [latest["price"], forecast["forecast"]],
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
            label="Прогноз",
        )

    ax.set(
        title="История цены и прогноз",
        xlabel="Дата",
        ylabel=f"Цена, £/{data['unit'].iloc[0]}",
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_seasonality(data):
    monthly = (
        data.dropna(subset=["price"])
        .assign(month=lambda x: x["period_start"].dt.month)
        .groupby("month")["price"]
        .mean()
        .reindex(range(1, 13))
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(monthly.index, monthly.values, marker="o")
    ax.set_xticks(range(1, 13), MONTHS)
    ax.set(
        title="Сезонность",
        xlabel="Месяц",
        ylabel=f"Средняя цена, £/{data['unit'].iloc[0]}",
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_correlations(row):
    labels = list(CORRELATIONS.values())
    values = [row[column] for column in CORRELATIONS]

    fig, ax = plt.subplots(figsize=(8, 4))
    image = ax.imshow([values], vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")

    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks([0], ["Цена"])
    ax.set_title("Корреляция с внешними факторами")

    for i, value in enumerate(values):
        if pd.notna(value):
            ax.text(i, 0, f"{value:.2f}", ha="center", va="center")

    fig.colorbar(image, ax=ax, label="Корреляция")
    fig.tight_layout()
    return fig


st.title("Мониторинг и прогнозирование оптовых цен")
st.caption("Плодоовощная продукция Англии и Уэльса")

required = [
    ANALYSIS_DATASET_FILE,
    MODEL_METRICS_FILE,
    FORECASTS_FILE,
    ANALYTICAL_CONCLUSIONS_FILE,
]

if not all(path.exists() for path in required):
    st.warning("Результаты системы ещё не сформированы.")

    if st.button("Загрузить данные и выполнить расчёты"):
        update_system()

    st.stop()


data, metrics, forecasts, conclusions, log = load_results()
correlations = analyze_external_factors(data)


if st.button("Обновить данные и пересчитать", use_container_width=True):
    update_system()


col1, col2 = st.columns(2)

with col1:
    item = st.selectbox("Товар", sorted(data["item"].unique()))

with col2:
    period = st.selectbox(
        "Период отображения истории",
        [*PERIODS, "Вся история"],
        index=1,
    )


if st.session_state.pop("updated", False):
    st.success("Данные успешно обновлены.")


product = (
    data[data["item"] == item]
    .sort_values("period_start")
    .reset_index(drop=True)
)

forecast = forecasts[forecasts["item"] == item].iloc[0]
product_metrics = metrics[metrics["item"] == item].copy()
conclusion = conclusions[conclusions["item"] == item].iloc[0]

latest = product.dropna(subset=["price"]).iloc[-1]
latest_price = latest["price"]
unit = latest["unit"]

last_update = "нет данных"

if not log.empty:
    dates = pd.to_datetime(log["datetime"], errors="coerce")
    if dates.notna().any():
        last_update = dates.max().strftime("%d.%m.%Y %H:%M")


st.caption(
    f"Разновидность: {latest['variety']} · "
    f"Единица: {unit} · Последнее обновление: {last_update}"
)

st.caption(
    "Период отображения изменяет только исторический график. "
    "Прогноз строится по всей доступной истории."
)


if not log.empty:
    log = log.copy()
    log["datetime"] = pd.to_datetime(log["datetime"], errors="coerce")
    latest_log = log.sort_values("datetime").groupby("source").tail(1)

    fallback = latest_log[latest_log["status"] == "fallback"]["source"].tolist()
    errors = latest_log[latest_log["status"] == "error"]["source"].tolist()

    if fallback:
        st.warning("Используются локальные данные: " + ", ".join(fallback))
    if errors:
        st.error("Недоступны источники: " + ", ".join(errors))


change = (
    (forecast["forecast"] - latest_price) / latest_price * 100
    if pd.notna(forecast["forecast"])
    else None
)


st.divider()

left, right = st.columns([1, 2])

with left:
    st.markdown(f"### {item}")
    st.markdown(
        f"#### Прогноз на {forecast['forecast_period_start']:%d.%m.%Y} — "
        f"{forecast['forecast_period_end']:%d.%m.%Y}"
    )

    st.caption("Прогноз")
    st.markdown(
        f"## {forecast['forecast']:.3f} £/{unit}"
        if pd.notna(forecast["forecast"])
        else "## Недоступен"
    )

    interval = (
        f"{forecast['lower_bound']:.3f} – "
        f"{forecast['upper_bound']:.3f} £/{unit}"
        if pd.notna(forecast["lower_bound"])
        else "—"
    )

    st.markdown(f"**Прогнозный интервал:** {interval}")
    st.markdown(f"**Последняя цена:** {latest_price:.3f} £/{unit}")
    st.markdown(
        f"**Ожидаемое изменение:** {change:+.1f}%"
        if change is not None
        else "**Ожидаемое изменение:** —"
    )
    st.markdown(
        f"**Модель:** {forecast['model']}  \n"
        f"MAE: {forecast['validation_MAE']:.4f} · "
        f"RMSE: {forecast['validation_RMSE']:.4f}"
    )

with right:
    fig = plot_forecast(product, forecast, period)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


st.markdown("#### Аналитический вывод")
st.write(conclusion["conclusion"])


st.divider()

left, right = st.columns(2)

with left:
    fig = plot_seasonality(product)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with right:
    fig = plot_correlations(correlations.loc[item])
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


with st.expander("Сравнение прогнозных моделей"):
    table = product_metrics[
        ["model", "MAE", "RMSE", "evaluated_periods", "status"]
    ].copy()

    table.insert(
        0,
        "Выбрана",
        (table["model"] == forecast["model"]).map({True: "✓", False: ""}),
    )

    table = table.rename(
        columns={
            "model": "Модель",
            "evaluated_periods": "Тестовых периодов",
            "status": "Статус",
        }
    )

    st.dataframe(table, hide_index=True, use_container_width=True)


with st.expander("Данные и экспорт"):
    st.write(f"Последний период данных: **{latest['period_start']:%d.%m.%Y}**")
    st.write(f"Периодов для товара: **{len(product)}**")
    st.write(f"Пропущенных цен: **{product['price'].isna().sum()}**")

    d1, d2, d3, d4 = st.columns(4)

    d1.download_button(
        "Скачать данные",
        data.to_csv(index=False).encode("utf-8-sig"),
        "analysis_dataset.csv",
        "text/csv",
        use_container_width=True,
    )

    d2.download_button(
        "Скачать прогнозы",
        forecasts.to_csv(index=False).encode("utf-8-sig"),
        "forecasts.csv",
        "text/csv",
        use_container_width=True,
    )

    d3.download_button(
        "Скачать метрики",
        metrics.to_csv(index=False).encode("utf-8-sig"),
        "model_metrics.csv",
        "text/csv",
        use_container_width=True,
    )

    d4.download_button(
        "Скачать вывод",
        conclusion["conclusion"].encode("utf-8"),
        f"{item}_analytical_conclusion.txt",
        "text/plain",
        use_container_width=True,
    )