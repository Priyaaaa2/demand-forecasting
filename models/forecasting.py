"""
Week 2–4: Forecasting Models
Baselines → ARIMA/SARIMA → Prophet → LSTM
Outputs model comparison report with RMSE and MAPE.
"""
import sqlite3
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

DB_PATH = Path(__file__).parent.parent / "data" / "retail_sales.db"
OUT_DIR = Path(__file__).parent

PRODUCT_ID = "p05"   # Protein Powder — highest volume for forecasting
STORE_ID   = "store_1"
FORECAST_DAYS = 30


# ── Data Loading ──────────────────────────────────────────────────────────────
def load_series(product_id: str, store_id: str) -> pd.Series:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(f"""
        SELECT date, SUM(units_sold) AS units
        FROM sales
        WHERE product_id='{product_id}' AND store_id='{store_id}'
        GROUP BY date ORDER BY date
    """, conn, parse_dates=["date"])
    conn.close()
    df = df.set_index("date").asfreq("D").fillna(0)
    return df["units"]


def train_test_split(series: pd.Series, test_days: int = 30):
    return series[:-test_days], series[-test_days:]


def mape(actual, predicted) -> float:
    actual, predicted = np.array(actual), np.array(predicted)
    mask = actual != 0
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100


def rmse(actual, predicted) -> float:
    return np.sqrt(mean_squared_error(actual, predicted))


# ── 1. Naive Baseline ─────────────────────────────────────────────────────────
def naive_forecast(train: pd.Series, n: int) -> np.ndarray:
    """Predict last observed value for all future periods."""
    return np.full(n, train.iloc[-1])


# ── 2. Moving Average ─────────────────────────────────────────────────────────
def moving_average_forecast(train: pd.Series, n: int, window: int = 7) -> np.ndarray:
    baseline = train.rolling(window).mean().iloc[-1]
    return np.full(n, baseline)


# ── 3. Seasonal Naive (Weekly) ────────────────────────────────────────────────
def seasonal_naive_forecast(train: pd.Series, n: int, season: int = 7) -> np.ndarray:
    tail = train.iloc[-season:].values
    result = []
    for i in range(n):
        result.append(tail[i % season])
    return np.array(result)


# ── 4. SARIMA ─────────────────────────────────────────────────────────────────
def sarima_forecast(train: pd.Series, n: int) -> np.ndarray:
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        model = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,7),
                        enforce_stationarity=False, enforce_invertibility=False)
        fit = model.fit(disp=False)
        forecast = fit.forecast(steps=n)
        return np.maximum(0, forecast.values)
    except Exception as e:
        print(f"  SARIMA failed: {e}. Using seasonal naive.")
        return seasonal_naive_forecast(train, n)


# ── 5. Prophet ────────────────────────────────────────────────────────────────
def prophet_forecast(train: pd.Series, n: int) -> np.ndarray:
    try:
        from prophet import Prophet
        df = train.reset_index().rename(columns={"date": "ds", "units": "y"})
        m = Prophet(yearly_seasonality=True, weekly_seasonality=True,
                    daily_seasonality=False, seasonality_mode="multiplicative")
        m.fit(df)
        future = m.make_future_dataframe(periods=n)
        forecast = m.predict(future)
        return np.maximum(0, forecast["yhat"].values[-n:])
    except Exception as e:
        print(f"  Prophet failed: {e}. Using seasonal naive.")
        return seasonal_naive_forecast(train, n)


# ── 6. LSTM ───────────────────────────────────────────────────────────────────
def lstm_forecast(train: pd.Series, n: int, lookback: int = 14) -> np.ndarray:
    try:
        import tensorflow as tf
        from sklearn.preprocessing import MinMaxScaler

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(train.values.reshape(-1, 1))

        X, y = [], []
        for i in range(lookback, len(scaled)):
            X.append(scaled[i-lookback:i, 0])
            y.append(scaled[i, 0])
        X, y = np.array(X), np.array(y)
        X = X.reshape(X.shape[0], X.shape[1], 1)

        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(lookback, 1)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(1),
        ])
        model.compile(optimizer="adam", loss="mse")
        model.fit(X, y, epochs=20, batch_size=16, verbose=0)

        preds = []
        last  = scaled[-lookback:].reshape(1, lookback, 1)
        for _ in range(n):
            p = model.predict(last, verbose=0)[0, 0]
            preds.append(p)
            last = np.append(last[:, 1:, :], [[[p]]], axis=1)

        return np.maximum(0, scaler.inverse_transform(
            np.array(preds).reshape(-1, 1)).flatten())
    except Exception as e:
        print(f"  LSTM failed: {e}. Using seasonal naive.")
        return seasonal_naive_forecast(train, n)


# ── Compare All Models ────────────────────────────────────────────────────────
def run_comparison():
    print(f"\n🔮  Forecasting {PRODUCT_ID} @ {STORE_ID} — {FORECAST_DAYS}-day horizon\n")
    series = load_series(PRODUCT_ID, STORE_ID)
    train, test = train_test_split(series, FORECAST_DAYS)

    models = {
        "Naive Baseline":  naive_forecast(train, FORECAST_DAYS),
        "Moving Avg (7d)": moving_average_forecast(train, FORECAST_DAYS),
        "Seasonal Naive":  seasonal_naive_forecast(train, FORECAST_DAYS),
        "SARIMA(1,1,1)x(1,1,1,7)": sarima_forecast(train, FORECAST_DAYS),
        "Prophet":         prophet_forecast(train, FORECAST_DAYS),
        "LSTM":            lstm_forecast(train, FORECAST_DAYS),
    }

    results = []
    for name, preds in models.items():
        r = rmse(test.values, preds)
        m = mape(test.values, preds)
        results.append({"Model": name, "RMSE": round(r, 3), "MAPE (%)": round(m, 2)})
        print(f"  {name:<35} RMSE={r:.3f}  MAPE={m:.2f}%")

    results_df = pd.DataFrame(results).sort_values("RMSE")
    results_df.to_csv(OUT_DIR / "model_comparison.csv", index=False)

    best = results_df.iloc[0]["Model"]
    print(f"\n  🏆  Best model: {best}")
    print(f"     RMSE = {results_df.iloc[0]['RMSE']}")
    print(f"     MAPE = {results_df.iloc[0]['MAPE (%)']:.2f}%")

    # Save forecasts for the API
    best_preds = models[best]
    future_dates = pd.date_range(train.index[-1] + pd.Timedelta("1D"), periods=FORECAST_DAYS)
    forecast_df = pd.DataFrame({
        "date":     future_dates.strftime("%Y-%m-%d"),
        "forecast": best_preds.round(0).astype(int),
        "model":    best,
    })
    forecast_df.to_csv(OUT_DIR / "best_forecast.csv", index=False)

    return results_df, models, train, test


if __name__ == "__main__":
    run_comparison()
