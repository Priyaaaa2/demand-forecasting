"""
Week 6: FastAPI — Forecast + Inventory Recommendation Service
Run: uvicorn api.main:app --reload --port 8000

Endpoints:
  GET  /health
  GET  /products
  POST /forecast
  POST /recommend_inventory
  GET  /model_comparison
  GET  /stockout_impact
"""
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.inventory_optimizer import generate_recommendations, stockout_impact_report
from models.forecasting import (
    load_series, seasonal_naive_forecast, sarima_forecast
)

DB_PATH   = Path(__file__).parent.parent / "data" / "retail_sales.db"
MODEL_DIR = Path(__file__).parent.parent / "models"

app = FastAPI(
    title="Demand Forecasting API",
    description="Demand forecasting + inventory optimisation for retail.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


# ── Schemas ────────────────────────────────────────────────────────────────────

class ForecastRequest(BaseModel):
    product_id: str
    store_id: str
    horizon_days: int = 30
    model: str = "sarima"   # naive | seasonal_naive | sarima


class InventoryRequest(BaseModel):
    product_id: Optional[str] = None
    forecast_units: Optional[int] = None
    service_level: float = 0.95


# ── Helpers ────────────────────────────────────────────────────────────────────

def db_conn():
    return sqlite3.connect(DB_PATH)


def validate_product(product_id: str):
    conn = db_conn()
    row = pd.read_sql(
        f"SELECT product_id FROM products WHERE product_id='{product_id}'", conn
    )
    conn.close()
    if row.empty:
        raise HTTPException(status_code=404, detail=f"Product '{product_id}' not found.")


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Status"])
def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.get("/products", tags=["Data"])
def list_products():
    conn = db_conn()
    df = pd.read_sql("SELECT * FROM products", conn)
    conn.close()
    return df.to_dict(orient="records")


@app.post("/forecast", tags=["Forecasting"])
def forecast(req: ForecastRequest):
    validate_product(req.product_id)
    try:
        series = load_series(req.product_id, req.store_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not load series. Check product/store ID.")

    if req.model == "naive":
        preds = np.full(req.horizon_days, float(series.iloc[-1]))
    elif req.model == "seasonal_naive":
        preds = seasonal_naive_forecast(series, req.horizon_days)
    else:
        preds = sarima_forecast(series, req.horizon_days)

    preds = np.maximum(0, preds).round(1)
    future_dates = pd.date_range(
        series.index[-1] + pd.Timedelta("1D"), periods=req.horizon_days
    )

    return {
        "product_id":  req.product_id,
        "store_id":    req.store_id,
        "model":       req.model,
        "horizon_days": req.horizon_days,
        "generated_at": datetime.utcnow().isoformat(),
        "forecast": [
            {"date": d.strftime("%Y-%m-%d"), "units": int(p)}
            for d, p in zip(future_dates, preds)
        ],
        "summary": {
            "total_units_forecast": int(preds.sum()),
            "avg_daily_units":      round(preds.mean(), 1),
            "peak_day_units":       int(preds.max()),
        }
    }


@app.post("/recommend_inventory", tags=["Inventory"])
def recommend_inventory(req: InventoryRequest):
    recs = generate_recommendations(forecast_units=req.forecast_units)
    if req.product_id:
        recs = recs[recs["product_id"] == req.product_id]
        if recs.empty:
            raise HTTPException(status_code=404, detail="Product not found.")
    return {
        "generated_at": datetime.utcnow().isoformat(),
        "service_level_pct": req.service_level * 100,
        "recommendations": recs.to_dict(orient="records"),
    }


@app.get("/model_comparison", tags=["Forecasting"])
def model_comparison():
    path = MODEL_DIR / "model_comparison.csv"
    if not path.exists():
        raise HTTPException(status_code=404,
                            detail="Run models/forecasting.py first to generate comparison.")
    df = pd.read_csv(path)
    return df.to_dict(orient="records")


@app.get("/stockout_impact", tags=["Business Impact"])
def stockout_impact():
    return stockout_impact_report()


@app.get("/demand_trend", tags=["Data"])
def demand_trend(
    product_id: str = Query("p05"),
    store_id:   str = Query("store_1"),
    days:       int = Query(90),
):
    validate_product(product_id)
    conn = db_conn()
    df = pd.read_sql(f"""
        SELECT date, SUM(units_sold) AS units
        FROM sales
        WHERE product_id='{product_id}' AND store_id='{store_id}'
          AND date >= DATE('2024-12-31', '-{days} days')
        GROUP BY date ORDER BY date
    """, conn)
    conn.close()
    return df.to_dict(orient="records")
