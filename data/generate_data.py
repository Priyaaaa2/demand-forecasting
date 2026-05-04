"""
Project 2 — Demand Forecasting: Data Generation
Simulates 2 years of daily retail sales for 10 products across 5 stores.
"""
import random
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

random.seed(0); np.random.seed(0)

START = pd.Timestamp("2023-01-01")
END   = pd.Timestamp("2024-12-31")
dates = pd.date_range(START, END, freq="D")

STORES   = [f"store_{i}" for i in range(1, 6)]
PRODUCTS = {
    "p01": {"name": "Wireless Headphones", "base_demand": 12, "price": 89.99, "cost": 34.0,  "lead_days": 7,  "category": "Electronics"},
    "p02": {"name": "Running Shoes",        "base_demand": 18, "price": 65.00, "cost": 22.0,  "lead_days": 5,  "category": "Footwear"},
    "p03": {"name": "Coffee Maker",         "base_demand":  8, "price": 49.99, "cost": 18.5,  "lead_days": 10, "category": "Kitchen"},
    "p04": {"name": "Yoga Mat",             "base_demand": 22, "price": 28.99, "cost":  8.0,  "lead_days": 4,  "category": "Sports"},
    "p05": {"name": "Protein Powder",       "base_demand": 35, "price": 34.99, "cost": 12.0,  "lead_days": 3,  "category": "Nutrition"},
    "p06": {"name": "Bluetooth Speaker",    "base_demand": 10, "price": 59.99, "cost": 21.0,  "lead_days": 7,  "category": "Electronics"},
    "p07": {"name": "Moisturiser SPF50",    "base_demand": 28, "price": 18.99, "cost":  5.5,  "lead_days": 5,  "category": "Beauty"},
    "p08": {"name": "Resistance Bands Set", "base_demand": 20, "price": 22.99, "cost":  6.0,  "lead_days": 4,  "category": "Sports"},
    "p09": {"name": "Slim-Fit Jeans",       "base_demand": 15, "price": 44.99, "cost": 14.0,  "lead_days": 8,  "category": "Clothing"},
    "p10": {"name": "Instant Pot",          "base_demand":  6, "price": 79.99, "cost": 28.0,  "lead_days": 10, "category": "Kitchen"},
}

OUT = Path(__file__).parent
DB  = OUT / "retail_sales.db"


def seasonal_factor(date: pd.Timestamp) -> float:
    """Cosine seasonality: peaks in Dec (Christmas) and dips in Feb."""
    day_of_year = date.day_of_year
    return 1 + 0.45 * np.cos(2 * np.pi * (day_of_year - 355) / 365)


def trend_factor(date: pd.Timestamp) -> float:
    """Slight upward trend over 2 years."""
    days_elapsed = (date - START).days
    return 1 + 0.0002 * days_elapsed


def day_of_week_factor(date: pd.Timestamp) -> float:
    """Weekends sell ~30% more for most categories."""
    return 1.30 if date.dayofweek >= 5 else 1.0


def generate_sales() -> pd.DataFrame:
    rows = []
    for store in STORES:
        for pid, meta in PRODUCTS.items():
            for date in dates:
                base = meta["base_demand"]
                # Seasonality × Trend × Day-of-week × Store multiplier × Noise
                store_mult = 0.8 + 0.4 * STORES.index(store) / len(STORES)
                mu = (base * seasonal_factor(date) * trend_factor(date)
                      * day_of_week_factor(date) * store_mult)
                units = max(0, int(np.random.poisson(max(1, mu))))

                # Random promotions (10% of days, 40% demand lift)
                is_promo = random.random() < 0.10
                if is_promo:
                    units = int(units * 1.4)

                rows.append({
                    "date":        date.strftime("%Y-%m-%d"),
                    "store_id":    store,
                    "product_id":  pid,
                    "units_sold":  units,
                    "revenue_gbp": round(units * meta["price"], 2),
                    "is_promo":    int(is_promo),
                    "day_of_week": date.day_name(),
                    "month":       date.month,
                    "year":        date.year,
                    "week":        date.isocalendar().week,
                })

    df = pd.DataFrame(rows)
    print(f"  ✅  Sales rows: {len(df):,}")
    return df


def generate_inventory(sales: pd.DataFrame) -> pd.DataFrame:
    """Simulate inventory levels with reorder logic."""
    rows = []
    for store in STORES:
        for pid, meta in PRODUCTS.items():
            inv  = meta["base_demand"] * 14   # start with 2-week stock
            reorder_point = meta["base_demand"] * (meta["lead_days"] + 2)
            eoq  = int(np.sqrt(2 * meta["base_demand"] * 365 * 10 / 0.20 / meta["cost"]))

            for date in dates:
                day_sales = sales[
                    (sales["date"] == date.strftime("%Y-%m-%d")) &
                    (sales["store_id"] == store) &
                    (sales["product_id"] == pid)
                ]["units_sold"].values
                sold = int(day_sales[0]) if len(day_sales) else 0
                stockout = max(0, sold - inv)
                inv = max(0, inv - sold)

                if inv <= reorder_point:
                    inv += eoq

                rows.append({
                    "date":           date.strftime("%Y-%m-%d"),
                    "store_id":       store,
                    "product_id":     pid,
                    "inventory_level": inv,
                    "reorder_point":  reorder_point,
                    "eoq":            eoq,
                    "stockout_units": stockout,
                    "is_stockout":    int(stockout > 0),
                })

    return pd.DataFrame(rows)


def save(sales, inventory):
    csv = OUT / "csv"
    csv.mkdir(exist_ok=True)

    products_df = pd.DataFrame([
        {"product_id": pid, **meta} for pid, meta in PRODUCTS.items()
    ])

    for df, name in [(sales, "sales"), (inventory, "inventory"), (products_df, "products")]:
        df.to_csv(csv / f"{name}.csv", index=False)

    conn = sqlite3.connect(DB)
    sales.to_sql("sales",          conn, if_exists="replace", index=False)
    inventory.to_sql("inventory",  conn, if_exists="replace", index=False)
    products_df.to_sql("products", conn, if_exists="replace", index=False)
    conn.close()
    print(f"  ✅  DB saved: {DB}")


if __name__ == "__main__":
    print("🚀 Generating retail sales dataset …")
    sales = generate_sales()
    print("📦 Simulating inventory …")
    inventory = generate_inventory(sales)
    save(sales, inventory)
    print(f"\n📊  Stockout rate: {inventory['is_stockout'].mean()*100:.1f}%")
