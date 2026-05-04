# 🔮 Demand Forecasting + Inventory Optimisation System

> **Production-grade ML system: time series forecasting pipeline → inventory decisions → FastAPI deployment → Docker containerisation.**

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Deploy-Docker-2496ED)](https://docker.com)

---

## 🎯 What This Project Demonstrates

| Skill | Implementation |
|-------|---------------|
| **Time Series** | Naive, Moving Average, Seasonal Naive, SARIMA, Prophet, LSTM |
| **Optimisation** | EOQ, Safety Stock, Reorder Point (Operations Research) |
| **ML Engineering** | Model comparison, RMSE/MAPE evaluation, best-model selection |
| **API Design** | FastAPI with `/forecast` + `/recommend_inventory` endpoints |
| **Deployment** | Dockerised service, production-ready |
| **Business Impact** | Stockout reduction quantified in £ |

---

## 🏗️ Architecture

```
Retail Sales DB (SQLite)
        │
        ▼
┌──────────────────────────────────┐
│         Forecasting Pipeline     │
│  Naive → MA → Seasonal Naive     │
│  → SARIMA → Prophet → LSTM       │
│  (ranked by RMSE + MAPE)         │
└────────────────┬─────────────────┘
                 │ Best Model Forecast
                 ▼
┌──────────────────────────────────┐
│      Inventory Optimiser         │
│  Safety Stock = Z × σ × √LT     │
│  EOQ = √(2DS/hC)                │
│  ROP = μ × LT + SS              │
└────────────────┬─────────────────┘
                 │ Recommendations
                 ▼
┌──────────────────────────────────┐
│    FastAPI Service (port 8000)   │
│  POST /forecast                  │
│  POST /recommend_inventory       │
│  GET  /stockout_impact           │
│  GET  /model_comparison          │
└──────────────────────────────────┘
                 │
                 ▼
         🐳 Docker Container
```

---

## 📁 Project Structure

```
project2-demand-forecasting/
├── data/
│   ├── generate_data.py      # 2-year retail sales (10 products × 5 stores)
│   ├── retail_sales.db       # SQLite database
│   └── csv/                  # CSV exports
├── models/
│   ├── forecasting.py        # 6 forecasting models + comparison
│   ├── inventory_optimizer.py # EOQ + safety stock + ROP
│   ├── model_comparison.csv  # RMSE/MAPE leaderboard
│   └── inventory_recommendations.csv
├── api/
│   └── main.py               # FastAPI application
├── Dockerfile
└── requirements.txt
```

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate 2 years of retail data
python data/generate_data.py

# 3. Run model comparison
python models/forecasting.py

# 4. Generate inventory recommendations
python models/inventory_optimizer.py

# 5. Start API
uvicorn api.main:app --reload --port 8000
# Docs at: http://localhost:8000/docs
```

### Docker

```bash
docker build -t demand-forecasting .
docker run -p 8000:8000 demand-forecasting
```

---

## 📊 Dataset

| Table | Rows | Description |
|-------|------|-------------|
| `sales` | 36,500 | Daily sales: 10 products × 5 stores × 730 days |
| `inventory` | 36,500 | Daily inventory levels + reorder events |
| `products` | 10 | Price, cost, lead time, category |

**Synthetic features:**
- 📈 Seasonal Christmas peak (+40% in Dec/Jan)
- 📅 Weekend uplift (+35%)
- 📊 Upward trend (+0.02%/day)
- 🏷️ Promotional days (10% of days, +40% demand)

---

## 🤖 Model Comparison

| Model | RMSE | MAPE |
|-------|------|------|
| Moving Average (7d) | **9.84** | **14.9%** |
| Naive Baseline | 9.89 | 15.9% |
| Seasonal Naive | 10.15 | 13.5% |
| SARIMA(1,1,1)×(1,1,1,7) | 10.43 | 16.7% |
| Prophet *(optional)* | — | — |
| LSTM *(optional)* | — | — |

---

## 📦 Inventory Recommendations (Sample)

| Product | Safety Stock | Reorder Point | EOQ | Annual Cost |
|---------|-------------|---------------|-----|-------------|
| Protein Powder | 49 units | 171 units | 704 units | £2,257 |
| Yoga Mat | 37 units | 140 units | 684 units | £1,440 |
| Wireless Headphones | 29 units | 127 units | 245 units | £2,323 |

---

## 🔌 API Endpoints

```bash
# Forecast 30 days of demand
curl -X POST http://localhost:8000/forecast \
  -H "Content-Type: application/json" \
  -d '{"product_id": "p05", "store_id": "store_1", "horizon_days": 30}'

# Get inventory recommendation
curl -X POST http://localhost:8000/recommend_inventory \
  -H "Content-Type: application/json" \
  -d '{"product_id": "p05"}'

# Business impact report
curl http://localhost:8000/stockout_impact
```

---

## 💼 Business Impact (Resume Bullet)

> *"Built and deployed demand forecasting system (SARIMA + Moving Average) with FastAPI + Docker. Implemented inventory optimisation using EOQ and safety stock models, projecting 18–30% stockout reduction and measurable holding cost savings across 10 product lines."*
