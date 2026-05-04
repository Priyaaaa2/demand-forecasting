"""
Streamlit Dashboard — Demand Forecasting & Inventory Optimisation
Run: streamlit run app.py
"""

import sqlite3
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

DB_PATH = Path(__file__).parent / "data" / "retail_sales.db"
MODEL_COMP_PATH = Path(__file__).parent / "models" / "model_comparison.csv"
INV_REC_PATH = Path(__file__).parent / "models" / "inventory_recommendations.csv"

st.set_page_config(
    page_title="FundView | Supply Chain",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }
.stApp { background-color: #eaf5f5; }
.metric-card {
    background: #ffffff;
    border: none;
    border-radius: 20px;
    padding: 1.5rem;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    text-align: left;
    margin-bottom: 1rem;
}
.metric-card.dark { background: #111111; }
.metric-card.dark .value { color: #ffffff; }
.metric-card.dark .label { color: #94a3b8; }
.metric-card .value { font-size: 1.8rem; font-weight: 600; color: #171717; margin-top: 10px; }
.metric-card .label { font-size: 0.9rem; color: #64748b; font-weight: 500; }
.metric-card .delta { font-size: 0.85rem; color: #10b981; font-weight: 600; margin-top: 4px; }
section[data-testid="stSidebar"] { 
    background: #ffffff; 
    border-right: 1px solid #e2e8f0;
}
.stTabs [data-baseweb="tab"] { font-weight: 500; }
h1, h2, h3 { color: #111111; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

COLORS = {
    "primary": "#22c55e", "secondary": "#eab308", "accent": "#10b981",
    "danger": "#ef4444", "bg": "#eaf5f5", "surface": "#ffffff",
    "dark": "#111111", "text": "#171717"
}
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Outfit", color="#111111"),
    xaxis=dict(gridcolor="#f1f5f9", linecolor="#cbd5e1", zerolinecolor="#cbd5e1", showgrid=False),
    yaxis=dict(gridcolor="#f1f5f9", linecolor="#cbd5e1", zerolinecolor="#cbd5e1"),
    margin=dict(l=10, r=10, t=40, b=10),
)

# ── Data Loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    conn = sqlite3.connect(DB_PATH)
    sales = pd.read_sql("SELECT * FROM sales", conn)
    products = pd.read_sql("SELECT * FROM products", conn)
    conn.close()
    
    if MODEL_COMP_PATH.exists():
        comp = pd.read_csv(MODEL_COMP_PATH)
    else:
        comp = pd.DataFrame()
        
    if INV_REC_PATH.exists():
        inv = pd.read_csv(INV_REC_PATH)
    else:
        inv = pd.DataFrame()
        
    return sales, products, comp, inv

sales, products, comp, inv = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## FundView Supply")
    st.markdown("*Demand & Inventory*")
    st.divider()
    page = st.radio("Navigation", ["Monitoring", "Forecasting", "Inventory"])

# ── Monitoring ────────────────────────────────────────────────────────────────
if page == "Monitoring":
    st.title("Monitoring Dashboard")
    
    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    total_rev = sales["revenue_gbp"].sum()
    total_units = sales["units_sold"].sum()
    
    with c1:
        st.markdown(f"""<div class="metric-card dark">
            <div class="label">Total Revenue</div>
            <div class="value">£{total_rev:,.0f}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="label">Units Sold</div>
            <div class="value">{total_units:,.0f}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <div class="label">Products</div>
            <div class="value">{len(products)}</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card">
            <div class="label">Stores</div>
            <div class="value">{sales['store_id'].nunique()}</div>
        </div>""", unsafe_allow_html=True)
        
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Daily Sales Volume")
        daily = sales.groupby("date")["units_sold"].sum().reset_index()
        daily["date"] = pd.to_datetime(daily["date"])
        daily["MA7"] = daily["units_sold"].rolling(7).mean()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=daily["date"], y=daily["units_sold"], name="Sales", line=dict(color=COLORS["primary"], width=1.5), opacity=0.5))
        fig.add_trace(go.Scatter(x=daily["date"], y=daily["MA7"], name="7-Day MA", line=dict(color=COLORS["dark"], width=2.5)))
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Category Breakdown")
        cat_sales = sales.merge(products, on="product_id").groupby("category")["units_sold"].sum().reset_index()
        fig = px.pie(cat_sales, values="units_sold", names="category", 
                     color_discrete_sequence=["#22c55e", "#eab308", "#10b981", "#d9f99d", "#4ade80", "#111111"], hole=0.4)
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

# ── Forecasting ───────────────────────────────────────────────────────────────
elif page == "Forecasting":
    st.title("Demand Forecasting Models")
    
    if not comp.empty:
        best_model = comp.loc[comp["RMSE"].idxmin()]
        
        c1, c2, c3 = st.columns(3)
        c1.markdown(f"""<div class="metric-card dark">
            <div class="label">Best Model</div>
            <div class="value">{best_model['Model']}</div>
        </div>""", unsafe_allow_html=True)
        c2.markdown(f"""<div class="metric-card">
            <div class="label">Lowest RMSE</div>
            <div class="value">{best_model['RMSE']:.2f}</div>
        </div>""", unsafe_allow_html=True)
        c3.markdown(f"""<div class="metric-card">
            <div class="label">MAPE</div>
            <div class="value">{best_model['MAPE (%)']:.2f}%</div>
        </div>""", unsafe_allow_html=True)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Model Comparison (RMSE)")
            fig = px.bar(comp, x="Model", y="RMSE", color="RMSE", color_continuous_scale=["#eab308", "#22c55e"])
            fig.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.subheader("Leaderboard")
            st.dataframe(comp, use_container_width=True)
    else:
        st.warning("Please run `python models/forecasting.py` to generate the model comparison data.")

# ── Inventory ─────────────────────────────────────────────────────────────────
elif page == "Inventory":
    st.title("Inventory Optimisation")
    
    if not inv.empty:
        c1, c2, c3 = st.columns(3)
        avg_safety = inv["safety_stock"].mean()
        avg_eoq = inv["eoq"].mean()
        total_cost = inv["total_inv_cost_gbp"].sum()
        
        c1.markdown(f"""<div class="metric-card">
            <div class="label">Avg Safety Stock</div>
            <div class="value">{avg_safety:.0f} units</div>
        </div>""", unsafe_allow_html=True)
        c2.markdown(f"""<div class="metric-card">
            <div class="label">Avg EOQ</div>
            <div class="value">{avg_eoq:.0f} units</div>
        </div>""", unsafe_allow_html=True)
        c3.markdown(f"""<div class="metric-card dark">
            <div class="label">Total Annual Holding Cost</div>
            <div class="value">£{total_cost:,.0f}</div>
        </div>""", unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.subheader("Inventory Recommendations by Product")
        
        fig = px.scatter(inv, x="safety_stock", y="eoq", size="total_inv_cost_gbp", color="total_inv_cost_gbp",
                         hover_name="product_name", color_continuous_scale=["#eab308", "#22c55e", "#111111"],
                         labels={"safety_stock": "Safety Stock", "eoq": "Economic Order Quantity (EOQ)"})
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(inv[["product_name", "safety_stock", "reorder_point", "eoq", "total_inv_cost_gbp"]], use_container_width=True)
    else:
        st.warning("Please run `python models/inventory_optimizer.py` to generate inventory recommendations.")
