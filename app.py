# ═══════════════════════════════════════════════════════════════════════════════
#  PUMA Algeria — AI-Driven Customer Intelligence & Sales Forecasting Dashboard
#  Master's Thesis | SARL Great Way | Statistics & Data Science
#
#  Launch:  streamlit run app.py
#  Requires: python train_models.py (run once first)
# ═══════════════════════════════════════════════════════════════════════════════

import warnings
warnings.filterwarnings("ignore")

import base64
import json
import pickle
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np

# Increase Pandas Styler limit for large RFM tables
pd.set_option("styler.render.max_elements", 1000000)
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import config as cfg


# ─── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PUMA Algeria — CRM Intelligence",
    page_icon="🐆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── COLOUR PALETTES ──────────────────────────────────────────────────────────
SEGMENT_COLORS_5 = cfg.SEGMENT_COLORS_5
KMEANS_COLORS    = cfg.KMEANS_COLORS
TIER_COLORS      = cfg.TIER_COLORS
MODEL_COLORS     = cfg.MODEL_COLORS


# ─── GLOBAL CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;600;700;800&family=Barlow:wght@300;400;500;600&display=swap');
  html, body, [class*="css"] {
    font-family:'Barlow',sans-serif;
    background-color:#0D0D0D;
    color:#F5F5F5;
  }
  [data-testid="stSidebar"] {
    background:linear-gradient(180deg,#1A1A1A 0%,#0D0D0D 100%);
    border-right:2px solid #FF0000;
  }
  [data-testid="stSidebar"] * { color:#F5F5F5 !important; }
  .main .block-container { background:#0D0D0D; padding-top:1rem; }
  [data-testid="metric-container"] {
    background:#1A1A1A; border:1px solid #2A2A2A;
    border-left:3px solid #FF0000; border-radius:4px; padding:12px 16px;
  }
  [data-testid="metric-container"] label {
    color:#9A9A9A !important; font-size:11px !important;
    letter-spacing:1px; text-transform:uppercase;
  }
  [data-testid="metric-container"] [data-testid="stMetricValue"] {
    color:#F5F5F5 !important; font-family:'Barlow Condensed',sans-serif;
    font-size:20px !important; font-weight:700;
  }
  [data-testid="metric-container"] [data-testid="stMetricValue"] > div {
    overflow: visible !important;
    white-space: normal !important;
  }
  [data-testid="metric-container"] [data-testid="stMetricDelta"] {
    color:#2ecc71 !important;
  }
  h1,h2,h3 { font-family:'Barlow Condensed',sans-serif !important; letter-spacing:1px; }
  h1 { color:#FF0000 !important; font-weight:800 !important; font-size:2.2rem !important; }
  h2 { color:#F5F5F5 !important; font-weight:700 !important; border-bottom:1px solid #2A2A2A; padding-bottom:6px; }
  h3 { color:#CCCCCC !important; font-weight:600 !important; }
  [data-testid="stTabs"] button {
    font-family:'Barlow Condensed',sans-serif !important;
    font-weight:600 !important; font-size:13px !important;
    letter-spacing:1px; text-transform:uppercase;
    color:#9A9A9A !important; border:none !important; background:transparent !important;
  }
  [data-testid="stTabs"] button[aria-selected="true"] {
    color:#FF0000 !important; border-bottom:2px solid #FF0000 !important;
  }
  [data-baseweb="select"] * { background:#1A1A1A !important; color:#F5F5F5 !important; border-color:#2A2A2A !important; }
  [data-testid="stMultiSelect"] * { background:#1A1A1A !important; color:#F5F5F5 !important; }
  .stSlider * { color:#F5F5F5 !important; }
  [data-testid="stDateInput"] * { background:#1A1A1A !important; color:#F5F5F5 !important; }
  [data-testid="stDataFrame"] { border:1px solid #2A2A2A !important; border-radius:4px; }
  .section-header {
    background:linear-gradient(90deg,#FF0000 0%,transparent 100%);
    height:2px; margin:20px 0 16px 0;
  }
  .insight-card {
    background:#1A1A1A; border:1px solid #2A2A2A;
    border-left:3px solid #FF0000; border-radius:4px;
    padding:14px 18px; margin-bottom:10px;
    font-size:13px; line-height:1.6;
  }
  .insight-card strong { color:#FF0000; }
  .badge {
    display:inline-block; background:#FF0000; color:white;
    font-family:'Barlow Condensed',sans-serif; font-size:10px;
    font-weight:700; letter-spacing:1px; padding:2px 8px;
    border-radius:2px; text-transform:uppercase; margin-right:6px;
  }
  #MainMenu,footer { visibility:hidden; }
  .stDeployButton { display:none; }
  ::-webkit-scrollbar { width:6px; }
  ::-webkit-scrollbar-track { background:#1A1A1A; }
  ::-webkit-scrollbar-thumb { background:#FF0000; border-radius:3px; }
  [data-testid="stExpander"] { background:#1A1A1A !important; border:1px solid #2A2A2A !important; border-radius:4px !important; }
  /* ── Multiselect dropdown: cap height & scroll internally ── */
  [data-baseweb="popover"] ul[role="listbox"] {
    max-height: 200px !important;
    overflow-y: auto !important;
    scrollbar-width: thin;
    scrollbar-color: #FF0000 #1A1A1A;
  }
  [data-baseweb="popover"] ul[role="listbox"]::-webkit-scrollbar { width: 5px; }
  [data-baseweb="popover"] ul[role="listbox"]::-webkit-scrollbar-track { background: #1A1A1A; }
  [data-baseweb="popover"] ul[role="listbox"]::-webkit-scrollbar-thumb { background: #FF0000; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ─── HELPERS ───────────────────────────────────────────────────────────────────

def fmt(value, suffix=" DZD"):
    """Format number with thousand separators."""
    return f"{value:,.0f}{suffix}"


def fmt_currency(value, suffix=" DZD"):
    """Human-readable currency: ≥1 M → 'X.X M DZD', ≥1 K → 'X.X K DZD', else raw."""
    abs_val = abs(value)
    if abs_val >= 1_000_000:
        return f"{value / 1_000_000:.2f} M{suffix}"
    if abs_val >= 1_000:
        return f"{value / 1_000:.1f} K{suffix}"
    return f"{value:,.0f}{suffix}"


def hex_to_rgba(hex_color: str, alpha: float = 0.2) -> str:
    """Convert hex colour to rgba string."""
    h = hex_color.lstrip('#')
    if len(h) == 6:
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"
    return f"rgba(136,136,136,{alpha})"


def puma_theme(title='', height=None):
    """Consistent dark theme for all Plotly charts."""
    layout = dict(
        paper_bgcolor='#1A1A1A',
        plot_bgcolor='#111111',
        font=dict(family='Barlow, sans-serif', color='#CCCCCC', size=12),
        title=dict(text=title,
                   font=dict(family='Barlow Condensed, sans-serif',
                             color='#F5F5F5', size=15)),
        xaxis=dict(gridcolor='#222222', linecolor='#333333', tickcolor='#555555'),
        yaxis=dict(gridcolor='#222222', linecolor='#333333', tickcolor='#555555'),
        legend=dict(bgcolor='#1A1A1A', bordercolor='#333333', borderwidth=1),
        margin=dict(l=60, r=20, t=50, b=40),
        colorway=['#FF0000', '#3498db', '#2ecc71', '#f39c12', '#e67e22', '#9b59b6'],
    )
    if height:
        layout['height'] = height
    return layout


# ══ Data Loading — three-tier strategy ═══════════════════════════════════════
#
#  Priority 1: st.session_state  → user just uploaded a file in this session
#  Priority 2: on-disk artifacts → local dev (run train_models.py once first)
#  Priority 3: show upload screen and stop
#
# This lets the same app.py work both locally (fast, disk-based) and on
# Streamlit Community Cloud (no data in repo → upload required).
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def _load_disk_artifacts(bust_cache=None):
    """Load pre-computed artifacts from disk. Returns dict or None."""
    required = [
        cfg.DF_CLEAN, cfg.SALES_CLEAN, cfg.RFM_SCORED, cfg.RFM_CLUSTERED,
        cfg.CLV_TABLE, cfg.DAILY_SALES, cfg.FORECAST_TEST, cfg.KMEANS_META,
        cfg.CLV_META, cfg.FORECAST_META, cfg.TRAINING_LOG,
        cfg.SARIMA_FIT, cfg.PROPHET_MODEL, cfg.XGB_FORECAST,
        cfg.CLV_VALIDATION, cfg.BGF_MODEL,
    ]
    if any(not p.exists() for p in required):
        return None

    art = {}
    art['df_clean']      = pd.read_parquet(cfg.DF_CLEAN)
    art['sales_clean']   = pd.read_parquet(cfg.SALES_CLEAN)
    art['rfm_scored']    = pd.read_parquet(cfg.RFM_SCORED)
    art['rfm_clustered'] = pd.read_parquet(cfg.RFM_CLUSTERED)
    art['clv_table']     = pd.read_parquet(cfg.CLV_TABLE)
    art['daily_sales']   = pd.read_parquet(cfg.DAILY_SALES)
    art['forecast_test'] = pd.read_parquet(cfg.FORECAST_TEST)
    art['clv_validation'] = pd.read_parquet(cfg.CLV_VALIDATION)
    with open(cfg.KMEANS_META)   as f: art['kmeans_meta']   = json.load(f)
    with open(cfg.CLV_META)      as f: art['clv_meta']      = json.load(f)
    with open(cfg.FORECAST_META) as f: art['forecast_meta'] = json.load(f)
    with open(cfg.TRAINING_LOG)  as f: art['training_log']  = json.load(f)
    art['bgf_model']     = pickle.load(open(cfg.BGF_MODEL,     'rb'))
    art['sarima_fit']    = pickle.load(open(cfg.SARIMA_FIT,    'rb'))
    art['prophet_model'] = pickle.load(open(cfg.PROPHET_MODEL, 'rb'))
    art['xgb_model']     = pickle.load(open(cfg.XGB_FORECAST,  'rb'))
    art['rf_model']      = (pickle.load(open(cfg.RF_FORECAST,  'rb'))
                            if cfg.RF_FORECAST.exists() else None)
    return art


def _show_upload_screen():
    """Render the data-upload landing page."""
    st.markdown("""
    <div style="text-align:center;padding:50px 20px 30px 20px;">
      <div style="font-size:56px;font-weight:900;letter-spacing:12px;
                  color:#FFFFFF;text-shadow:0 0 24px rgba(255,0,0,0.6);">
        PUMA
      </div>
      <div style="width:60px;height:3px;background:#FF0000;
                  margin:12px auto;border-radius:2px;"></div>
      <div style="font-size:11px;letter-spacing:5px;color:#888;
                  text-transform:uppercase;margin-bottom:28px;">
        Algeria &middot; AI Customer Intelligence
      </div>
      <h2 style="color:#F5F5F5;font-weight:600;">Upload your sales data to launch the dashboard</h2>
      <p style="color:#9A9A9A;font-size:1rem;max-width:560px;margin:16px auto 0 auto;">
        Drop the <strong style="color:#F5F5F5;">VENTES PUMA 2025.xlsx</strong> ERP export below.
        The full analytics pipeline — RFM segmentation, CLV modelling, and demand forecasting —
        will run automatically. Your file is never stored anywhere.
      </p>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        uploaded = st.file_uploader(
            "Select Excel file",
            type=['xlsx', 'xls'],
            label_visibility="collapsed",
            help="Raw ERP export from SARL GREAT WAY — stays private, never persisted.",
        )

        if uploaded is not None:
            st.markdown("---")
            st.info("📊 File received — running the full analytics pipeline. "
                    "This takes **3–8 minutes** on first run. Please wait.")

            progress_bar = st.progress(0.0)
            status_txt   = st.empty()

            def _on_step(label, pct):
                progress_bar.progress(min(pct, 1.0))
                status_txt.markdown(f"⚙️ &nbsp; **{label}**")

            try:
                import pipeline as pl
                art = pl.run_full_pipeline(uploaded.read(), on_step=_on_step)
                st.session_state['_puma_artifacts'] = art
                status_txt.markdown("✅ &nbsp; **Pipeline complete — loading dashboard…**")
                progress_bar.progress(1.0)
                st.rerun()
            except Exception as exc:
                st.error(f"Pipeline failed: {exc}")
                st.exception(exc)

        st.markdown("""
        <div style="margin-top:32px;padding:16px;background:#1A1A1A;
                    border:1px solid #2A2A2A;border-radius:6px;
                    color:#666;font-size:0.85rem;text-align:center;">
          🔒 &nbsp;Your data is processed in memory only and is never saved to disk or shared.
        </div>
        """, unsafe_allow_html=True)


# ── Resolve artifacts ────────────────────────────────────────────────────────
if '_puma_artifacts' in st.session_state:
    # Priority 1 — uploaded this session
    artifacts = st.session_state['_puma_artifacts']
else:
    # Priority 2 — on-disk (local dev)
    artifacts = _load_disk_artifacts(bust_cache="Refresh_v15_RestoreColors")

if artifacts is None:
    # Priority 3 — show upload screen
    _show_upload_screen()
    st.stop()

# ── Unpack ───────────────────────────────────────────────────────────────────
df_clean      = artifacts['df_clean']
sales_all     = artifacts['sales_clean']
rfm_scored    = artifacts['rfm_scored']
rfm_clustered = artifacts['rfm_clustered']
clv_table     = artifacts['clv_table']
daily_sales   = artifacts['daily_sales']
forecast_test = artifacts['forecast_test']
kmeans_meta   = artifacts['kmeans_meta']
clv_meta      = artifacts['clv_meta']
forecast_meta = artifacts['forecast_meta']
training_log  = artifacts['training_log']
sarima_fit    = artifacts['sarima_fit']
prophet_model = artifacts['prophet_model']
xgb_model     = artifacts['xgb_model']
rf_model      = artifacts.get('rf_model')
bgf_model     = artifacts['bgf_model']

# Ensure date columns are proper types
if 'Date_Only' in sales_all.columns:
    sales_all['Date_Only'] = pd.to_datetime(sales_all['Date_Only']).dt.date
if 'Date' in sales_all.columns:
    sales_all['Date'] = pd.to_datetime(sales_all['Date'], errors='coerce')


# ══ SIDEBAR ══════════════════════════════════════════════════════════════════

_PUMA_LOGO_B64_PLACEHOLDER = (
    "iVBORw0KGgoAAAANSUhEUgAAARMAAAC3CAMAAAAGjUrGAAAAjVBMVEXs7Ozt7e0AAADg4ODu7u7h4eHj4+Pl5eXb29vo"
    "6Ojq6urf39/Y2NjV1dXy8vLR0dGQkJDGxsZkZGQxMTF5eXmIiIj29vbAwMCpqamjo6MsLCxaWlrMzMx1dXV/f3+e"
    "np60tLRQUFAbGxsMDAyWlpY4ODhra2tBQUFAQEAmJiYWFhY2NjaFhYVMTExeXl7ZWuMDAAARzklEQVR4nO1dCXvb"
    "qBYFs2hBxkk6T07bJE3bdDqdmfb//7zHIomLABnZsiN3QvOdgparo2t0D6AFhKoCE4QFVn+I4aLCDGGG1Z8q8apb"
    "RzDimNfqf50luAYlgZUJtZMpqf0Cg3aFMWHNm1JnojM/GBTaoC3Z4xqDHA1W7MGseWIM6oPaLQvD1zc/lNBgvuId"
    "C9aZ7w1ibRChlE9I5xNx0CeWZn2cT8SET0i2T6zB032ClvKJOMUnU/XkKn0icnxSHF1P4tdOAcwT6HLrE5F0+Tyf"
    "cOMFoUlXlS7x/hRq56FCl7ixo45hSmoF73xidzKrrQnWbalL2P5Zn9gtmaY5mOddiXeOHMwbFoN5orcczA98seNr"
    "XG5K1gQHpe642uWOL/L49vFEJaz/9DGwK5F0SZ0p2M+WcL8lgyV/nSINSsoL2CshwAIfZnGIL+DksxBeKcLXJSxg"
    "GZN0CU2UrE8SW3JYqqOlfBYztvRZePuhKb5vPgn5LuWTqS1/L5+IiT0n9httyWb7JJvT5DqR65PRlooWSqepdcum"
    "/CMdy3cd5/mW3tJbekv/1VRXb8lPNSoZIYxtfVT/TJ5YbDQKg2UZoNDYQCQa9a7HIRvyzKPWr21Yb78j1WFPivXU"
    "LB1HkI2pMQIO1edLxChWiRgEeQryqv2pkupMqlRZrDQWFmuHqn/ZI2V4MENJKu8OS8Fhuzwk5SEeMKRWZFCLHAqQ"
    "4l3P1fYVbVuQjPO2TW5bmlWhFxQWdbfadq4xrx0yNBiIoAjyLLFlSMQiAxhSq8bUiklqIjggO6NPEp65Ap/Yikhc"
    "dYR5ROIV1Lt2QDUFFdQzljB/IJ9ae+q145kPkaGy0KkxWAYIlnOhkRkUBCAzm3CAU2Zm5yeRp6gxgBnU4AGF8gnn"
    "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"
)

with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:28px 8px 20px 8px;">
      <div style="font-size:42px;font-weight:900;letter-spacing:10px;
                  color:#FFFFFF;text-transform:uppercase;
                  text-shadow:0 0 18px rgba(255,0,0,0.5);">
        PUMA
      </div>
      <div style="width:50px;height:3px;background:#FF0000;
                  margin:10px auto 10px auto;border-radius:2px;"></div>
      <div style="font-size:9px;letter-spacing:4px;color:#888888;
                  text-transform:uppercase;">
        Algeria &middot; CRM Intelligence
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### Reporting Period")
    
    # Meta Ads-style Date Presets
    date_preset = st.selectbox(
        "Date Preset",
        options=[
            "Lifetime",
            "This Month",
            "Last 7 Days", 
            "Last 30 Days", 
            "Last Month",
            "Custom Range"
        ],
        index=0
    )

    _hi = sales_all['Date_Only'].max()
    _lo = sales_all['Date_Only'].min()
    
    # Calculate dates based on preset
    def get_dates(preset, start_all, end_all):
        if preset == "Last 7 Days":
            return max(start_all, end_all - pd.Timedelta(days=7)), end_all
        elif preset == "Last 30 Days":
            return max(start_all, end_all - pd.Timedelta(days=30)), end_all
        elif preset == "This Month":
            return end_all.replace(day=1), end_all
        elif preset == "Last Month":
            last_month_end = end_all.replace(day=1) - pd.Timedelta(days=1)
            last_month_start = last_month_end.replace(day=1)
            return max(start_all, last_month_start), min(end_all, last_month_end)
        elif preset == "Lifetime":
            return start_all, end_all
        return None, None

    calc_start, calc_end = get_dates(date_preset, _lo, _hi)
    
    if date_preset == "Custom Range":
        date_range = st.date_input("Custom Range Picker", value=(_lo, _hi),
                                   min_value=_lo, max_value=_hi)
    else:
        date_range = (calc_start, calc_end)
        st.info(f"Period: **{calc_start}** to **{calc_end}**")

    st.markdown("---")
    
    all_stores = sorted(sales_all['Store'].dropna().unique().tolist()) if 'Store' in sales_all.columns else []
    store_filter = st.multiselect("Store Selector", options=all_stores,
                                  default=[], placeholder="All stores")
    
    # Client Type Filter
    all_types = sorted(sales_all['Client_Type'].dropna().unique().tolist()) if 'Client_Type' in sales_all.columns else []
    type_filter = st.multiselect("Client Type Filter", options=all_types,
                                 default=[], placeholder="All types")

    forecast_days = st.slider("Forecast Horizon (days)", 7, 90, 30)
    st.caption("ℹ️ Applies to the 📈 Sales Forecasting tab only.")

    st.markdown("---")
    trained_at = training_log.get('trained_at', 'Unknown')[:19]
    st.caption(f"**Models trained on:** {trained_at}")
    st.caption(f"**CLV:** BG/NBD + Gamma-Gamma (52 weeks)")
    st.caption(f"**Forecast:** SARIMA + Prophet + XGBoost + RF")


# ── Apply filters ────────────────────────────────────────────────────────────
sales = sales_all.copy()
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    sales = sales[(sales['Date_Only'] >= date_range[0]) &
                  (sales['Date_Only'] <= date_range[1])]
if store_filter and 'Store' in sales.columns:
    sales = sales[sales['Store'].isin(store_filter)]
if type_filter and 'Client_Type' in sales.columns:
    sales = sales[sales['Client_Type'].isin(type_filter)]


# ══ HEADER + KPIs ════════════════════════════════════════════════════════════

filter_label = f" | {', '.join(store_filter)}" if store_filter else ""
active_days  = sales['Date_Only'].nunique() if len(sales) else 0

st.markdown(f"""
<div style="padding:8px 0 16px 0;">
  <div style="font-family:'Barlow Condensed',sans-serif;font-size:11px;letter-spacing:3px;
              color:#FF0000;text-transform:uppercase;margin-bottom:4px;">
    <span class="badge">Live</span> SARL GREAT WAY &middot; MASTER'S THESIS{filter_label}
  </div>
  <h1 style="margin:0;font-size:2.2rem;">AI-Driven Customer Intelligence &amp; Sales Forecasting</h1>
  <div style="font-size:13px;color:#AAAAAA;margin-top:4px;">
    PUMA Algeria &nbsp;&middot;&nbsp; {len(sales):,} transactions &nbsp;&middot;&nbsp; {active_days} active days
  </div>
</div>""", unsafe_allow_html=True)

rev_total  = sales['Revenue'].sum()
n_trans    = len(sales)
n_cust     = sales['Client'].nunique() if 'Client' in sales.columns else 0
avg_basket = rev_total / n_trans if n_trans else 0
ret_rate   = df_clean['Is_Return'].mean() * 100
n_stores   = sales['Store'].nunique() if 'Store' in sales.columns else 0

# Row 1: Primary Value Drivers
k1, k2, k3 = st.columns(3)
k1.metric("Total Revenue",       fmt(rev_total),        delta="Period Total",    delta_color="off")
k2.metric("Total Transactions",  f"{n_trans:,}",         delta="All ticket types", delta_color="off")
k3.metric("Unique Customers",    f"{n_cust:,}",          delta="Active clients",  delta_color="off")

# Row 2: Performance Metrics
k4, k5, k6 = st.columns(3)
k4.metric("Average Basket",      fmt(avg_basket),        delta="Per transaction", delta_color="off")
k5.metric("Return Rate",         f"{ret_rate:.1f}%",     delta="Of all rows",     delta_color="off")
k6.metric("Active Stores",       f"{n_stores}",           delta="Locations",       delta_color="off")

st.markdown("<div class='section-header'></div>", unsafe_allow_html=True)


# ══ TAB DEFINITIONS ══════════════════════════════════════════════════════════

tabs = st.tabs([
    "📊 EDA & Overview",
    "🎯 RFM Analysis",
    "🤖 K-Means Clustering",
    "💰 CLV (BG/NBD)",
    "📈 Sales Forecasting",
    "👥 Clients",
    "📖 Documentation",
])


# ══ Tab 1 — EDA & Overview ═══════════════════════════════════════════════════

def render_eda(sales):
    """Render EDA tab with 7 charts."""
    st.markdown("## Exploratory Data Analysis")

    c1, c2 = st.columns([3, 2])
    with c1:
        if 'Store' in sales.columns:
            sr = sales.groupby('Store')['Revenue'].sum().sort_values(
                ascending=False).reset_index()
            fig = go.Figure(go.Bar(
                x=sr['Store'], y=sr['Revenue'],
                marker=dict(color=sr['Revenue'],
                            colorscale=[[0, '#3A0000'], [0.5, '#e74c3c'], [1, '#FF0000']]),
                text=[fmt(v) for v in sr['Revenue']],
                textposition='outside', textfont=dict(size=9, color='#CCCCCC'),
            ))
            fig.update_layout(**puma_theme("Total Revenue by Store"),
                              xaxis_tickangle=-35, showlegend=False, height=370)
            fig.update_yaxes(tickformat=",")
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        if 'Category' in sales.columns:
            cr = sales.groupby('Category')['Revenue'].sum().sort_values().reset_index()
            pal = ['#e74c3c', '#e67e22', '#f39c12', '#2ecc71', '#3498db', '#9b59b6']
            fig = go.Figure(go.Bar(
                y=cr['Category'], x=cr['Revenue'], orientation='h',
                marker=dict(color=pal[:len(cr)]),
                text=[fmt(v) for v in cr['Revenue']],
                textposition='outside', textfont=dict(size=9, color='#CCCCCC'),
            ))
            fig.update_layout(**puma_theme("Revenue by Category"),
                              showlegend=False, height=370)
            fig.update_xaxes(tickformat=",")
            st.plotly_chart(fig, use_container_width=True)

    # Monthly trend
    mo = sales.groupby(['Year', 'Month'])['Revenue'].sum().reset_index()
    mo['Period'] = pd.to_datetime(mo[['Year', 'Month']].assign(day=1))
    mo.sort_values('Period', inplace=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=mo['Period'], y=mo['Revenue'], mode='lines+markers',
        line=dict(color='#2ecc71', width=2.5),
        marker=dict(size=5, color='#2ecc71'),
        fill='tozeroy', fillcolor='rgba(46,204,113,0.07)', name='Monthly Revenue',
        hovertemplate='%{x|%b %Y}<br>Revenue: %{y:,.0f} DZD<extra></extra>',
    ))
    fig.update_layout(**puma_theme("Monthly Revenue Trend"),
                      yaxis_title="Revenue (DZD)", height=280, showlegend=False)
    fig.update_yaxes(tickformat=",")
    st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns([3, 2])
    with c3:
        if 'Product' in sales.columns and 'Qty' in sales.columns:
            t15 = (sales.groupby('Product')['Qty'].sum()
                   .nlargest(15).sort_values().reset_index())
            fig = go.Figure(go.Bar(
                y=t15['Product'], x=t15['Qty'], orientation='h',
                marker=dict(color='#e74c3c',
                            opacity=np.linspace(0.4, 1.0, len(t15)).tolist()),
                text=[f"{v:,}" for v in t15['Qty']], textposition='outside',
                textfont=dict(size=9, color='#CCCCCC'),
            ))
            fig.update_layout(**puma_theme("Top 15 Best-Selling Products (Units)",
                                          height=400), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with c4:
        if 'Day_Name' in sales.columns and 'Hour' in sales.columns:
            day_order = ["Monday", "Tuesday", "Wednesday", "Thursday",
                         "Friday", "Saturday", "Sunday"]
            avail = [d for d in day_order if d in sales['Day_Name'].unique()]
            hrly = sales.groupby(['Day_Name', 'Hour'])['Revenue'].sum().reset_index()
            if not hrly.empty:
                pivot = (hrly.pivot(index='Day_Name', columns='Hour', values='Revenue')
                         .fillna(0)
                         .reindex([d for d in avail if d in hrly['Day_Name'].unique()]))
                fig = go.Figure(go.Heatmap(
                    z=pivot.values,
                    x=[f"{h}h" for h in pivot.columns],
                    y=pivot.index.tolist(),
                    colorscale=[[0, '#0D0D0D'], [0.5, '#660000'], [1, '#FF0000']],
                    showscale=True,
                    hovertemplate='%{y} %{x}<br>Revenue: %{z:,.0f} DZD<extra></extra>',
                    colorbar=dict(title=dict(text='DZD', font=dict(color='#999')),
                                  tickfont=dict(color='#999'), tickformat=','),
                ))
                fig.update_layout(**puma_theme("Revenue Heatmap (Day x Hour)",
                                              height=400))
                st.plotly_chart(fig, use_container_width=True)

    # Top 15 products by revenue
    st.markdown("### Top 15 Products by Revenue")
    if 'Product' in sales.columns:
        t15r = (sales.groupby('Product')['Revenue']
                .sum().nlargest(15).sort_values().reset_index())
        pal_rev = np.linspace(0.35, 1.0, len(t15r)).tolist()
        fig = go.Figure(go.Bar(
            y=t15r['Product'], x=t15r['Revenue'], orientation='h',
            marker=dict(color='#f39c12', opacity=pal_rev),
            text=[fmt(v) for v in t15r['Revenue']],
            textposition='outside',
            textfont=dict(size=9, color='#CCCCCC'),
        ))
        fig.update_layout(**puma_theme('Top 15 Products by Revenue (DZD)', height=430),
                          showlegend=False)
        fig.update_xaxes(tickformat=',')
        st.plotly_chart(fig, use_container_width=True)

    c5, c6 = st.columns(2)
    with c5:
        ct = sales.groupby('Client_Type')['Revenue'].sum().reset_index()
        ct.columns = ['Type', 'Revenue']
        ct = ct.sort_values('Revenue', ascending=False)
        total_rev = ct['Revenue'].sum()
        ct['pct'] = ct['Revenue'] / total_rev * 100
        ct['text'] = ct['pct'].apply(lambda p: f"{p:.1f}%" if p >= 5 else "")
        fig = go.Figure(go.Pie(
            labels=ct['Type'], values=ct['Revenue'], hole=0.55,
            text=ct['text'], textinfo='text', textposition='inside',
            marker=dict(colors=['#e74c3c', '#e67e22', '#3498db', '#2ecc71'],
                        line=dict(color='#0D0D0D', width=2)),
            textfont=dict(size=12, color='#FFFFFF'),
            hovertemplate='%{label}<br>Revenue: %{value:,.0f} DZD<br>Share: %{percent}<extra></extra>',
        ))
        fig.update_layout(**puma_theme("Revenue by Client Type", height=300))
        fig.update_layout(legend=dict(orientation='h', y=-0.15))
        st.plotly_chart(fig, use_container_width=True)

    with c6:
        if 'Season' in sales.columns:
            se = sales.groupby('Season')['Revenue'].sum().sort_values(ascending=False).reset_index()
            season_list = se['Season'].tolist()
            season_palette = ['#e67e22', '#3498db', '#2ecc71', '#9b59b6', '#f39c12', '#e74c3c']
            season_colors = [season_palette[i % len(season_palette)] for i in range(len(season_list))]
            fig = go.Figure(go.Bar(
                x=se['Season'], y=se['Revenue'],
                marker=dict(color=season_colors, opacity=0.85),
                text=[fmt(v) for v in se['Revenue']],
                textposition='outside', textfont=dict(size=9, color='#CCCCCC'),
            ))
            fig.update_layout(**puma_theme("Revenue by Season", height=300),
                              showlegend=False)
            fig.update_yaxes(tickformat=",")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='section-header'></div>", unsafe_allow_html=True)
    
    # ── Executive Presentation Summary ─────────────────────────────────────────
    st.markdown("### 🎓 Executive Strategy (Graduation Presentation Summary)")
    st.markdown("""
    <div style="background:linear-gradient(135deg, #111 0%, #1a1a1a 100%); 
                border-left: 4px solid #FF0000; padding: 20px; border-radius: 6px; 
                border-top: 1px solid #333; border-right: 1px solid #333; border-bottom: 1px solid #333;
                margin-top: 10px; margin-bottom: 30px;">
        <h4 style="color:#FF0000; margin-top:0; font-family:'Barlow Condensed', sans-serif; letter-spacing:1px;">
            MACRO-ECONOMIC BUSINESS INTELLIGENCE
        </h4>
        <p style="color:#E0E0E0; font-size:15px; line-height:1.7;">
            Exploratory Data Analysis (EDA) forms the foundational truth of PUMA's commercial performance. Before running complex predictive Machine Learning, we must mathematically identify the behavioral and seasonal patterns driving top-line revenue.
        </p>
        <ul style="color:#CCCCCC; font-size:14px; line-height:1.6; padding-left:20px;">
            <li><b style="color:#2ecc71;">Macro Trends:</b> PUMA's revenue shows extremely distinct seasonal waves. <i>Action: Cross-reference inventory warehousing cycles to ensure stock peaks exactly 3 weeks prior to historical demand spikes.</i></li>
            <li><b style="color:#3498db;">Product Hierarchy:</b> A small fraction of Top 15 absolute SKUs drives a disproportionate amount of our revenue. <i>Action: Guarantee supply chain priority for these core items to prevent massive opportunity cost via stock-outs.</i></li>
            <li><b style="color:#f39c12;">Temporal Heatmaps:</b> Transactions cluster heavily around specific days of the week and hours of the day. <i>Action: Optimize physical store staffing rotations and digital ad-spend schedules to directly match peak consumption hours.</i></li>
        </ul>
        <p style="color:#999; font-size:13px; font-style:italic; margin-bottom:0; border-top: 1px solid #333; padding-top: 12px; margin-top: 15px;">
            * Conclusion: EDA transitions our management team from "guessing" supply requirements to strategically pre-empting demand. We will now drill down from Global Sales into Individual Customer Behavior.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ══ Tab 2 — RFM Analysis ════════════════════════════════════════════════════

def render_rfm(rfm):
    """Render RFM tab."""
    st.markdown("## RFM Analysis")
    st.info("RFM scores computed at snapshot date (2026-01-02). "
            "Date filters apply to EDA tab only.")

    if rfm.empty:
        st.warning("No RFM data available.")
        return

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Customers", f"{len(rfm):,}")
    r2.metric("Champions",
              f"{(rfm['Segment'] == 'Champions').mean() * 100:.1f}%")
    r3.metric("At Risk",
              f"{(rfm['Segment'] == 'At Risk').mean() * 100:.1f}%")
    r4.metric("Avg RFM Score", f"{rfm['RFM_Score'].mean():.1f} / 15")
    st.markdown("<div class='section-header'></div>", unsafe_allow_html=True)

    # ── Descriptive statistics ──────────────────────────────────────────────
    st.markdown("### Descriptive Statistics — Recency · Frequency · Monetary")
    _desc = rfm[['Recency', 'Frequency', 'Monetary']].describe()
    
    metrics = ["Count", "Average (Mean)", "Volatility (Std Dev)", "Minimum", 
               "25th Percentile", "Median (50%)", "75th Percentile", "Maximum"]
    
    # Custom formatters
    recency_vals = [f"{v:,.0f}" if i == 0 else f"{v:,.1f} days" for i, v in enumerate(_desc['Recency'])]
    freq_vals = [f"{v:,.0f}" if i == 0 else f"{v:,.2f}" for i, v in enumerate(_desc['Frequency'])]
    mon_vals = [f"{v:,.0f}" if i == 0 else f"{v:,.0f} DZD" for i, v in enumerate(_desc['Monetary'])]
    
    fig_table = go.Figure(data=[go.Table(
        columnorder=[1, 2, 3, 4],
        columnwidth=[120, 100, 100, 120],
        header=dict(
            values=['<b>Statistic</b>', '<b>Recency</b>', '<b>Frequency</b>', '<b>Monetary</b>'],
            fill_color='#1A1A1A',
            font=dict(color='#FF0000', size=15, family="Barlow Condensed, sans-serif"),
            align=['left', 'center', 'center', 'right'],
            height=35,
            line_color='#2A2A2A',
        ),
        cells=dict(
            values=[metrics, recency_vals, freq_vals, mon_vals],
            fill_color=[['#0D0D0D', '#111111'] * 4],
            font=dict(color='#F5F5F5', size=13, family="Barlow, sans-serif"),
            align=['left', 'center', 'center', 'right'],
            height=32,
            line_color='#2A2A2A',
        )
    )])
    
    fig_table.update_layout(
        height=320,
        margin=dict(l=0, r=0, t=5, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_table, use_container_width=True)
    st.markdown("<div class='section-header'></div>", unsafe_allow_html=True)
    
    # ── RFM Distributions ────────────────────────────────────────────────────
    st.markdown("### RFM Distributions")
    d1, d2, d3 = st.columns(3)
    
    with d1:
        fig = go.Figure(go.Histogram(
            x=rfm['Recency'], nbinsx=30,
            marker_color='#e74c3c', opacity=0.85,
            hovertemplate='Recency: %{x} days<br>Count: %{y}<extra></extra>'
        ))
        fig.update_layout(**puma_theme("Recency Distribution", height=280))
        fig.update_xaxes(title="Days since last purchase")
        st.plotly_chart(fig, use_container_width=True)
        
    with d2:
        fig = go.Figure(go.Histogram(
            x=rfm['Frequency'], nbinsx=30,
            marker_color='#3498db', opacity=0.85,
            hovertemplate='Frequency: %{x}<br>Count: %{y}<extra></extra>'
        ))
        fig.update_layout(**puma_theme("Frequency Distribution", height=280))
        fig.update_xaxes(title="Number of transactions")
        st.plotly_chart(fig, use_container_width=True)
        
    with d3:
        # For monetary, clipping the top 1% makes the plot much more readable
        m_clip = rfm['Monetary'].quantile(0.99)
        fig = go.Figure(go.Histogram(
            x=rfm['Monetary'].clip(upper=m_clip), nbinsx=30,
            marker_color='#2ecc71', opacity=0.85,
            hovertemplate='Monetary: %{x:,.0f} DZD<br>Count: %{y}<extra></extra>'
        ))
        fig.update_layout(**puma_theme("Monetary Distribution (up to 99th pctl)", height=280))
        fig.update_xaxes(title="Total Revenue (DZD)", tickformat=",")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='section-header'></div>", unsafe_allow_html=True)

    sa1, sa2 = st.columns(2)
    with sa1:
        sc = rfm['Segment'].value_counts().reset_index()
        sc.columns = ['Segment', 'Count']
        fig = go.Figure(go.Pie(
            labels=sc['Segment'], values=sc['Count'], hole=0.55,
            marker=dict(
                colors=[SEGMENT_COLORS_5.get(s.strip() if hasattr(s, 'strip') else s, '#888') for s in sc['Segment']],
                line=dict(color='#0D0D0D', width=2)),
            hovertemplate='%{label}<br>Count: %{value:,}<br>%{percent}<extra></extra>',
        ))
        fig.update_layout(**puma_theme("RFM Segment Distribution", height=340))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Each slice = % of customers in that segment. Champions & Loyal = high-value retention targets.")

    with sa2:
        fig = go.Figure()
        for seg in cfg.SEG_ORDER_5:
            g = rfm[rfm['Segment'] == seg]['RFM_Score']
            if len(g):
                clr = SEGMENT_COLORS_5.get(seg, '#888')
                fig.add_trace(go.Box(
                    y=g, name=seg, marker_color=clr, line_color=clr,
                    fillcolor=hex_to_rgba(clr, 0.25)))
        fig.update_layout(**puma_theme("RFM Score Distribution by Segment", height=340))
        st.plotly_chart(fig, use_container_width=True)

    # Average Spend by Segment
    st.markdown("### Average Spend by Segment")
    avg_spend = rfm.groupby('Segment')['Monetary'].mean().reindex(cfg.SEG_ORDER_5).reset_index()
    fig = go.Figure(go.Bar(
        x=avg_spend['Segment'], y=avg_spend['Monetary'],
        marker=dict(color=[SEGMENT_COLORS_5.get(s, '#888') for s in avg_spend['Segment']],
                    opacity=0.85),
        text=[fmt(v) for v in avg_spend['Monetary']],
        textposition='outside', textfont=dict(size=10, color='#CCCCCC'),
    ))
    fig.update_layout(**puma_theme("Average Spend per Segment (DZD)", height=340),
                      showlegend=False)
    fig.update_yaxes(tickformat=",")
    st.plotly_chart(fig, use_container_width=True)

    # Scatter plots
    st.markdown("### Scatter Plots")
    sdf = rfm.sample(min(3000, len(rfm)), random_state=42)
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        fig = px.scatter(sdf, x='Recency', y='Frequency', color='Segment',
                         color_discrete_map=SEGMENT_COLORS_5, opacity=0.65,
                         template='plotly_dark')
        fig.update_layout(**puma_theme("Recency vs Frequency", height=340))
        st.plotly_chart(fig, use_container_width=True)
    with sc2:
        fig = px.scatter(sdf, x='Frequency', y='Monetary', color='Segment',
                         color_discrete_map=SEGMENT_COLORS_5, opacity=0.65,
                         template='plotly_dark')
        fig.update_layout(**puma_theme("Frequency vs Monetary", height=340))
        fig.update_yaxes(tickformat=",")
        st.plotly_chart(fig, use_container_width=True)
    with sc3:
        fig = px.scatter(sdf, x='Recency', y='Monetary', color='Segment',
                         color_discrete_map=SEGMENT_COLORS_5, opacity=0.65,
                         template='plotly_dark')
        fig.update_layout(**puma_theme("Recency vs Monetary", height=340))
        fig.update_yaxes(tickformat=",")
        st.plotly_chart(fig, use_container_width=True)

    # Snake plot
    st.markdown("### Snake Plot")
    snake = rfm.groupby('Segment')[['Recency', 'Frequency', 'Monetary']].mean()
    sn = (snake - snake.min()) / (snake.max() - snake.min() + 1e-9)
    fig = go.Figure()
    for seg in sn.index:
        fig.add_trace(go.Scatter(
            x=['Recency', 'Frequency', 'Monetary'],
            y=sn.loc[seg].tolist(),
            mode='lines+markers', name=seg,
            line=dict(color=SEGMENT_COLORS_5.get(seg, '#888'), width=2.5),
            marker=dict(size=10),
        ))
    fig.update_layout(**puma_theme("Segment Snake Plot (Normalized R, F, M)",
                                   height=340),
                      yaxis_title="Normalized (0-1)")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("👁️ How to read: Segments closer to **1.0** on Frequency & Monetary are your highest-value customers. Segments near **0** on Recency have the most recent purchases.")

    # ── 3D RFM Scatter ───────────────────────────────────────────────────────
    st.markdown("### 3D RFM Space (Recency · Frequency · Monetary)")
    _s3 = rfm.sample(min(3000, len(rfm)), random_state=42)
    fig3d = go.Figure()
    for _seg in cfg.SEG_ORDER_5:
        _g = _s3[_s3['Segment'] == _seg]
        if len(_g):
            _clr = SEGMENT_COLORS_5.get(_seg, '#888')
            fig3d.add_trace(go.Scatter3d(
                x=_g['Recency'],
                y=_g['Frequency'],
                z=_g['Monetary'],
                mode='markers',
                name=_seg,
                marker=dict(size=3, color=_clr, opacity=0.75),
                hovertemplate=(
                    f'<b>{_seg}</b><br>'
                    'Recency: %{x:.0f} days<br>'
                    'Frequency: %{y:.0f}<br>'
                    'Monetary: %{z:,.0f} DZD'
                    '<extra></extra>'
                ),
            ))
    fig3d.update_layout(
        paper_bgcolor='#1A1A1A',
        scene=dict(
            bgcolor='#111111',
            xaxis=dict(title='Recency (days)', gridcolor='#2A2A2A',
                       linecolor='#333', color='#AAA'),
            yaxis=dict(title='Frequency',      gridcolor='#2A2A2A',
                       linecolor='#333', color='#AAA'),
            zaxis=dict(title='Monetary (DZD)', gridcolor='#2A2A2A',
                       linecolor='#333', color='#AAA'),
        ),
        font=dict(family='Barlow, sans-serif', color='#CCCCCC'),
        title=dict(
            text='3D RFM Space — Coloured by Segment',
            font=dict(family='Barlow Condensed, sans-serif',
                      color='#F5F5F5', size=15),
        ),
        legend=dict(bgcolor='#1A1A1A', bordercolor='#333', borderwidth=1),
        height=560,
        margin=dict(l=0, r=0, t=50, b=0),
    )
    st.plotly_chart(fig3d, use_container_width=True)
    st.caption("🔄 Rotate the chart by dragging. Each dot = 1 customer. Tightly packed cluster = homogeneous behaviour. Dispersed = high within-segment variance.")

    # Segment summary table
    st.markdown("### Segment Summary")
    summ = rfm.groupby('Segment').agg(
        Customers     = ('Client', 'count'),
        Avg_Recency   = ('Recency', 'mean'),
        Avg_Frequency = ('Frequency', 'mean'),
        Avg_Monetary  = ('Monetary', 'mean'),
        Total_Revenue = ('Monetary', 'sum'),
    ).round(1).reset_index()
    summ['Revenue_Share'] = (summ['Total_Revenue'] /
                              summ['Total_Revenue'].sum() * 100).round(1)
    st.dataframe(
        summ.style.background_gradient(subset=['Total_Revenue', 'Avg_Monetary'],
                                       cmap='Reds')
        .format({'Avg_Recency': '{:.0f} days', 'Avg_Frequency': '{:.1f}',
                 'Avg_Monetary': '{:,.0f} DZD', 'Total_Revenue': '{:,.0f} DZD',
                 'Revenue_Share': '{:.1f}%'}),
        use_container_width=True)

    st.markdown("<div class='section-header'></div>", unsafe_allow_html=True)
    
    # ── Full RFM Table with Filters ──────────────────────────────────────────
    st.markdown("### Customer RFM Detailed Table")
    
    # Filter by segment
    all_segs = cfg.SEG_ORDER_5
    selected_segs = st.multiselect("Filter by Segment", options=all_segs, default=all_segs)
    
    # Filter data
    rfm_filtered = rfm[rfm['Segment'].isin(selected_segs)].copy()
    
    # Sort and refine columns
    cols_to_show = ['Client', 'Recency', 'Frequency', 'Monetary', 
                    'R_Score', 'F_Score', 'M_Score', 'RFM_Score', 'Segment']
    
    def _style_segment(val):
        color = SEGMENT_COLORS_5.get(str(val).strip(), '#888888')
        return f'background-color: {color}22; color: {color}; font-weight: 600'

    st.dataframe(
        rfm_filtered[cols_to_show].sort_values('RFM_Score', ascending=False)
        .style
        .background_gradient(subset=['RFM_Score'], cmap='RdYlGn')
        .map(_style_segment, subset=['Segment'])
        .format({'Monetary': '{:,.0f}', 'Recency': '{:,.0f}'}),
        use_container_width=True,
        height=450
    )
    st.caption(f"Showing {len(rfm_filtered):,} customers out of {len(rfm):,}")

    st.markdown("<div class='section-header'></div>", unsafe_allow_html=True)
    
    # ── Executive Presentation Summary ─────────────────────────────────────────
    st.markdown("### 🎓 Executive Strategy (Graduation Presentation Summary)")
    st.markdown("""
    <div style="background:linear-gradient(135deg, #111 0%, #1a1a1a 100%); 
                border-left: 4px solid #FF0000; padding: 20px; border-radius: 6px; 
                border-top: 1px solid #333; border-right: 1px solid #333; border-bottom: 1px solid #333;
                margin-top: 10px;">
        <h4 style="color:#FF0000; margin-top:0; font-family:'Barlow Condensed', sans-serif; letter-spacing:1px;">
            BUSINESS RULES HYPOTHESIS
        </h4>
        <p style="color:#E0E0E0; font-size:15px; line-height:1.7;">
            Using the classic <b>RFM (Recency, Frequency, Monetary)</b> framework, we enforced strict, human-defined heuristic rules using quintiles (20% splits) to categorize the entire customer base into 5 traditional segments:
        </p>
        <ul style="color:#CCCCCC; font-size:14px; line-height:1.6; padding-left:20px;">
            <li><b style="color:#2ecc71;">Champions (5-5-5):</b> Our best clients who bought recently, often, and spend heavily. <i>Action: Reward them. They are early adopters for new PUMA lines.</i></li>
            <li><b style="color:#3498db;">Loyal Customers:</b> Consistent buyers with healthy recency. <i>Action: Up-sell higher-margin products and ask for reviews.</i></li>
            <li><b style="color:#f1c40f;">New Customers:</b> High Recency, but low Frequency. They just made their first purchase. <i>Action: Provide onboarding support and first-time-buyer discounts to incentivize visit #2.</i></li>
            <li><b style="color:#e67e22;">At Risk:</b> Customers who used to visit often but haven't been seen recently. <i>Action: Send personalized reactivation emails or SMS with 'We Miss You' offers.</i></li>
            <li><b style="color:#e74c3c;">Lost:</b> Ancient customers (>365 days since last purchase). <i>Action: Ignore, or dedicate minimal resources to massive discount blasts.</i></li>
        </ul>
        <p style="color:#999; font-size:13px; font-style:italic; margin-bottom:0; border-top: 1px solid #333; padding-top: 12px; margin-top: 15px;">
            * Limitation: While easy to interpret, this hardcoded approach forces arbitrary cutoffs (e.g., someone with 60 days recency is 'New', but 61 days is 'At Risk'). We transition to K-Means next to solve this.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ══ Tab 3 — K-Means Clustering ═══════════════════════════════════════════════

def render_kmeans(rfm_km, meta):
    """Render K-Means tab with improved readability."""
    st.markdown("## K-Means Clustering")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Clusters (k)", str(meta['k']))
    m2.metric("Silhouette Score", f"{meta['silhouette']:.4f}")
    m3.metric("Davies-Bouldin", f"{meta['davies_bouldin']:.4f}")
    m4.metric("Customers", f"{len(rfm_km):,}")
    st.markdown("<div class='section-header'></div>", unsafe_allow_html=True)

    # Row 1: Segment distribution (full width)
    kmc = (rfm_km['KMeans_Segment'].value_counts()
           .reindex(cfg.SEG_ORDER_KM).reset_index())
    kmc.columns = ['Segment', 'Count']
    kmc['Pct'] = (kmc['Count'] / kmc['Count'].sum() * 100).round(1)
    fig = go.Figure(go.Bar(
        x=kmc['Segment'], y=kmc['Count'],
        marker_color=[KMEANS_COLORS.get(s, '#888') for s in kmc['Segment']],
        text=[f"{c:,} ({p:.0f}%)" for c, p in zip(kmc['Count'], kmc['Pct'])],
        textposition='outside', textfont=dict(size=11, color='#CCCCCC'),
    ))
    fig.update_layout(**puma_theme("Segment Distribution", height=360),
                      showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Row 2: Detailed per-metric profiles (3 separate bar charts — readable)
    st.markdown("### Cluster Metrics Breakdown")
    mc1, mc2, mc3 = st.columns(3)

    for col_ctx, metric, color, label in [
        (mc1, 'Recency',   '#3498db', 'Avg Recency (days)'),
        (mc2, 'Frequency', '#e67e22', 'Avg Frequency (visits)'),
        (mc3, 'Monetary',  '#2ecc71', 'Avg Monetary (DZD)'),
    ]:
        with col_ctx:
            data = rfm_km.groupby('KMeans_Segment')[metric].mean().reindex(cfg.SEG_ORDER_KM)
            if metric == 'Monetary':
                txt = [fmt(v) for v in data.values]
            elif metric == 'Recency':
                txt = [f"{v:.0f} days" for v in data.values]
            else:
                txt = [f"{v:.1f}" for v in data.values]
            fig = go.Figure(go.Bar(
                x=data.index, y=data.values,
                marker_color=[KMEANS_COLORS.get(s, '#888') for s in data.index],
                text=txt, textposition='outside',
                textfont=dict(size=10, color='#CCCCCC'),
            ))
            fig.update_layout(**puma_theme(label, height=280), showlegend=False)
            if metric == 'Monetary':
                fig.update_yaxes(tickformat=",")
            st.plotly_chart(fig, use_container_width=True)

    # Row 3: Revenue by Cluster
    kr = rfm_km.groupby('KMeans_Segment')['Monetary'].sum().sort_values().reset_index()
    fig = go.Figure(go.Bar(
        y=kr['KMeans_Segment'], x=kr['Monetary'], orientation='h',
        marker_color=[KMEANS_COLORS.get(s, '#888') for s in kr['KMeans_Segment']],
        text=[fmt(v) for v in kr['Monetary']],
        textposition='outside', textfont=dict(size=10, color='#CCCCCC'),
    ))
    fig.update_layout(**puma_theme("Total Revenue by Cluster",
                                   height=340), showlegend=False)
    fig.update_xaxes(tickformat=",")
    st.plotly_chart(fig, use_container_width=True)

    # Summary table
    st.markdown("### Segment Summary")
    ks = rfm_km.groupby('KMeans_Segment').agg(
        Customers     = ('Client', 'count'),
        Avg_Recency   = ('Recency', 'mean'),
        Avg_Frequency = ('Frequency', 'mean'),
        Avg_Monetary  = ('Monetary', 'mean'),
        Total_Revenue = ('Monetary', 'sum'),
    ).round(1).reset_index()
    ks['Revenue_Share'] = (ks['Total_Revenue'] /
                            ks['Total_Revenue'].sum() * 100).round(1)
    st.dataframe(
        ks.style.background_gradient(subset=['Total_Revenue'], cmap='Reds')
        .format({'Avg_Recency': '{:.0f} days', 'Avg_Frequency': '{:.1f}',
                 'Avg_Monetary': '{:,.0f} DZD', 'Total_Revenue': '{:,.0f} DZD',
                 'Revenue_Share': '{:.1f}%'}),
        use_container_width=True)

    st.markdown("<div class='section-header'></div>", unsafe_allow_html=True)
    
    # ── Full K-Means Table with Filters ──────────────────────────────────────
    st.markdown("### Customer K-Means Detailed Table")
    
    # Filter by KMeans Segment
    all_km_segs = cfg.SEG_ORDER_KM
    selected_km_segs = st.multiselect("Filter by Cluster", options=all_km_segs, default=all_km_segs)
    
    # Filter data
    rfm_km_filtered = rfm_km[rfm_km['KMeans_Segment'].isin(selected_km_segs)].copy()
    
    # Sort and refine columns
    # Note: Using KMeans_Segment instead of Segment
    cols_to_show_km = ['Client', 'Recency', 'Frequency', 'Monetary', 'KMeans_Segment']
    
    st.dataframe(
        rfm_km_filtered[cols_to_show_km].sort_values('Monetary', ascending=False)
        .style.background_gradient(subset=['Monetary'], cmap='Greens')
        .format({'Monetary': '{:,.0f}', 'Recency': '{:,.0f}'}),
        use_container_width=True,
        height=450
    )
    st.caption(f"Showing {len(rfm_km_filtered):,} customers out of {len(rfm_km):,}")

    st.markdown("<div class='section-header'></div>", unsafe_allow_html=True)

    # ── Executive Presentation Summary ─────────────────────────────────────────
    st.markdown("### 🎓 Executive Strategy (Graduation Presentation Summary)")
    st.markdown("""
    <div style="background:linear-gradient(135deg, #111 0%, #1a1a1a 100%);
                border-left: 4px solid #FF0000; padding: 20px; border-radius: 6px;
                border-top: 1px solid #333; border-right: 1px solid #333; border-bottom: 1px solid #333;
                margin-top: 10px;">
        <h4 style="color:#FF0000; margin-top:0; font-family:'Barlow Condensed', sans-serif; letter-spacing:1px;">
            AI INTELLIGENCE REPORT
        </h4>
        <p style="color:#E0E0E0; font-size:15px; line-height:1.7;">
            By decoupling rigid human biases and allowing the <b>K-Means algorithm</b> to cluster purely on geometric distances of Log-Normalized temporal and financial data, we unlocked four organically distinct mathematical profiles:
        </p>
        <ul style="color:#CCCCCC; font-size:14px; line-height:1.6; padding-left:20px;">
            <li><b style="color:#2ecc71;">The Champions:</b> The absolute core of PUMA's revenue. They buy extremely frequently and spend premium amounts. <i>Action: VIP loyalty programs and early/exclusive access to drops.</i></li>
            <li><b style="color:#3498db;">The Promising:</b> These customers have excellent, healthy Recency but lack the extreme Frequency/Monetary depth of Champions. <i>Action: Up-sell programs and volume discounts to accelerate their habits.</i></li>
            <li><b style="color:#e67e22;">The At Risk:</b> The decaying middle-class. They were once valuable, but their growing Recency proves they are slipping away to competitors. <i>Action: Aggressive, targeted win-back email campaigns.</i></li>
            <li><b style="color:#e74c3c;">The Dormant:</b> Statistically lost. Their geometric distance puts them far away from healthy purchasing behaviour. <i>Action: Do not waste premium marketing budget on them; restrict outreach to massive clearance/liquidation sales.</i></li>
        </ul>
        <p style="color:#999; font-size:13px; font-style:italic; margin-bottom:0; border-top: 1px solid #333; padding-top: 12px; margin-top: 15px;">
            * Conclusion: Unsupervised Machine Learning drastically outperforms rigid heuristic business rules by objectively mapping organic spending patterns, preventing misallocation of the marketing budget.
        </p>
    </div>
    """, unsafe_allow_html=True)



# ══ Tab 4 — CLV ══════════════════════════════════════════════════════════════

def render_clv(clv, meta, val_df, sales, bgf):
    """Render CLV tab — Restructured into a 4-row narrative."""
    st.markdown("## Customer Lifetime Value (BG/NBD + Gamma-Gamma)")
    
    # Pre-calculate global metrics for the tab
    clv = clv.copy()
    clv['Churn_Risk'] = pd.cut(
        clv['P_Alive'],
        bins=[-1, 0.5, 0.8, 1.0],
        labels=['Churned / Lost', 'Warning (At Risk)', 'Active & Safe']
    )
    total_val = clv['CLV'].sum()
    pos = clv[clv['CLV'] > 0].copy()

    # ── ROW 1: Model Scope & Coverage ────────────────────────────────────────
    st.markdown("### Model Coverage & Database Scope")
    col_a, col_b, col_c = st.columns(3)
    
    # We use rfm_scored which contains the full database
    n_global = len(rfm_scored) 
    n_predictable = len(clv)
    coverage_pct = (n_predictable / n_global * 100) if n_global > 0 else 0
    
    col_a.metric("Total CRM Database", f"{n_global:,}")
    col_b.metric("Predictable Loyalists", f"{n_predictable:,}", help="Customers with at least 1 repeat purchase required for modeling.")
    col_c.metric("Model Coverage %", f"{coverage_pct:.1f}%")

    st.markdown("<div class='section-header'></div>", unsafe_allow_html=True)

    # ── ROW 2: The Big Numbers (Value KPIs) ───────────────────────────────────
    st.markdown("### 12-Month Financial Projections")
    r1c1, r1c2, r1c3 = st.columns(3)
    r1c1.metric("Individual Mean CLV", fmt(meta['clv_mean']))
    r1c2.metric("Total Predicted Database Value", fmt(total_val))
    r1c3.metric("Pareto % (Top-20% Revenue Share)", f"{meta['top20_revenue_share']:.1f}%")
    
    st.markdown("<div class='section-header'></div>", unsafe_allow_html=True)

    # ── ROW 2: Strategic Segments & Opportunity ───────────────────────────────
    r2c1, r2c2 = st.columns(2)
    with r2c1:
        if len(pos):
            # Calculate Average CLV per Tier for a clearer executive view
            avg_clv_tier = pos.groupby('CLV_Tier')['CLV'].mean().reindex(cfg.TIER_ORDER).reset_index()
            fig = go.Figure(go.Bar(
                x=avg_clv_tier['CLV_Tier'], 
                y=avg_clv_tier['CLV'],
                marker=dict(color=[TIER_COLORS.get(t, '#888') for t in avg_clv_tier['CLV_Tier']], opacity=0.85),
                text=[fmt(v) for v in avg_clv_tier['CLV']],
                textposition='outside',
                textfont=dict(size=11, color='#CCCCCC'),
                hovertemplate='%{x}<br>Average CLV: %{y:,.0f} DZD<extra></extra>'
            ))
            fig.update_layout(**puma_theme("Average Predicted CLV per Tier", height=320),
                              yaxis_title="Avg CLV (DZD)", showlegend=False)
            fig.update_yaxes(tickformat=",")
            st.plotly_chart(fig, use_container_width=True)

    with r2c2:
        if 'CLV_Tier' in clv.columns:
            tc = clv['CLV_Tier'].value_counts().reindex(cfg.TIER_ORDER).dropna().reset_index()
            tc.columns = ['Tier', 'Count']
            fig = go.Figure(go.Pie(
                labels=tc['Tier'], values=tc['Count'], hole=0.55,
                marker=dict(
                    colors=[TIER_COLORS.get(t, '#888') for t in tc['Tier']],
                    line=dict(color='#0D0D0D', width=2)),
                hovertemplate='%{label}<br>Count: %{value:,}<br>%{percent}<extra></extra>',
                textinfo='label+percent',
            ))
            fig.update_layout(**puma_theme("CLV Tier Distribution", height=320))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Tiers: **High CLV** = top 20% | **Medium CLV** = middle 30% | **Low CLV** = bottom 50%")


    # ── Strategic Opportunity Simulator ──────────────────────────────────────
    st.markdown("<div class='section-header'></div>", unsafe_allow_html=True)
    st.markdown("### Strategic Opportunity Simulator (What-If Analysis)")
    with st.container():
        op_c1, op_c2 = st.columns([1, 1])
        with op_c1:
            st.markdown("### Simulation Controls")
            move_pct = st.slider("Percentage of Customers to Upgrade", 0, 30, 10, format="%d%%")
            move_scenario = st.selectbox("Migration Scenario", 
                                         options=["Medium to High CLV", "Low to Medium CLV"])
            
            # Simulation Logic
            baseline_rev = clv['CLV'].sum()
            if "Medium to High" in move_scenario:
                source_tier, target_tier = "Medium CLV", "High CLV"
            else:
                source_tier, target_tier = "Low CLV", "Medium CLV"
            
            source_df = clv[clv['CLV_Tier'] == source_tier]
            target_df = clv[clv['CLV_Tier'] == target_tier]
            
            if not source_df.empty and not target_df.empty:
                avg_src = source_df['CLV'].mean()
                avg_tgt = target_df['CLV'].mean()
                diff = avg_tgt - avg_src
                num_to_move = int((move_pct/100) * len(source_df))
                uplift = num_to_move * diff
                sim_rev = baseline_rev + uplift
            else:
                uplift = 0
                sim_rev = baseline_rev

        with op_c2:
            fig_gap = go.Figure()
            fig_gap.add_trace(go.Bar(
                x=['Baseline (Current)', 'Simulated (Potential)'],
                y=[baseline_rev, sim_rev],
                marker_color=['#444444', '#FF0000'],
                text=[fmt(baseline_rev), fmt(sim_rev)],
                textposition='outside', textfont=dict(size=11, color='#CCC')
            ))
            fig_gap.update_layout(**puma_theme("Projected 12-Month Revenue Impact", height=320),
                                  yaxis_title="DZD", showlegend=False)
            fig_gap.update_yaxes(tickformat=",")
            st.plotly_chart(fig_gap, use_container_width=True)
            
            st.markdown(f"""
                <div style='background:#1A1A1A; border-left:3px solid #FF0000; padding:15px; border-radius:4px;'>
                    <div style='color:#FF0000; font-weight:700; font-size:14px; margin-bottom:5px;'>UPLIFT ANALYSIS</div>
                    <div style='color:#DDD; font-size:12px;'>
                        By converting <b>{move_pct}%</b> of {source_tier} customers ({num_to_move:,} people) 
                        to {target_tier} behavior, PUMA could capture an additional <b>{fmt(uplift)}</b> in annual revenue.
                    </div>
                </div>
            """, unsafe_allow_html=True)

    # ── Scientific Diagnostic Row ────────────────────────────────────────────
    st.markdown("<div class='section-header'>Scientific Validation & Diagnostic Analytics</div>", unsafe_allow_html=True)
    v1, v2, v3 = st.columns(3)
    
    with v1:
        # Expected vs Actual (90-Day Holdout)
        if val_df is not None and not val_df.empty:
            fig_val = go.Figure()
            fig_val.add_trace(go.Scatter(
                x=val_df['frequency_cal'], y=val_df['frequency_holdout'],
                mode='lines+markers', name='Actual',
                line=dict(color='#888', width=2), marker=dict(size=6)
            ))
            fig_val.add_trace(go.Scatter(
                x=val_df['frequency_cal'], y=val_df['pred_holdout'],
                mode='lines+markers', name='Model',
                line=dict(color='#FF0000', width=2, dash='dash'), marker=dict(size=6)
            ))
            fig_val.update_layout(
                **puma_theme("Estimated vs. Actual (90-Day Holdout)", height=280),
                xaxis_title="Calibration Purchases", yaxis_title="Holdout Purchases",
            )
            fig_val.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig_val, use_container_width=True)

    with v2:
        # P_Alive Heatmap (The 'Retention Grid')
        # Create a grid of Recency (0-52) and Frequency (0-20)
        max_f, max_r = 15, 52
        fs = np.arange(0, max_f + 1)
        rs = np.arange(0, max_r + 1)
        # T is the age of the customer (we'll assume 52 weeks)
        z = np.zeros((len(fs), len(rs)))
        for i, f in enumerate(fs):
            for j, r in enumerate(rs):
                p_alive = bgf.conditional_probability_alive(f, r, 52)
                # The lifetimes model returns a 1D numpy array or pandas Series here
                z[i, j] = p_alive[0] if hasattr(p_alive, '__iter__') else float(p_alive)
        
        fig_hm = go.Figure(data=go.Heatmap(
            z=z, x=rs, y=fs,
            colorscale=[[0, '#1A1A1A'], [0.5, '#660000'], [1, '#FF0000']],
            colorbar=dict(title='P(Alive)'),
            hovertemplate='Recency: %{x}w<br>Frequency: %{y}<br>Prob Alive: %{z:.1%}<extra></extra>'
        ))
        fig_hm.update_layout(
            **puma_theme("Retention Logic: Prob. of Being Alive", height=280),
            xaxis_title="Recency (Weeks)", yaxis_title="Frequency"
        )
        st.plotly_chart(fig_hm, use_container_width=True)

    with v3:
        # Cumulative Lift Curve
        srt = clv.sort_values('CLV', ascending=False).reset_index(drop=True)
        srt['Cum_Pct']  = srt['CLV'].cumsum() / srt['CLV'].sum() * 100
        srt['Cust_Pct'] = (srt.index + 1) / len(srt) * 100
        fig_lift = go.Figure()
        fig_lift.add_trace(go.Scatter(
            x=srt['Cust_Pct'], y=srt['Cum_Pct'], mode='lines', name='Model Lift',
            line=dict(color='#FF0000', width=2.5), fill='tozeroy', fillcolor='rgba(255,0,0,0.06)'))
        fig_lift.add_trace(go.Scatter(
            x=[0, 100], y=[0, 100], mode='lines', name='Random',
            line=dict(color='#555', width=1.5, dash='dash')))
        fig_lift.update_layout(**puma_theme("CLV Cumulative Lift", height=280),
                               xaxis_title="% of Customers", yaxis_title="% of Revenue")
        st.plotly_chart(fig_lift, use_container_width=True)

    st.markdown("<div class='section-header'></div>", unsafe_allow_html=True)
    st.markdown("### Actionable Customer Target List")
    st.markdown("""
        Use this searchable list to identify specific customers for retention. 
        **Strategy:** Filter for **High CLV** and **Churned / Lost** risk to target VIPs.
    """)
    # Column-specific filters
    f1, f2 = st.columns(2)
    tier_list = sorted(clv['CLV_Tier'].dropna().unique().tolist())
    risk_list = sorted(clv['Churn_Risk'].dropna().unique().tolist())
    
    sel_tiers = f1.multiselect("Filter by Tier", options=tier_list, default=tier_list)
    sel_risks = f2.multiselect("Filter by Risk Level", options=risk_list, default=risk_list)
    
    target_list = clv[
        (clv['CLV_Tier'].isin(sel_tiers)) &
        (clv['Churn_Risk'].isin(sel_risks))
    ].copy()

    display_cols = ['Client', 'CLV', 'CLV_Tier', 'P_Alive', 'Churn_Risk', 'pred_num_txn']
    st.dataframe(
        target_list[display_cols].sort_values('CLV', ascending=False),
        column_config={
            "Client": "Customer ID",
            "CLV": st.column_config.NumberColumn("12M Predicted CLV", format="%d DZD"),
            "P_Alive": st.column_config.ProgressColumn("Vitality (P-Alive)", min_value=0, max_value=1),
            "Churn_Risk": "Risk Level",
            "pred_num_txn": "Expected Txns (12M)"
        },
        use_container_width=True,
        hide_index=True
    )
    st.caption(f"Showing **{len(target_list):,}** targetable customers based on selected filters.")
    st.markdown("<div class='section-header'></div>", unsafe_allow_html=True)



# ══ Tab 5 — Sales Forecasting ═══════════════════════════════════════════════

def render_forecasting(sales, daily, ft, meta, s_fit, p_model, x_model, horizon, rf_model=None):
    """Render sales forecasting tab (Optimized for Business)."""
    st.markdown("## 📈 Sales Forecasting & Business Impact")

    metrics = meta['metrics']
    best = meta['best_model']
    best_mae = metrics[best]['MAE']
    
    # Calculate Business Metrics
    # Use recent 30 days for business insights instead of lifetime average (to account for seasonality)
    recent_history = daily.tail(30)
    recent_avg_rev = recent_history['y'].mean() if len(recent_history) > 0 else daily['y'].mean()
    lifetime_avg_rev = daily['y'].mean()
    # Forecast Error % = MAE / test-period mean (not lifetime average to avoid >100% values)
    test_mean_rev = daily.tail(60)['y'].mean() if len(daily) >= 60 else lifetime_avg_rev
    forecast_error_pct = min(100, 100 * (best_mae / test_mean_rev)) if test_mean_rev > 0 else 0

    # Generate the forward forecast — ALL models project from the true future
    # (the day after the last known date), so the chart always shows unseen dates.
    # For SARIMA (fitted only on training days 1-303), we forecast N_TEST+horizon
    # steps and discard the first N_TEST (which overlap the test window already shown
    # in the Historical Validation chart above), keeping only the truly new dates.
    N_TRAIN   = 303                              # days used to fit SARIMA
    N_TEST    = meta.get('n_test_days', 60)      # test days already seen (from meta)
    last_dt   = daily['ds'].max()                # last known date for ALL models
    base_date = last_dt + pd.Timedelta(days=1)
    history_y = daily['y'].values

    if best == 'Prophet':
        future    = p_model.make_future_dataframe(periods=horizon)
        fcst      = p_model.predict(future)
        fwd_preds = np.clip(fcst.tail(horizon)['yhat'], 0, None)
    elif best == 'SARIMA':
        # Forecast N_TEST + horizon steps from the training endpoint,
        # then drop the first N_TEST (already-seen test period) to keep
        # only the horizon days that are truly beyond Jan 1 2026.
        s_fcst    = s_fit.get_forecast(steps=N_TEST + horizon)
        fwd_preds = np.clip(s_fcst.predicted_mean[-horizon:], 0, None)
    elif best == 'XGBoost':
        fwd_preds = _xgb_recursive_forecast(x_model, history_y, cfg.FEAT_COLS, base_date, horizon)
    else:  # RF
        _mdl      = rf_model if rf_model is not None else x_model
        fwd_preds = _xgb_recursive_forecast(_mdl, history_y, cfg.FEAT_COLS, base_date, horizon)

    predicted_revenue = sum(fwd_preds)

    # ── Financial Bottom Line Card ──
    m1, m2, m3 = st.columns(3)
    m1.metric(f"Predicted {horizon}-Day Revenue", fmt_currency(predicted_revenue), delta="Expected Value")
    m2.metric("Forecast Error %", f"{forecast_error_pct:.1f}%", delta="MAE / Test-Period Mean")
    m3.metric("Best Model", best, delta="Lowest Error")
    
    st.markdown("<div class='section-header'></div>", unsafe_allow_html=True)
    
    # Actionable Insights Box
    st.markdown("### Actionable Business Insights")
    avg_pred_per_day = predicted_revenue / horizon
    if avg_pred_per_day > recent_avg_rev * 1.05:
        insight = f"📈 **High Demand Expected:** Average daily revenue is projected to be higher than the recent 30-day average. Ensure sufficient stock across top categories and optimize staffing for peak days."
    elif avg_pred_per_day < recent_avg_rev * 0.95:
        insight = f"⚠️ **Slower Period Expected:** Demand is projected to soften compared to last month. Consider activating dormant clients via email campaigns or targeted discounts to 'At Risk' segments."
    else:
        insight = f"⚖️ **Stable Demand:** Sales are projected to remain consistent with recent performance. Maintain current inventory rhythms."

    st.markdown(f"<div class='insight-card'>{insight}</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Test Set: Clean Chart (Winning Model Only) ──
    st.markdown("### Historical Validation (Test Set)")
    st.caption(f"Showing how accurately the {best} model predicted historical unseen data.")
    
    fig_test = go.Figure()
    fig_test.add_trace(go.Scatter(
        x=ft['ds'], y=ft['y'], mode='lines', name='Actual Revenue',
        line=dict(color='#FFFFFF', width=2)))
        
    pred_col_map = {
        'SARIMA' : 'sarima_pred',
        'Prophet': 'prophet_pred',
        'XGBoost': 'xgb_pred',
        'RF'     : 'rf_pred',
    }
    pred_col = pred_col_map.get(best, 'sarima_pred')
    if pred_col not in ft.columns:
        st.warning(f"Prediction column '{pred_col}' not found in saved artifacts. "
                   "Re-run `train_models.py` to regenerate forecast data including RF.")
        pred_col = 'sarima_pred'  # fallback
    
    fig_test.add_trace(go.Scatter(
        x=ft['ds'], y=ft[pred_col], mode='lines',
        name=f"Predicted ({best})",
        line=dict(color='#2ecc71', width=2, dash='dash')))
        
    fig_test.update_layout(**puma_theme(f"Actual vs Projected ({best} Model)", height=380),
                      yaxis_title="Revenue (DZD)")
    fig_test.update_yaxes(tickformat=",")
    st.plotly_chart(fig_test, use_container_width=True)

    # ── Forward Forecast with Best/Worst Case Scenarios ──
    st.markdown(f"### {horizon}-Day Future Forecast & Scenario Planning")
    fut_dates = pd.date_range(base_date, periods=horizon, freq='D')

    fwd_preds_np = np.array(fwd_preds)

    # Confidence bands grow with √horizon (statistically correct for cumulative uncertainty)
    # Factor at day 1 = 1.05, day 30 ≈ 1.27, day 60 ≈ 1.39 — avoids unrealistic linear bloat
    expanding_factor = 1.0 + (np.sqrt(np.arange(horizon)) * 0.05)
    upper_bound = fwd_preds_np + (best_mae * expanding_factor)
    lower_bound = np.maximum(0, fwd_preds_np - (best_mae * expanding_factor))

    fig2 = go.Figure()
    # Show 60 days of history leading up to the forecast origin (last_dt)
    h60 = daily[daily['ds'] <= last_dt].tail(60)
    fig2.add_trace(go.Scatter(
        x=h60['ds'], y=h60['y'], mode='lines', name='Historical (60 Days Pre-Forecast)',
        line=dict(color='#AAAAAA', width=1.5)))

    # Lower CI bound (invisible, anchors the fill)
    fig2.add_trace(go.Scatter(
        x=fut_dates, y=lower_bound, mode='lines',
        line=dict(width=0), showlegend=False, hoverinfo='skip'))

    # Upper CI bound fills back to lower (confidence envelope)
    fig2.add_trace(go.Scatter(
        x=fut_dates, y=upper_bound, mode='lines', fill='tonexty',
        fillcolor='rgba(46,204,113,0.12)', line=dict(width=0),
        name='Confidence Interval (±MAE × √t)'))

    # Expected Revenue — main forecast line, drawn last so it sits on top
    fig2.add_trace(go.Scatter(
        x=fut_dates, y=fwd_preds_np, mode='lines',
        name=f'Expected Daily Revenue ({best})',
        line=dict(color='#2ecc71', width=2.5)))

    # Forecast start marker — use add_shape to avoid Plotly string-date arithmetic bug
    fig2.add_shape(
        type='line',
        x0=last_dt, x1=last_dt, y0=0, y1=1,
        xref='x', yref='paper',
        line=dict(color='#FF0000', width=1, dash='dot'),
    )
    fig2.add_annotation(
        x=last_dt, y=1.0, xref='x', yref='paper',
        text='Forecast Start', showarrow=False,
        font=dict(size=10, color='#FF0000'),
        xanchor='left', yanchor='top',
        bgcolor='rgba(26,26,26,0.7)',
    )

    # Annotate total projected revenue at mid-point of horizon
    mid_idx = horizon // 2
    fig2.add_annotation(
        x=fut_dates[mid_idx], y=float(upper_bound[mid_idx]) * 1.05,
        text=f"Total {horizon}-Day Projection: {fmt(float(sum(fwd_preds_np)))}",
        showarrow=False,
        font=dict(size=11, color='#2ecc71'),
        bgcolor='rgba(26,26,26,0.8)',
        bordercolor='#2ecc71',
        borderwidth=1,
    )

    fig2.update_layout(**puma_theme(
        f"Expected Revenue — Next {horizon} Days", height=420))
    fig2.update_yaxes(tickformat=",", title_text="Daily Revenue (DZD)")
    fig2.update_xaxes(title_text="Date")
    st.plotly_chart(fig2, use_container_width=True)
    st.caption(
        "⚠️ **Methodology Note:** This forecast projects from the **training endpoint "
        f"(last known date: {last_dt.strftime('%d %b %Y')})**, not from today's date. "
        "The historical context (grey line) shows the 60 days preceding that endpoint. "
        "To generate a forecast from the current date, re-run `train_models.py` with "
        "updated sales data."
    )

    with st.expander("🛠️ Advanced Methodology & Technical Details"):
        st.markdown("Here we compare the raw evaluation metrics of all three models on the test set:")
        
        fc1, fc2 = st.columns(2)
        with fc1:
            mp = pd.DataFrame({'Model': cfg.MODEL_ORDER,
                               'MAE': [metrics[m]['MAE'] for m in cfg.MODEL_ORDER]})
            fig = go.Figure(go.Bar(
                x=mp['Model'], y=mp['MAE'],
                marker=dict(color=[MODEL_COLORS[m] for m in mp['Model']]),
                text=[fmt(v) for v in mp['MAE']],
                textposition='outside', textfont=dict(size=10, color='#CCCCCC'),
            ))
            fig.update_layout(**puma_theme("Model MAE Comparison", height=300),
                              showlegend=False)
            fig.update_yaxes(tickformat=",")
            st.plotly_chart(fig, use_container_width=True)

        with fc2:
            st.markdown("### XGBoost Feature Importance")
            fi = pd.DataFrame({
                'Feature': cfg.FEAT_COLS,
                'Importance': x_model.feature_importances_,
            }).sort_values('Importance', ascending=True)
            fig_fi = go.Figure(go.Bar(
                y=fi['Feature'], x=fi['Importance'], orientation='h',
                marker=dict(color='#2ecc71',
                            opacity=np.linspace(0.4, 1.0, len(fi)).tolist()),
            ))
            fig_fi.update_layout(**puma_theme("", height=280), showlegend=False)
            st.plotly_chart(fig_fi, use_container_width=True)
            
        st.caption("SARIMA explicitly models 7-day seasonality ($s=7$). Prophet applies Meta's changepoint algorithm with Fourier seasonality. XGBoost and Random Forest (300 trees) capture lag-based patterns via recursive one-step-ahead projection. **All four models are evaluated on the identical 60-day hold-out window** — no model saw the test data during training.")




def _xgb_recursive_forecast(model, history_y, feat_cols, base_date, n_days):
    """Recursive step-by-step forecast for XGBoost."""
    hist = list(history_y)
    preds = []
    for i in range(n_days):
        current_date = base_date + pd.Timedelta(days=i)
        row = {
            'dayofweek' : current_date.dayofweek,
            'month'     : current_date.month,
            'weekofyear': current_date.isocalendar()[1],
            'lag7'      : hist[-7]  if len(hist) >= 7  else 0,
            'lag14'     : hist[-14] if len(hist) >= 14 else 0,
            'lag28'     : hist[-28] if len(hist) >= 28 else 0,
            'rolling7'  : np.mean(hist[-8:-1]) if len(hist) >= 8 else np.mean(hist),
            'rolling14' : np.mean(hist[-15:-1]) if len(hist) >= 15 else np.mean(hist),
        }
        X_row = pd.DataFrame([row])[feat_cols]
        pred = float(model.predict(X_row)[0])
        pred = max(pred, 0)
        preds.append(pred)
        hist.append(pred)
    return preds





# ══ Tab 7 — Client Intelligence ══════════════════════════════════════════════

def render_clients(rfm_km, sales):
    """Render the master client intelligence tab."""
    st.markdown("## Client Intelligence Master List")
    st.markdown("Browse and filter your entire customer base using Business Rules or AI Modeling.")

    # 1. Selection logic
    r1c1, r1c2, r1c3 = st.columns(3)
    with r1c1:
        model_type = st.radio(
            "Segmentation Model",
            options=["Business Rules (RFM)", "AI Modeling (K-Means)"],
            horizontal=True
        )
    with r1c2:
        all_cats = sorted(rfm_km['Category'].dropna().unique().tolist())
        selected_cats = st.multiselect("Filter Category", options=all_cats, default=[],
                                       placeholder="All categories")
    with r1c3:
        all_brks = sorted(rfm_km['Breakout'].dropna().unique().tolist())
        selected_brks = st.multiselect("Filter Breakout", options=all_brks, default=[],
                                       placeholder="All breakouts")

    r2c1, r2c2, r2c3 = st.columns([1, 1, 2])
    with r2c1:
        all_dims = sorted(rfm_km['Dimension'].dropna().unique().tolist())
        selected_dims = st.multiselect("Filter Dimension", options=all_dims, default=[],
                                       placeholder="All dimensions")
    with r2c2:
        all_ages = sorted(rfm_km['Age_Cat'].dropna().unique().tolist())
        selected_ages = st.multiselect("Filter Age Group", options=all_ages, default=[],
                                       placeholder="All age groups")
    with r2c3:
        if model_type == "Business Rules (RFM)":
            segs = cfg.SEG_ORDER_5
            filt_col = 'Segment'
            selected_segs = st.multiselect("Filter RFM Segment", options=segs, default=[],
                                           placeholder="All segments")
        else:
            segs = cfg.SEG_ORDER_KM
            filt_col = 'KMeans_Segment'
            selected_segs = st.multiselect("Filter K-Means Cluster", options=segs, default=[],
                                           placeholder="All clusters")

    # 2. Filter data — empty selection = show all (no filter applied)
    mask = pd.Series(True, index=rfm_km.index)
    if selected_segs:
        mask &= rfm_km[filt_col].isin(selected_segs)
    if selected_cats:
        mask &= rfm_km['Category'].isin(selected_cats)
    if selected_brks:
        mask &= rfm_km['Breakout'].isin(selected_brks)
    if selected_dims:
        mask &= rfm_km['Dimension'].isin(selected_dims)
    if selected_ages:
        mask &= rfm_km['Age_Cat'].isin(selected_ages)
    df_filt = rfm_km[mask].copy()
    

    # 4. Display with Interaction (Click to Inspect)
    st.markdown(f"**Showing {len(df_filt):,} clients**")
    st.info("💡 **Tip**: Click on any row in the table below to instantly view that client's detailed purchase history.")
    
    # Dynamic Segment Column: Use KMeans_Segment if AI is selected, else original Segment
    if model_type == "AI Modeling (K-Means)":
        df_filt['Segment'] = df_filt['KMeans_Segment']

    # Final columns to show
    df_filt.rename(columns={'Age_Cat': 'Category Age'}, inplace=True)
    show_cols = ['Client', 'Category', 'Breakout', 'Dimension', 'Category Age', 'Recency', 'Frequency', 'Monetary', 'Segment']
    
    # Prepare display dataframe (sorted by Monetary)
    df_display = df_filt[show_cols].sort_values('Monetary', ascending=False)

    # Render interactive dataframe
    selection_event = st.dataframe(
        df_display.style.background_gradient(subset=['Monetary'], cmap='Greens')
        .format({'Monetary': '{:,.0f}', 'Recency': '{:,.0f}'}),
        use_container_width=True,
        height=500,
        on_select="rerun",
        selection_mode="single-row"
    )

    # ══ Individual Client 360° View ══════════════════════════════════════════
    
    # Selection Logic: Prioritize click selection, then fallback to first client
    selected_client_id = None
    
    # 1. Check if user clicked a row
    if selection_event.selection.rows:
        row_idx = selection_event.selection.rows[0]
        selected_client_id = df_display.iloc[row_idx]['Client']
    
    # 2. Add manual search box as fallback and status indicator
    st.markdown("<br>", unsafe_allow_html=True)
    all_clients = rfm_km['Client'].unique().tolist()
    
    search_client = st.selectbox(
        "🔎 Active Inspection (Search or click a row above)",
        options=all_clients,
        index=all_clients.index(selected_client_id) if selected_client_id in all_clients else 0,
    )
    
    # Final selected client
    final_client = selected_client_id if selected_client_id else search_client
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>Client 360° Journey Profiling</div>", unsafe_allow_html=True)

    if final_client:
        # 1. Individual Data
        c_stats = rfm_km[rfm_km['Client'] == final_client].iloc[0]
        c_txns = sales[sales['Client'] == final_client].copy()
        c_txns = c_txns.sort_values('Date', ascending=False)
        
        # 2. KPI Header
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Lifetime Value", fmt(c_stats['Monetary']))
        m2.metric("Total Transactions", f"{int(c_stats['Frequency'])}")
        m3.metric("Avg Order Value", fmt(c_stats['Monetary'] / c_stats['Frequency']))
        
        # Calculate tenure relative to dataset snapshot
        c_txns['Date'] = pd.to_datetime(c_txns['Date'])
        first_txn = c_txns['Date'].min()
        max_date = pd.to_datetime(sales['Date']).max()
        tenure = max(0, (max_date - first_txn).days)
        m4.metric("Customer Age (Days)", f"{tenure}")

        # 3. Visual Journey
        st.markdown("### Historical Spend Timeline")
        
        # Cumulative Revenue over time
        c_cum = c_txns.sort_values('Date').copy()
        c_cum['Cumulative_Revenue'] = c_cum['Revenue'].cumsum()
        
        fig = go.Figure(go.Scatter(
            x=c_cum['Date'], y=c_cum['Cumulative_Revenue'],
            mode='lines+markers',
            name='Cumulative LTV',
            line=dict(color='#FF0000', width=3),
            marker=dict(size=8, color='#000', line=dict(width=2, color='#FF0000')),
            fill='tozeroy',
            fillcolor='rgba(255,0,0,0.1)'
        ))
        fig.update_layout(
            **puma_theme("Individual Wealth Growth Journey"),
            xaxis_title="Purchase Date",
            yaxis_title="Cumulative Revenue (DZD)"
        )
        st.plotly_chart(fig, use_container_width=True)

        # 4. Detailed Purchase Audit
        st.markdown(f"### Purchase History Details (Audit Trail)")
        audit_cols = ['Date_Disp', 'Product', 'Category', 'Breakout', 'Dimension', 'Revenue', 'Store']
        
        # Formatting Date for display
        c_txns['Date_Disp'] = c_txns['Date'].dt.strftime('%d %b %Y')
        
        st.dataframe(
            c_txns[audit_cols]
            .sort_values('Date_Disp', ascending=False)
            .style.background_gradient(subset=['Revenue'], cmap='Reds')
            .format({'Revenue': '{:,.0f} DZD'}),
            use_container_width=True
        )

    st.markdown("---")
    # Restore Master Download
    st.markdown("### Bulk Data Export")
    csv = df_filt[show_cols].to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Filtered Client List (CSV)",
        data=csv,
        file_name=f"puma_clients_{model_type.lower().replace(' ', '_')}.csv",
        mime="text/csv",
    )


# ══ Tab 7 — Documentation ════════════════════════════════════════════════════

def render_docs(km_meta, clv_meta, fc_meta):
    st.markdown("## 📖 Project Documentation")
    st.caption("PUMA Algeria Analytics Dashboard — SARL GREAT WAY | ENSSEA 2025-2026")

    # ── Section 1: Pipeline Overview ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔄 Analytics Pipeline Overview")
    st.markdown("""
    The dashboard is powered by a fully automated Python pipeline (`train_models.py`) that
    transforms raw ERP export data into actionable customer intelligence in four stages:
    """)

    cols = st.columns(4)
    stages = [
        ("1 — Data Ingestion\n& Cleaning",
         "Raw XLSX → 300,761 clean sales rows. Drops duplicates, fixes dtypes, "
         "excludes Passage walk-in code (SC000004) for customer analytics."),
        ("2 — RFM Scoring\n& K-Means",
         "35,696 identified customers scored on Recency / Frequency / Monetary. "
         "K-Means (k=4) fitted on log-scaled features to produce 4 behavioural segments."),
        ("3 — CLV Modelling\n(BG/NBD + GG)",
         "8,562 repeat-purchase customers modelled. BG/NBD predicts future transactions; "
         "Gamma-Gamma predicts average spend. 52-week CLV horizon."),
        ("4 — Demand\nForecasting",
         "Daily revenue series (363 days). SARIMA(2,1,2)(1,0,1)₇ selected as best model "
         "with MAE = 792,505 DZD over a 60-day test window."),
    ]
    for col, (title, desc) in zip(cols, stages):
        with col:
            st.info(f"**{title}**\n\n{desc}")

    # ── Section 2: Data Dictionary ────────────────────────────────────────────
    st.markdown("<div class='section-header'></div>", unsafe_allow_html=True)
    st.markdown("### 📋 Data Dictionary — Raw ERP Columns")
    st.markdown("Source file: **VENTES PUMA 2025.xlsx** — 333,843 raw rows → 300,761 after cleaning")

    dict_data = {
        "ERP Column (French)": [
            "Heure/Date de saisie du doc.",
            "N° du document",
            "Type du document",
            "Client",
            "Libellé Client",
            "Magasin du document",
            "Code article ligne doc.",
            "Désignation ligne doc.",
            "Brand ligne doc.",
            "Famille ligne doc.",
            "Breakout ligne doc.",
            "LIBDIM1",
            "Collection ligne doc.",
            "Prix unitaire TTC net ligne",
            "Quantité ligne doc.",
            "Total TTC ligne doc.",
            "Etablissement du doc.",
            "Vendeur du doc.",
        ],
        "Dashboard Name": [
            "Date", "Doc_ID", "Doc_Type", "Client", "Client_Name",
            "Store", "— (dropped)", "— (dropped)", "— (dropped)", "Category",
            "Breakout", "Dimension", "Collection", "— (dropped)",
            "Qty", "Revenue", "— (dropped)", "Seller",
        ],
        "Status": [
            "✅ Used", "✅ Used", "✅ Used", "✅ Used", "✅ Used",
            "✅ Used", "❌ Dropped", "❌ Dropped", "❌ Dropped", "✅ Used",
            "✅ Dashboard only", "✅ Dashboard only", "✅ Dashboard only", "❌ Dropped",
            "✅ Used", "✅ Used", "❌ Dropped", "✅ Used",
        ],
        "Description": [
            "Transaction datetime", "Unique document number", "Sale / Return / Transfer",
            "Customer code — SC000004 = Passage (walk-in)", "Customer display name",
            "Store / boutique identifier", "Internal article code — not needed post-join",
            "Long article description — redundant", "Brand label — redundant with Category",
            "Product family: Sportstyle, Running, Kids, Teamsport…",
            "Product breakout / style variant — Clients tab filter",
            "Size dimension — Clients tab filter",
            "Season collection — Clients tab filter",
            "Unit price incl. tax — Revenue already computed",
            "Units sold per line", "Line total revenue (DZD) — primary KPI",
            "Legal entity — single entity, zero variance",
            "Salesperson code",
        ],
    }
    st.dataframe(
        dict_data,
        use_container_width=True,
        height=560,
    )

    # ── Section 3: Methodology ────────────────────────────────────────────────
    st.markdown("<div class='section-header'></div>", unsafe_allow_html=True)
    st.markdown("### 🔬 Methodology")

    m1, m2 = st.columns(2)

    with m1:
        st.markdown("#### RFM Scoring")
        st.markdown("""
Each of the 35,696 identified customers (Passage excluded) receives three integer scores (1–5):

| Score | Recency (days) | Frequency (purchases) | Monetary (percentile) |
|---|---|---|---|
| 5 | 0–60 | ≥ 6 | Top 10 % |
| 4 | 60–120 | 4–5 | P70–P90 |
| 3 | 120–210 | 3 | P50–P70 |
| 2 | 210–270 | 2 | P30–P50 |
| 1 | ≥ 270 | 1 | Bottom 30 % |

**RFM Score = R + F + M** (integer, 3–15).
Five rule-based segments assigned via score grid: Champions, Loyal, At Risk, New, Lost.
        """)

        st.markdown("#### K-Means Clustering")
        st.markdown(f"""
K-Means run on **log(1+x)** transformed Frequency & Monetary + raw Recency, then StandardScaler.

| k | Silhouette | Davies-Bouldin | Inertia |
|---|---|---|---|
| 2 | 0.481 | 1.000 | 64,511 |
| 3 | 0.432 | 0.953 | 47,460 |
| **4** | **{km_meta['silhouette']:.3f}** | **{km_meta['davies_bouldin']:.3f}** | **36,547** |
| 5 | 0.325 | 0.984 | 30,719 |
| 6 | 0.328 | 0.929 | 26,797 |

**k = 4 selected** — elbow point, best balance of Silhouette and DB index.
Segments: **Champions, Promising, At Risk, Dormant**.
        """)

    with m2:
        st.markdown("#### BG/NBD + Gamma-Gamma (CLV)")
        st.markdown(f"""
Probabilistic CLV models fitted on **{clv_meta['n_customers']:,} repeat-purchase customers**
(frequency > 0, i.e. ≥ 2 distinct purchase occasions).

- **BG/NBD**: models the number of future transactions given past behaviour.
  Parameters (r, α) govern purchase rate heterogeneity; (a, b) govern churn heterogeneity.
- **Gamma-Gamma**: models average transaction value.
  Parameters (p, q, γ) describe spend distribution.
- Spearman ρ (frequency × monetary) = **{clv_meta['spearman_corr_freq_monetary']:.4f}** < 0.20 ✅ independence confirmed.
- L2 penaliser = 0.01 — prevents overfitting on high-frequency, high-spending customers.
- **52-week horizon** | Weekly discount rate = {clv_meta['discount_rate_weekly']:.4f}
- Median CLV = **{clv_meta['clv_median']:,.0f} DZD** | Top-20% share = **{clv_meta['top20_revenue_share']:.1f}%**
        """)

        st.markdown("#### Sales Forecasting")
        order    = fc_meta['sarima_order']
        seasonal = fc_meta['sarima_seasonal_order']
        metrics  = fc_meta['metrics']
        st.markdown(f"""
Daily revenue series: **363 days** | Train: 303 days | Test: **60 days**
(3 Nov 2025 – 1 Jan 2026 — includes pre-Eid ramp-up and year-end surge).

Four models evaluated in test window (60 days):

| Model | MAE (DZD) | RMSE (DZD) |
|---|---|---|
| **SARIMA({order[0]},{order[1]},{order[2]})({seasonal[0]},{seasonal[1]},{seasonal[2]},{seasonal[3]})** | **{metrics['SARIMA']['MAE']:,.0f}** | **{metrics['SARIMA']['RMSE']:,.0f}** |
| Prophet | {metrics['Prophet']['MAE']:,.0f} | {metrics['Prophet']['RMSE']:,.0f} |
| XGBoost | {metrics['XGBoost']['MAE']:,.0f} | {metrics['XGBoost']['RMSE']:,.0f} |
{f"| Random Forest | {metrics['RF']['MAE']:,.0f} | {metrics['RF']['RMSE']:,.0f} |" if 'RF' in metrics else ''}

**{fc_meta['best_model']} selected** — lowest MAE.
Features used by XGBoost/RF: day-of-week, month, ISO week, lag-7/14/28, rolling-7/14.
        """)

    # ── Section 4: Model Artifacts ────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🗂️ Model Artifacts & Training Log")

    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Training Date",      training_log.get('trained_at', 'Unknown')[:10])
    a2.metric("Raw Rows",           f"{training_log.get('n_rows_raw', 0):,}")
    a3.metric("Clean Sales Rows",   f"{training_log.get('n_rows_sales', 0):,}")
    a4.metric("Snapshot Date",      training_log.get('snapshot_date', '2026-01-02'))

    b1, b2, b3, b4 = st.columns(4)
    b1.metric("K-Means k",          str(km_meta['k']))
    b2.metric("Silhouette Score",   f"{km_meta['silhouette']:.4f}")
    b3.metric("Davies-Bouldin",     f"{km_meta['davies_bouldin']:.4f}")
    b4.metric("CLV Customers",      f"{clv_meta['n_customers']:,}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Median CLV",         f"{clv_meta['clv_median']:,.0f} DZD")
    c2.metric("SARIMA MAE",         f"{metrics['SARIMA']['MAE']/1e6:.2f} M DZD")
    c3.metric("Forecast Horizon",   "60-day test")
    c4.metric("Best Forecast Model", forecast_meta.get('best_model', 'SARIMA'))

    st.markdown("---")
    st.markdown("### 📁 Artifact Files")
    artifact_files = {
        "File": [
            "sales_clean.parquet", "rfm_scored.parquet", "rfm_clustered.parquet",
            "clv_table.parquet", "daily_sales.parquet",
            "kmeans_model.joblib", "kmeans_scaler.joblib", "kmeans_meta.json",
            "bgf_model.pkl", "ggf_model.pkl", "clv_meta.json",
            "sarima_model.pkl", "prophet_model.pkl", "xgb_model.pkl", "rf_forecast.pkl",
            "forecast_meta.json", "training_log.json",
        ],
        "Description": [
            f"{training_log.get('n_rows_sales', 300761):,} cleaned sales transactions (all client types)",
            "35,696 customers with R/F/M scores and rule-based segments",
            "35,696 customers with K-Means cluster assignments",
            "8,562 customers with 52-week CLV predictions",
            "363-day daily revenue series for forecasting",
            "Fitted K-Means model (scikit-learn)",
            "StandardScaler fitted on RFM features",
            "K-Means hyperparameters and k-sweep results",
            "Fitted BG/NBD model (lifetimes library)",
            "Fitted Gamma-Gamma model (lifetimes library)",
            "CLV model parameters and coverage metrics",
            "Fitted SARIMA model (statsmodels)",
            "Fitted Prophet model (Meta)",
            "Fitted XGBoost model (xgboost)",
            "Fitted Random Forest model (scikit-learn) — optional, re-run train_models.py",
            "Forecast model metrics and feature list",
            "Full training run log with timestamps",
        ],
    }
    st.dataframe(artifact_files, use_container_width=True, height=460)


# ══ RENDER TABS ══════════════════════════════════════════════════════════════

with tabs[0]:
    render_eda(sales)

with tabs[1]:
    render_rfm(rfm_scored)

with tabs[2]:
    render_kmeans(rfm_clustered, kmeans_meta)

with tabs[3]:
    render_clv(clv_table, clv_meta, artifacts['clv_validation'], sales, bgf_model)

with tabs[4]:
    render_forecasting(sales, daily_sales, forecast_test, forecast_meta,
                       sarima_fit, prophet_model, xgb_model, forecast_days,
                       rf_model=rf_model)

with tabs[5]:
    render_clients(rfm_clustered, sales_all)

with tabs[6]:
    render_docs(kmeans_meta, clv_meta, forecast_meta)
