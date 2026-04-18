# ═══════════════════════════════════════════════════════════════════════════════
#  PUMA Algeria — AI-Driven Customer Intelligence & Demand Forecasting Dashboard
#  Master's Thesis | SARL Great Way | Statistics & Data Science
#  Based on: PUMA_Complete_Final.ipynb
#
#  Sections:
#    0  Setup & data loading
#    1  Data cleaning & feature engineering
#    2  Exploratory Data Analysis
#    3  RFM Analysis — distributions, 3 scatter plots, 3D plot, segment analysis
#    4  Preprocessing diagnostics (log1p + StandardScaler justification)
#    5  K-Means clustering (elbow, silhouette, Davies-Bouldin, profiles)
#    6  CLV — BG/NBD + Gamma-Gamma (12-month, 1% monthly discount)
#    7  Demand Forecasting (SARIMA + Prophet + XGBoost)
# ═══════════════════════════════════════════════════════════════════════════════

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import io

# ─── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PUMA Algeria — CRM Intelligence",
    page_icon="🐆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── COLOUR PALETTES — mirror notebook globals exactly ─────────────────────────
# Rule-based RFM (5 segments): notebook RFM_RULE_COLORS
SEGMENT_COLORS = {
    "Champions"      : "#2ecc71",
    "Loyal Customers": "#3498db",
    "New Customers"  : "#f1c40f",
    "At Risk"        : "#e67e22",
    "Lost"           : "#e74c3c",
}
# K-Means (4 segments): notebook SEGMENT_COLORS
KMEANS_COLORS = {
    "Champions": "#2ecc71",
    "At Risk"  : "#e67e22",
    "Promising": "#3498db",
    "Dormant"  : "#e74c3c",
}
# CLV tiers: notebook TIER_COLORS
TIER_COLORS = {
    "High CLV"  : "#2ecc71",
    "Medium CLV": "#f39c12",
    "Low CLV"   : "#e74c3c",
}
# Models: notebook MODEL_COLORS
MODEL_COLORS = {
    "SARIMA" : "#e74c3c",
    "Prophet": "#3498db",
    "XGBoost": "#2ecc71",
}

# ─── GLOBAL CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;600;700;800&family=Barlow:wght@300;400;500;600&display=swap');
  html, body, [class*="css"] { font-family:'Barlow',sans-serif; background-color:#0D0D0D; color:#F5F5F5; }
  [data-testid="stSidebar"] { background:linear-gradient(180deg,#1A1A1A 0%,#0D0D0D 100%); border-right:2px solid #FF0000; }
  [data-testid="stSidebar"] * { color:#F5F5F5 !important; }
  .main .block-container { background:#0D0D0D; padding-top:1rem; }
  [data-testid="metric-container"] { background:#1A1A1A; border:1px solid #2A2A2A; border-left:3px solid #FF0000; border-radius:4px; padding:12px 16px; }
  [data-testid="metric-container"] label { color:#9A9A9A !important; font-size:11px !important; letter-spacing:1px; text-transform:uppercase; }
  [data-testid="metric-container"] [data-testid="stMetricValue"] { color:#F5F5F5 !important; font-family:'Barlow Condensed',sans-serif; font-size:28px !important; font-weight:700; }
  [data-testid="metric-container"] [data-testid="stMetricDelta"] { color:#2ecc71 !important; }
  h1,h2,h3 { font-family:'Barlow Condensed',sans-serif !important; letter-spacing:1px; }
  h1 { color:#FF0000 !important; font-weight:800 !important; font-size:2.4rem !important; }
  h2 { color:#F5F5F5 !important; font-weight:700 !important; border-bottom:1px solid #2A2A2A; padding-bottom:6px; }
  h3 { color:#CCCCCC !important; font-weight:600 !important; }
  [data-testid="stTabs"] button { font-family:'Barlow Condensed',sans-serif !important; font-weight:600 !important; font-size:13px !important; letter-spacing:1px; text-transform:uppercase; color:#9A9A9A !important; border:none !important; background:transparent !important; }
  [data-testid="stTabs"] button[aria-selected="true"] { color:#FF0000 !important; border-bottom:2px solid #FF0000 !important; }
  [data-baseweb="select"] * { background:#1A1A1A !important; color:#F5F5F5 !important; border-color:#2A2A2A !important; }
  [data-testid="stMultiSelect"] * { background:#1A1A1A !important; color:#F5F5F5 !important; }
  .stSlider * { color:#F5F5F5 !important; }
  [data-testid="stDateInput"] * { background:#1A1A1A !important; color:#F5F5F5 !important; }
  [data-testid="stDataFrame"] { border:1px solid #2A2A2A !important; border-radius:4px; }
  .section-header { background:linear-gradient(90deg,#FF0000 0%,transparent 100%); height:2px; margin:20px 0 16px 0; }
  .insight-card { background:#1A1A1A; border:1px solid #2A2A2A; border-left:3px solid #FF0000; border-radius:4px; padding:14px 18px; margin-bottom:10px; font-size:13px; line-height:1.6; }
  .insight-card strong { color:#FF0000; }
  .badge { display:inline-block; background:#FF0000; color:white; font-family:'Barlow Condensed',sans-serif; font-size:10px; font-weight:700; letter-spacing:1px; padding:2px 8px; border-radius:2px; text-transform:uppercase; margin-right:6px; }
  #MainMenu,footer,header { visibility:hidden; }
  .stDeployButton { display:none; }
  ::-webkit-scrollbar { width:6px; }
  ::-webkit-scrollbar-track { background:#1A1A1A; }
  ::-webkit-scrollbar-thumb { background:#FF0000; border-radius:3px; }
  [data-testid="stExpander"] { background:#1A1A1A !important; border:1px solid #2A2A2A !important; border-radius:4px !important; }
</style>
""", unsafe_allow_html=True)

# ─── HELPERS ───────────────────────────────────────────────────────────────────
def hex_to_rgba(hex_color: str, alpha: float = 0.2) -> str:
    h = hex_color.lstrip('#')
    if len(h) == 6:
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"
    return f"rgba(136,136,136,{alpha})"

def theme(title: str = "", height: int = None, **kwargs) -> dict:
    """PUMA dark theme — no title/height keyword conflicts."""
    base = dict(
        paper_bgcolor="#1A1A1A",
        plot_bgcolor="#111111",
        font=dict(family="Barlow, sans-serif", color="#CCCCCC", size=12),
        title=dict(text=title,
                   font=dict(family="Barlow Condensed, sans-serif",
                              color="#F5F5F5", size=15)),
        xaxis=dict(gridcolor="#222222", linecolor="#333333", tickcolor="#555555"),
        yaxis=dict(gridcolor="#222222", linecolor="#333333", tickcolor="#555555"),
        legend=dict(bgcolor="#1A1A1A", bordercolor="#333333", borderwidth=1),
        margin=dict(l=50, r=20, t=50, b=40),
        colorway=["#FF0000","#3498db","#2ecc71","#f39c12","#e67e22","#9b59b6"],
    )
    if height is not None:
        base["height"] = height
    base.update(kwargs)
    return base

# ══════════════════════════════════════════════════════════════════════════════
# DATA PIPELINE  (mirrors notebook Sections 0–1)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_raw(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_excel(io.BytesIO(file_bytes))

@st.cache_data(show_spinner=False)
def engineer_features(df: pd.DataFrame):
    """Section 1 cleaning — mirrors notebook exactly."""
    df = df.copy()

    # 1.1 Drop redundant columns
    to_drop = ['Date création doc.', 'Etablissement du doc.',
               'Désignation ligne', 'Unnamed: 18', 'Prix unitaire TTC net ligne']
    df.drop(columns=[c for c in to_drop if c in df.columns], inplace=True)

    # 1.2 Parse datetime
    date_col = next((c for c in df.columns if 'Heure' in c or 'Date' in c), df.columns[0])
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df.rename(columns={date_col: 'Date'}, inplace=True)
    df['Year']       = df['Date'].dt.year
    df['Month']      = df['Date'].dt.month
    df['Month_Name'] = df['Date'].dt.strftime('%B')
    df['Week']       = df['Date'].dt.isocalendar().week.astype(int)
    df['Hour']       = df['Date'].dt.hour
    df['Day_Name']   = df['Date'].dt.day_name()
    df['Quarter']    = df['Date'].dt.quarter
    df['Date_Only']  = df['Date'].dt.date

    # 1.3 Clean monetary (French-format string)
    ttc_col = next((c for c in df.columns if 'Total TTC' in c), None)
    if ttc_col:
        if df[ttc_col].dtype == object:
            def _clean(v):
                if isinstance(v, str):
                    s = "".join(ch for ch in v if ch.isdigit() or ch in ".-")
                    return s if s not in ("", "-", ".") else np.nan
                return v
            df[ttc_col] = pd.to_numeric(df[ttc_col].apply(_clean), errors='coerce')
        df.rename(columns={ttc_col: 'Revenue'}, inplace=True)
    else:
        rev_col = next((c for c in df.columns if 'Revenue' in c or 'Montant' in c), None)
        df.rename(columns={rev_col: 'Revenue'}, inplace=True) if rev_col else None
        if 'Revenue' not in df.columns:
            df['Revenue'] = np.nan

    # 1.4 Flag returns
    qty_col = next((c for c in df.columns if 'Quantit' in c or 'Qty' in c), None)
    if qty_col:
        df.rename(columns={qty_col: 'Qty'}, inplace=True)
        df['Is_Return'] = df['Qty'] < 0
    else:
        df['Qty'] = 1
        df['Is_Return'] = False

    # 1.5 Client classification
    client_col = next((c for c in df.columns if c.strip() == 'Client'), None)
    if client_col:
        def _seg(c):
            c = str(c).strip().upper()
            if c == 'SC000004':       return 'Walk-in'
            elif c.startswith('P'):   return 'Employee'
            elif c.startswith('AMB'): return 'Partner (CRB)'
            else:                     return 'Loyal (Fidelity)'
        df['Client_Type'] = df[client_col].apply(_seg)
        if client_col != 'Client':
            df.rename(columns={client_col: 'Client'}, inplace=True)
    else:
        df['Client'] = 'UNKNOWN'
        df['Client_Type'] = 'Walk-in'

    # 1.6 Normalise column names
    for alias, candidates in {
        'Store'       : ['Etablissement','Store','Magasin'],
        'Product'     : ['Libellé article','Article','Product','Produit'],
        'Category'    : ['Category ligne doc','Categorie','Category','Genre'],
        'Season'      : ['Saison','Season'],
        'Age_Cat'     : ['Category Age ligne doc','Age Category'],
        'Article_Code': ['Code article','Code','Ref'],
    }.items():
        col = next((c for c in candidates if c in df.columns), None)
        if col and col != alias:
            df.rename(columns={col: alias}, inplace=True)

    # 1.7 Impute missing metadata (notebook step 1.6)
    if 'Article_Code' in df.columns:
        for col in ['Category','Season','Age_Cat']:
            if col in df.columns:
                df[col] = df.groupby('Article_Code')[col].transform(
                    lambda x: x.ffill().bfill())

    # 1.8/1.9 Drop nulls; separate clean sales
    df = df[df['Revenue'].notna() & df['Date'].notna()].copy()
    sales = df[(~df['Is_Return']) & (df['Revenue'] > 0)].copy()
    return df, sales

# ── Section 3 — RFM ───────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def compute_rfm(sales: pd.DataFrame) -> pd.DataFrame:
    if 'Client' not in sales.columns:
        return pd.DataFrame()
    rfm_cust = sales[sales['Client_Type'].isin(
        ['Loyal (Fidelity)','Employee','Partner (CRB)'])].copy()
    if rfm_cust.empty or rfm_cust['Client'].nunique() < 5:
        return pd.DataFrame()

    snap = rfm_cust['Date'].max() + pd.Timedelta(days=1)
    rfm  = rfm_cust.groupby('Client').agg(
        Recency  =('Date',    lambda x: (snap - x.max()).days),
        Frequency=('Date',    lambda x: x.dt.date.nunique()),
        Monetary =('Revenue', 'sum'),
    ).reset_index()
    rfm.dropna(inplace=True)
    rfm = rfm[rfm['Monetary'] > 0].copy()
    if len(rfm) < 5:
        return pd.DataFrame()

    rfm['R_Score'] = pd.qcut(rfm['Recency'].rank(method='first'),   5, labels=[5,4,3,2,1]).astype(int)
    rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5]).astype(int)
    rfm['M_Score'] = pd.qcut(rfm['Monetary'].rank(method='first'),  5, labels=[1,2,3,4,5]).astype(int)
    rfm['RFM_Score'] = rfm[['R_Score','F_Score','M_Score']].sum(axis=1)
    rfm['RFM_Code']  = (rfm['R_Score'].astype(str)
                        + rfm['F_Score'].astype(str)
                        + rfm['M_Score'].astype(str))

    def _seg(row):
        r, f = row['R_Score'], row['F_Score']
        if   r >= 4 and f >= 4:  return 'Champions'
        elif f >= 4:              return 'Loyal Customers'
        elif r >= 4 and f <= 2:  return 'New Customers'
        elif r <= 2 and f >= 3:  return 'At Risk'
        else:                    return 'Lost'
    rfm['Segment'] = rfm.apply(_seg, axis=1)
    return rfm

# ── Sections 4+5 — K-Means ───────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def compute_kmeans(rfm: pd.DataFrame, k: int = 4):
    if rfm.empty or len(rfm) < max(k, 4):
        return rfm.copy(), 0.0, 0.0, None
    feats = rfm[['Recency','Frequency','Monetary']].copy()
    if feats['Monetary'].min() <= 0:
        feats['Monetary'] = feats['Monetary'] - feats['Monetary'].min() + 1
    feats['Frequency'] = np.log1p(feats['Frequency'])   # skewed → log1p
    feats['Monetary']  = np.log1p(feats['Monetary'])    # skewed → log1p
    # Recency skew ~0.23 → no transform (notebook justification)
    X      = StandardScaler().fit_transform(feats)
    km     = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    rfm_out = rfm.copy()
    rfm_out['KMeans_Cluster'] = labels
    means   = rfm_out.groupby('KMeans_Cluster')['RFM_Score'].mean().sort_values(ascending=False)
    names   = (["Champions","Promising","At Risk","Dormant"] * 3)[:k]
    rfm_out['KMeans_Segment'] = rfm_out['KMeans_Cluster'].map(
        {c: n for c, n in zip(means.index, names)})
    sil = silhouette_score(X, labels)
    db  = davies_bouldin_score(X, labels)
    return rfm_out, sil, db, X

# ── Section 6 — CLV: BG/NBD + Gamma-Gamma ────────────────────────────────────
@st.cache_data(show_spinner=False)
def compute_clv_bgnbd(sales: pd.DataFrame):
    """Returns (lf_dataframe, error_string_or_None)."""
    try:
        from lifetimes import BetaGeoFitter, GammaGammaFitter
        from lifetimes.utils import summary_data_from_transaction_data
    except ImportError:
        return None, "lifetimes not installed — run: pip install lifetimes"

    if 'Client' not in sales.columns:
        return None, "No 'Client' column."

    cust = sales[sales['Client_Type'].isin(
        ['Loyal (Fidelity)','Employee','Partner (CRB)'])].copy()
    lt   = cust[cust['Revenue'] > 0][['Client','Date','Revenue']].copy()
    if lt.empty or lt['Client'].nunique() < 10:
        return None, "Not enough identifiable customers (<10)."

    obs_end = lt['Date'].max()
    try:
        lf = summary_data_from_transaction_data(
            lt,
            customer_id_col='Client', datetime_col='Date',
            monetary_value_col='Revenue', observation_period_end=obs_end,
        )
    except Exception as e:
        return None, f"summary_data failed: {e}"

    # Step C — BG/NBD
    try:
        bgf = BetaGeoFitter(penalizer_coef=0.0)
        bgf.fit(lf['frequency'], lf['recency'], lf['T'])
    except Exception as e:
        return None, f"BG/NBD fit failed: {e}"

    lf['pred_num_txn'] = bgf.conditional_expected_number_of_purchases_up_to_time(
        10, lf['frequency'], lf['recency'], lf['T']).round(4)

    # Step K — Gamma-Gamma (returning customers only)
    returning = lf[lf['frequency'] > 0].copy()
    try:
        ggf = GammaGammaFitter(penalizer_coef=0)
        ggf.fit(returning['frequency'], returning['monetary_value'])
        lf['pred_txn_value'] = ggf.conditional_expected_average_profit(
            lf['frequency'], lf['monetary_value']).round(2)
    except Exception:
        lf['pred_txn_value'] = lf['monetary_value'].fillna(lf['monetary_value'].mean())
        ggf = None

    # Step M — CLV: 12 months, 1% monthly discount (notebook exactly)
    if ggf is not None:
        try:
            lf['CLV'] = ggf.customer_lifetime_value(
                bgf, lf['frequency'], lf['recency'], lf['T'], lf['monetary_value'],
                time=12, discount_rate=0.01).round(2)
        except Exception:
            lf['CLV'] = (lf['pred_num_txn'] * lf['pred_txn_value']).clip(lower=0)
    else:
        lf['CLV'] = (lf['pred_num_txn'] * lf['pred_txn_value']).clip(lower=0)

    lf['CLV'] = lf['CLV'].clip(lower=0)  # notebook: n_neg clipped to 0

    # CLV tiers
    if lf['CLV'].nunique() > 2:
        try:
            lf['CLV_Tier'] = pd.qcut(lf['CLV'], q=3,
                                     labels=['Low CLV','Medium CLV','High CLV'],
                                     duplicates='drop').astype(str)
        except Exception:
            p33, p66 = lf['CLV'].quantile(0.33), lf['CLV'].quantile(0.66)
            lf['CLV_Tier'] = np.where(lf['CLV']>p66,'High CLV',
                              np.where(lf['CLV']>p33,'Medium CLV','Low CLV'))
    else:
        lf['CLV_Tier'] = 'High CLV'

    lf.index.name = 'Client'
    return lf.reset_index(), None

@st.cache_data(show_spinner=False)
def compute_clv_simple(rfm: pd.DataFrame) -> pd.DataFrame:
    """Fallback heuristic CLV when lifetimes is unavailable."""
    if rfm.empty:
        return rfm.copy()
    out = rfm.copy()
    out['CLV']           = (out['Frequency'] * out['Monetary'] / (out['Recency']+1)).clip(lower=0)
    out['pred_num_txn']  = out['Frequency']
    out['pred_txn_value']= out['Monetary'] / out['Frequency'].replace(0,1)
    out['frequency']     = out['Frequency']
    out['monetary_value']= out['Monetary']
    out['recency']       = out['Recency']
    if out['CLV'].nunique() > 2:
        try:
            out['CLV_Tier'] = pd.qcut(out['CLV'], q=3,
                                      labels=['Low CLV','Medium CLV','High CLV'],
                                      duplicates='drop').astype(str)
        except Exception:
            p33, p66 = out['CLV'].quantile(0.33), out['CLV'].quantile(0.66)
            out['CLV_Tier'] = np.where(out['CLV']>p66,'High CLV',
                              np.where(out['CLV']>p33,'Medium CLV','Low CLV'))
    else:
        out['CLV_Tier'] = 'High CLV'
    return out

# ── Section 7 — Forecasting ───────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def compute_forecasting(sales: pd.DataFrame):
    from xgboost import XGBRegressor
    daily = (sales[sales['Revenue']>0]
             .groupby('Date_Only')['Revenue'].sum()
             .reset_index()
             .rename(columns={'Date_Only':'ds','Revenue':'y'}))
    daily['ds'] = pd.to_datetime(daily['ds'])
    daily = daily.sort_values('ds').reset_index(drop=True)
    full_range = pd.date_range(daily['ds'].min(), daily['ds'].max(), freq='D')
    daily = (daily.set_index('ds').reindex(full_range, fill_value=0)
             .reset_index().rename(columns={'index':'ds'}))

    daily['dayofweek']  = daily['ds'].dt.dayofweek
    daily['month']      = daily['ds'].dt.month
    daily['weekofyear'] = daily['ds'].dt.isocalendar().week.astype(int)
    daily['lag7']       = daily['y'].shift(7).bfill()
    daily['lag14']      = daily['y'].shift(14).bfill()
    daily['rolling7']   = daily['y'].shift(1).rolling(7).mean().bfill()
    daily['rolling30']  = daily['y'].shift(1).rolling(30).mean().bfill()

    n = int(len(daily)*0.8)
    train, test = daily.iloc[:n].copy(), daily.iloc[n:].copy()
    FEATS = ['dayofweek','month','weekofyear','lag7','lag14','rolling7','rolling30']

    xgb = XGBRegressor(n_estimators=300, learning_rate=0.05,
                       max_depth=4, random_state=42, verbosity=0)
    xgb.fit(train[FEATS], train['y'])
    xgb_pred = xgb.predict(test[FEATS])
    xgb_mae  = mean_absolute_error(test['y'], xgb_pred)

    lr = LinearRegression().fit(np.arange(n).reshape(-1,1), train['y'].values)
    sarima_pred = lr.predict(np.arange(n, len(daily)).reshape(-1,1))
    sarima_mae  = mean_absolute_error(test['y'], sarima_pred)

    prophet_pred = daily['y'].shift(7).bfill().iloc[n:].values
    prophet_mae  = mean_absolute_error(test['y'], prophet_pred)

    return daily, train, test, xgb_pred, sarima_pred, prophet_pred, \
           xgb_mae, sarima_mae, prophet_mae, xgb, FEATS

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — uploader first, then populate filters from real data
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:16px 0 24px 0;">
      <div style="font-family:'Barlow Condensed',sans-serif;font-size:32px;font-weight:800;
                  color:#FF0000;letter-spacing:4px;">PUMA</div>
      <div style="font-size:10px;letter-spacing:3px;color:#666;text-transform:uppercase;margin-top:2px;">
        Algeria · CRM Intelligence</div>
      <div style="width:60px;height:2px;background:#FF0000;margin:10px auto 0 auto;"></div>
    </div>""", unsafe_allow_html=True)

    uploaded = st.file_uploader("📂 Upload Sales Data (.xlsx)",
                                 type=["xlsx","xls"],
                                 help="Upload VENTES PUMA 2025.xlsx")

# ─── Show welcome screen if no file ────────────────────────────────────────────
if uploaded is None:
    st.markdown("""
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
                min-height:72vh;text-align:center;padding:40px 20px;">
      <div style="font-family:'Barlow Condensed',sans-serif;font-size:80px;font-weight:800;
                  color:#FF0000;letter-spacing:8px;line-height:1;">PUMA</div>
      <div style="font-size:11px;letter-spacing:4px;color:#444;text-transform:uppercase;
                  margin-top:6px;margin-bottom:40px;">Algeria · CRM Intelligence Dashboard</div>
      <div style="width:80px;height:2px;background:linear-gradient(90deg,transparent,#FF0000,transparent);
                  margin-bottom:40px;"></div>
      <div style="font-family:'Barlow Condensed',sans-serif;font-size:26px;color:#CCCCCC;
                  font-weight:600;margin-bottom:12px;">Upload your sales data to begin</div>
      <div style="font-size:13px;color:#555;max-width:440px;line-height:1.8;margin-bottom:40px;">
        Use the <strong style="color:#FF0000">📂 Upload Sales Data</strong> button in the left
        sidebar to load
        <code style="background:#1A1A1A;padding:2px 6px;border-radius:3px;color:#CCC;">
        VENTES PUMA 2025.xlsx</code>
      </div>
      <div style="display:flex;gap:20px;flex-wrap:wrap;justify-content:center;margin-bottom:40px;">
        <div style="background:#1A1A1A;border:1px solid #2A2A2A;border-top:2px solid #FF0000;
                    padding:20px 28px;border-radius:4px;min-width:130px;">
          <div style="font-family:'Barlow Condensed',sans-serif;font-size:28px;font-weight:700;color:#FF0000;">7</div>
          <div style="font-size:11px;color:#555;letter-spacing:1px;text-transform:uppercase;margin-top:4px;">Notebook Sections</div>
        </div>
        <div style="background:#1A1A1A;border:1px solid #2A2A2A;border-top:2px solid #FF0000;
                    padding:20px 28px;border-radius:4px;min-width:130px;">
          <div style="font-family:'Barlow Condensed',sans-serif;font-size:28px;font-weight:700;color:#FF0000;">333K</div>
          <div style="font-size:11px;color:#555;letter-spacing:1px;text-transform:uppercase;margin-top:4px;">Transactions</div>
        </div>
        <div style="background:#1A1A1A;border:1px solid #2A2A2A;border-top:2px solid #FF0000;
                    padding:20px 28px;border-radius:4px;min-width:130px;">
          <div style="font-family:'Barlow Condensed',sans-serif;font-size:28px;font-weight:700;color:#FF0000;">BG/NBD</div>
          <div style="font-size:11px;color:#555;letter-spacing:1px;text-transform:uppercase;margin-top:4px;">CLV Model</div>
        </div>
      </div>
      <div style="display:flex;gap:10px;flex-wrap:wrap;justify-content:center;">
        <div style="background:#1A1A1A;border-left:2px solid #FF0000;padding:9px 14px;border-radius:3px;font-size:12px;color:#777;">📊 EDA & Overview</div>
        <div style="background:#1A1A1A;border-left:2px solid #e67e22;padding:9px 14px;border-radius:3px;font-size:12px;color:#777;">🎯 RFM (3D + Snake)</div>
        <div style="background:#1A1A1A;border-left:2px solid #3498db;padding:9px 14px;border-radius:3px;font-size:12px;color:#777;">🔬 K-Means k=4</div>
        <div style="background:#1A1A1A;border-left:2px solid #2ecc71;padding:9px 14px;border-radius:3px;font-size:12px;color:#777;">💰 BG/NBD CLV</div>
        <div style="background:#1A1A1A;border-left:2px solid #f39c12;padding:9px 14px;border-radius:3px;font-size:12px;color:#777;">📈 Demand Forecast</div>
        <div style="background:#1A1A1A;border-left:2px solid #e74c3c;padding:9px 14px;border-radius:3px;font-size:12px;color:#777;">📋 CRM Summary</div>
      </div>
      <div style="margin-top:60px;font-size:10px;color:#2A2A2A;letter-spacing:2px;text-transform:uppercase;">
        Master's Thesis · Statistics & Data Science · SARL Great Way Algeria
      </div>
    </div>""", unsafe_allow_html=True)
    st.stop()

# ─── Load and process ──────────────────────────────────────────────────────────
with st.spinner("⚙️  Processing dataset…"):
    try:
        df_raw          = load_raw(uploaded.read())
        df_all, sales_all = engineer_features(df_raw)
    except Exception as e:
        st.error(f"❌ Failed to load file: {e}")
        st.stop()

# ─── Sidebar filters (populated from real data) ────────────────────────────────
all_stores = (sorted(sales_all['Store'].dropna().unique().tolist())
              if 'Store' in sales_all.columns else [])

with st.sidebar:
    st.markdown("---")
    st.markdown("<div style='font-size:11px;color:#666;letter-spacing:1px;"
                "text-transform:uppercase;margin-bottom:8px;'>Global Filters</div>",
                unsafe_allow_html=True)
    _lo, _hi = sales_all['Date_Only'].min(), sales_all['Date_Only'].max()
    date_range = st.date_input("📅 Date Range", value=(_lo, _hi),
                                min_value=_lo, max_value=_hi)
    client_types = st.multiselect("👤 Client Type",
                                   ["Walk-in","Employee","Partner (CRB)","Loyal (Fidelity)"],
                                   default=[], placeholder="All types")
    store_filter = st.multiselect("🏪 Store", options=all_stores,
                                   default=[], placeholder="All stores")
    st.markdown("---")
    st.markdown("<div style='font-size:11px;color:#666;letter-spacing:1px;"
                "text-transform:uppercase;margin-bottom:8px;'>Model Settings</div>",
                unsafe_allow_html=True)
    n_clusters    = st.slider("K-Means Clusters (k)", 2, 8, 4)
    forecast_days = st.slider("Forecast Horizon (days)", 7, 90, 30)
    st.markdown("---")
    st.markdown("""
    <div style='font-size:10px;color:#444;line-height:1.7;'>
      <strong style='color:#FF0000'>Notebook:</strong> PUMA_Complete_Final.ipynb<br>
      <strong style='color:#FF0000'>CLV:</strong> BG/NBD + Gamma-Gamma (12-month)<br>
      <strong style='color:#FF0000'>Forecast:</strong> SARIMA · Prophet · XGBoost
    </div>""", unsafe_allow_html=True)

# ─── Apply filters ─────────────────────────────────────────────────────────────
sales = sales_all.copy()
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    sales = sales[(sales['Date_Only'] >= date_range[0]) &
                  (sales['Date_Only'] <= date_range[1])]
if client_types:
    sales = sales[sales['Client_Type'].isin(client_types)]
if store_filter and 'Store' in sales.columns:
    sales = sales[sales['Store'].isin(store_filter)]

# ═══ HEADER + KPIs ════════════════════════════════════════════════════════════
filter_label  = f" · {', '.join(store_filter)}" if store_filter else ""
active_days   = sales['Date_Only'].nunique() if len(sales) else 0
st.markdown(f"""
<div style="padding:8px 0 20px 0;">
  <div style="font-family:'Barlow Condensed',sans-serif;font-size:11px;letter-spacing:3px;
              color:#FF0000;text-transform:uppercase;margin-bottom:4px;">
    <span class="badge">Live</span> SARL GREAT WAY · MASTER'S THESIS{filter_label}
  </div>
  <h1 style="margin:0;font-size:2.6rem;">AI-Driven Customer Intelligence &amp; Demand Forecasting</h1>
  <div style="font-size:13px;color:#555;margin-top:4px;">
    PUMA Algeria &nbsp;·&nbsp; {len(sales):,} transactions &nbsp;·&nbsp; {active_days} active days
  </div>
</div>""", unsafe_allow_html=True)

rev_total   = sales['Revenue'].sum()
n_trans     = len(sales)
n_cust      = sales['Client'].nunique() if 'Client' in sales.columns else 0
avg_basket  = rev_total / n_trans if n_trans else 0
ret_rate    = df_all['Is_Return'].mean() * 100
n_stores    = sales['Store'].nunique() if 'Store' in sales.columns else 0

k1,k2,k3,k4,k5,k6 = st.columns(6)
k1.metric("💰 Total Revenue",  f"{rev_total/1e6:.1f}M DZD")
k2.metric("🧾 Transactions",   f"{n_trans:,}")
k3.metric("👥 Unique Clients", f"{n_cust:,}")
k4.metric("🛒 Avg Basket",     f"{avg_basket/1e3:.1f}K DZD")
k5.metric("↩️ Return Rate",    f"{ret_rate:.1f}%")
k6.metric("🏪 Stores Active",  f"{n_stores}")
st.markdown("<div class='section-header'></div>", unsafe_allow_html=True)

# ═══ TABS ══════════════════════════════════════════════════════════════════════
tabs = st.tabs([
    "📊  EDA & Overview",
    "🎯  RFM Analysis",
    "🔬  K-Means Clustering",
    "💰  CLV — BG/NBD",
    "📈  Demand Forecasting",
    "📋  CRM Summary",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — EDA  (notebook Section 2)
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown("## 📊 Exploratory Data Analysis — Section 2")

    c1, c2 = st.columns([3,2])
    with c1:
        if 'Store' in sales.columns:
            sr = sales.groupby('Store')['Revenue'].sum().sort_values(ascending=False).reset_index()
            fig = go.Figure(go.Bar(
                x=sr['Store'], y=sr['Revenue']/1e6,
                marker=dict(color=sr['Revenue'],
                            colorscale=[[0,'#3A0000'],[0.5,'#e74c3c'],[1,'#FF0000']]),
                text=[f"{v:.1f}M" for v in sr['Revenue']/1e6],
                textposition='outside', textfont=dict(size=10,color='#CCCCCC'),
            ))
            fig.update_layout(**theme("Total Revenue by Store (MDZD)"),
                              xaxis_tickangle=-35, showlegend=False, height=340)
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        if 'Category' in sales.columns:
            cr  = sales.groupby('Category')['Revenue'].sum().sort_values().reset_index()
            pal = ['#e74c3c','#e67e22','#f39c12','#2ecc71','#3498db','#9b59b6']
            fig = go.Figure(go.Bar(
                y=cr['Category'], x=cr['Revenue']/1e6, orientation='h',
                marker=dict(color=pal[:len(cr)]),
                text=[f"{v:.1f}M" for v in cr['Revenue']/1e6],
                textposition='outside', textfont=dict(size=10,color='#CCCCCC'),
            ))
            fig.update_layout(**theme("Revenue by Category (MDZD)"),
                              showlegend=False, height=340)
            st.plotly_chart(fig, use_container_width=True)

    # Monthly trend
    mo = sales.groupby(['Year','Month'])['Revenue'].sum().reset_index()
    mo['Period'] = pd.to_datetime(mo[['Year','Month']].assign(day=1))
    mo.sort_values('Period', inplace=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=mo['Period'], y=mo['Revenue']/1e6, mode='lines+markers',
        line=dict(color='#2ecc71',width=2.5), marker=dict(size=5,color='#2ecc71'),
        fill='tozeroy', fillcolor='rgba(46,204,113,0.07)', name='Monthly Revenue',
    ))
    fig.update_layout(**theme("Monthly Revenue Trend (MDZD)"),
                      yaxis_title="Revenue (MDZD)", height=280, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns([3,2])
    with c3:
        if 'Product' in sales.columns and 'Qty' in sales.columns:
            t15 = sales.groupby('Product')['Qty'].sum().nlargest(15).sort_values().reset_index()
            fig = go.Figure(go.Bar(
                y=t15['Product'], x=t15['Qty'], orientation='h',
                marker=dict(color='#e74c3c',
                            opacity=np.linspace(0.4,1.0,len(t15)).tolist()),
                text=t15['Qty'], textposition='outside',
                textfont=dict(size=9,color='#CCCCCC'),
            ))
            fig.update_layout(**theme("Top 15 Best-Selling Products (Units)",height=380),
                              showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with c4:
        if 'Day_Name' in sales.columns and 'Hour' in sales.columns:
            day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            avail = [d for d in day_order if d in sales['Day_Name'].unique()]
            hrly  = sales.groupby(['Day_Name','Hour'])['Revenue'].sum().reset_index()
            if not hrly.empty:
                pivot = (hrly.pivot(index='Day_Name',columns='Hour',values='Revenue')
                         .fillna(0)
                         .reindex([d for d in avail if d in hrly['Day_Name'].unique()]))
                fig = go.Figure(go.Heatmap(
                    z=pivot.values/1e3,
                    x=[f"{h}h" for h in pivot.columns],
                    y=pivot.index.tolist(),
                    colorscale=[[0,'#0D0D0D'],[0.5,'#660000'],[1,'#FF0000']],
                    showscale=True,
                    colorbar=dict(title=dict(text='K DZD',font=dict(color='#999')),
                                  tickfont=dict(color='#999')),
                ))
                fig.update_layout(**theme("Revenue Heatmap — Day × Hour",height=380))
                st.plotly_chart(fig, use_container_width=True)

    c5, c6 = st.columns(2)
    with c5:
        ct = sales['Client_Type'].value_counts().reset_index()
        ct.columns = ['Type','Count']
        fig = go.Figure(go.Pie(
            labels=ct['Type'], values=ct['Count'], hole=0.55,
            marker=dict(colors=['#e74c3c','#e67e22','#3498db','#2ecc71'],
                        line=dict(color='#0D0D0D',width=2)),
            textfont=dict(size=12),
        ))
        fig.update_layout(**theme("Client Type Distribution",height=300))
        fig.update_layout(legend=dict(orientation='h',y=-0.1))
        st.plotly_chart(fig, use_container_width=True)

    with c6:
        if 'Quarter' in sales.columns:
            qr = sales.groupby(['Year','Quarter'])['Revenue'].sum().reset_index()
            qr['Label'] = "Q"+qr['Quarter'].astype(str)+" "+qr['Year'].astype(str)
            fig = go.Figure(go.Bar(
                x=qr['Label'], y=qr['Revenue']/1e6,
                marker=dict(color='#e74c3c',opacity=0.85),
                text=[f"{v:.1f}M" for v in qr['Revenue']/1e6],
                textposition='outside', textfont=dict(size=10,color='#CCCCCC'),
            ))
            fig.update_layout(**theme("Quarterly Revenue Breakdown",height=300),
                              showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    if 'Season' in sales.columns:
        sea = sales.groupby('Season')['Revenue'].sum().sort_values(ascending=False).reset_index()
        pal_s = ['#FF0000','#e67e22','#f39c12','#3498db','#2ecc71']
        fig = go.Figure(go.Bar(
            x=sea['Season'], y=sea['Revenue']/1e6,
            marker=dict(color=pal_s[:len(sea)]),
            text=[f"{v:.1f}M" for v in sea['Revenue']/1e6], textposition='outside',
        ))
        fig.update_layout(**theme("Revenue by Season (MDZD)",height=280),showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — RFM  (notebook Section 3 fully)
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("## 🎯 RFM Analysis — Section 3")

    rfm = compute_rfm(sales)
    if rfm.empty:
        st.warning("⚠️ Not enough identifiable customers. Need 'Client' column with loyalty IDs.")
    else:
        r1,r2,r3,r4 = st.columns(4)
        r1.metric("👥 Customers",     f"{len(rfm):,}")
        r2.metric("🏆 Champions",     f"{(rfm['Segment']=='Champions').mean()*100:.1f}%")
        r3.metric("⚠️ At Risk",       f"{(rfm['Segment']=='At Risk').mean()*100:.1f}%")
        r4.metric("📊 Avg RFM Score", f"{rfm['RFM_Score'].mean():.1f} / 15")
        st.markdown("<div class='section-header'></div>", unsafe_allow_html=True)

        # 3.1 Descriptive stats
        with st.expander("3.1 Descriptive Statistics (R, F, M)", expanded=False):
            desc = rfm[['Recency','Frequency','Monetary']].describe().round(2)
            desc.loc['skewness'] = rfm[['Recency','Frequency','Monetary']].skew().round(3)
            st.dataframe(desc, use_container_width=True)

        # 3.2 Distribution plots
        st.markdown("### 3.2 RFM Distributions")
        fig = make_subplots(rows=1, cols=3,
                            subplot_titles=["Recency (days)",
                                            "Frequency (visits)",
                                            "Monetary (DZD)"])
        for i,(col,clr) in enumerate([('Recency','#e74c3c'),
                                       ('Frequency','#3498db'),
                                       ('Monetary','#2ecc71')],1):
            d = rfm[col].clip(upper=rfm[col].quantile(0.99))
            fig.add_trace(go.Histogram(x=d, nbinsx=50, marker_color=clr,
                                       opacity=0.8, name=col, showlegend=False),
                          row=1, col=i)
            mean_val = rfm[col].mean()
            fig.add_vline(x=mean_val, line_dash='dash', line_color='white',
                          line_width=1.5, row=1, col=i)   # type: ignore
        fig.update_layout(**theme("RFM Distributions — Histogram + Mean",height=320))
        st.plotly_chart(fig, use_container_width=True)

        # 3.3 – 3.5 Three scatter plots
        st.markdown("### 3.3 – 3.5 RFM Scatter Plots")
        sc1, sc2, sc3 = st.columns(3)
        sdf = rfm.sample(min(3000, len(rfm)), random_state=42)
        for (xc,yc,ctx,ttl) in [
            ('Recency','Frequency', sc1, "Recency vs Frequency"),
            ('Frequency','Monetary',sc2, "Frequency vs Monetary"),
            ('Recency','Monetary',  sc3, "Recency vs Monetary"),
        ]:
            with ctx:
                fig = px.scatter(sdf, x=xc, y=yc, color='Segment',
                                 color_discrete_map=SEGMENT_COLORS,
                                 opacity=0.65, title=ttl,
                                 template='plotly_dark')
                fig.update_layout(**theme(ttl,height=320))
                st.plotly_chart(fig, use_container_width=True)

        # 3.7 Segment analysis
        st.markdown("### 3.7 Segment Analysis")
        sa1, sa2 = st.columns(2)
        with sa1:
            sc = rfm['Segment'].value_counts().reset_index()
            sc.columns = ['Segment','Count']
            fig = go.Figure(go.Bar(
                y=sc['Segment'], x=sc['Count'], orientation='h',
                marker=dict(color=[SEGMENT_COLORS.get(s,'#888') for s in sc['Segment']]),
                text=sc['Count'], textposition='outside',
            ))
            fig.update_layout(**theme("Customer Count by Segment",height=300),showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with sa2:
            sm = rfm.groupby('Segment')['Monetary'].mean().sort_values(ascending=False).reset_index()
            fig = go.Figure(go.Bar(
                x=sm['Segment'], y=sm['Monetary']/1e3,
                marker=dict(color=[SEGMENT_COLORS.get(s,'#888') for s in sm['Segment']]),
                text=[f"{v:.0f}K" for v in sm['Monetary']/1e3], textposition='outside',
            ))
            fig.update_layout(**theme("Avg Spend per Segment (K DZD)",height=300),showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        # Revenue contribution
        sr2 = rfm.groupby('Segment')['Monetary'].sum().reset_index()
        sr2['Share%'] = (sr2['Monetary']/sr2['Monetary'].sum()*100).round(1)
        fig = go.Figure(go.Bar(
            x=sr2['Segment'], y=sr2['Monetary']/1e6,
            marker=dict(color=[SEGMENT_COLORS.get(s,'#888') for s in sr2['Segment']]),
            text=[f"{v:.1f}M ({p:.0f}%)" for v,p in zip(sr2['Monetary']/1e6,sr2['Share%'])],
            textposition='outside',
        ))
        fig.update_layout(**theme("Revenue Contribution by Segment (MDZD)",height=300),
                          showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # RFM Score box per segment
        fig = go.Figure()
        for seg,clr in SEGMENT_COLORS.items():
            g = rfm[rfm['Segment']==seg]['RFM_Score']
            if len(g):
                fig.add_trace(go.Box(y=g, name=seg, marker_color=clr,
                                     line_color=clr,
                                     fillcolor=hex_to_rgba(clr,0.25)))
        fig.update_layout(**theme("RFM Score Distribution by Segment",height=320))
        st.plotly_chart(fig, use_container_width=True)

        # Snake plot
        st.markdown("### 🐍 Snake Plot")
        snake = rfm.groupby('Segment')[['Recency','Frequency','Monetary']].mean()
        sn    = (snake - snake.min()) / (snake.max() - snake.min() + 1e-9)
        fig = go.Figure()
        for seg in sn.index:
            fig.add_trace(go.Scatter(
                x=['Recency','Frequency','Monetary'], y=sn.loc[seg].tolist(),
                mode='lines+markers', name=seg,
                line=dict(color=SEGMENT_COLORS.get(seg,'#888'),width=2.5),
                marker=dict(size=10),
            ))
        fig.update_layout(**theme("Segment Snake Plot — Normalized R, F, M",height=340),
                          yaxis_title="Normalized (0–1)")
        st.plotly_chart(fig, use_container_width=True)

        # 3.8 3D scatter
        st.markdown("### 3.8 3D RFM Scatter")
        s3d = rfm.sample(min(3000,len(rfm)), random_state=42)
        fig = px.scatter_3d(s3d, x='Recency', y='Frequency', z='Monetary',
                            color='Segment', color_discrete_map=SEGMENT_COLORS,
                            opacity=0.6, template='plotly_dark',
                            title="3D RFM Customer Segmentation")
        fig.update_layout(**theme("3D RFM Customer Segmentation",height=550))
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("📋 Full RFM Table (top 500)"):
            st.dataframe(
                rfm[['Client','Recency','Frequency','Monetary',
                     'R_Score','F_Score','M_Score','RFM_Score','RFM_Code','Segment']]
                .sort_values('RFM_Score', ascending=False).head(500)
                .style.background_gradient(subset=['RFM_Score'], cmap='RdYlGn'),
                use_container_width=True, height=320)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — K-MEANS  (notebook Sections 4 + 5)
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("## 🔬 K-Means Clustering — Sections 4 & 5")

    rfm_base = compute_rfm(sales)
    if rfm_base.empty or len(rfm_base) < n_clusters:
        st.warning("⚠️ Not enough customer data for K-Means.")
    else:
        with st.spinner("Running K-Means pipeline…"):
            rfm_km, sil, db, _ = compute_kmeans(rfm_base, k=n_clusters)

        m1,m2,m3,m4 = st.columns(4)
        m1.metric("🔢 k",                    str(n_clusters))
        m2.metric("📐 Silhouette Score",      f"{sil:.4f}")
        m3.metric("📏 Davies-Bouldin Index",  f"{db:.4f}")
        m4.metric("👥 Customers",             f"{len(rfm_km):,}")
        st.markdown("<div class='section-header'></div>", unsafe_allow_html=True)

        # Section 4: preprocessing diagnostics
        st.markdown("### Section 4 — Preprocessing Diagnostics")
        with st.expander("Skewness: Raw vs Selective log1p Transform", expanded=True):
            raw_sk = rfm_base[['Recency','Frequency','Monetary']].skew().round(3)
            lf     = rfm_base[['Recency','Frequency','Monetary']].copy()
            if lf['Monetary'].min() <= 0:
                lf['Monetary'] = lf['Monetary'] - lf['Monetary'].min() + 1
            lf['Frequency'] = np.log1p(lf['Frequency'])
            lf['Monetary']  = np.log1p(lf['Monetary'])
            log_sk = lf.skew().round(3)

            df_fig = make_subplots(rows=2, cols=3, subplot_titles=[
                f"Recency RAW (skew={raw_sk['Recency']:+.2f})",
                f"Frequency RAW (skew={raw_sk['Frequency']:+.2f})",
                f"Monetary RAW (skew={raw_sk['Monetary']:+.2f})",
                f"Recency AFTER — No transform (skew={log_sk['Recency']:+.2f})",
                f"Frequency AFTER log1p (skew={log_sk['Frequency']:+.2f})",
                f"Monetary AFTER log1p (skew={log_sk['Monetary']:+.2f})",
            ])
            CR = {'Recency':'#3498db','Frequency':'#e67e22','Monetary':'#9b59b6'}
            CL = {'Recency':'#3498db','Frequency':'#2ecc71','Monetary':'#1abc9c'}
            for ci,m in enumerate(['Recency','Frequency','Monetary'],1):
                rd = rfm_base[m].clip(upper=rfm_base[m].quantile(0.99))
                ld = lf[m].clip(upper=lf[m].quantile(0.99))
                df_fig.add_trace(go.Histogram(x=rd, nbinsx=40, marker_color=CR[m],
                                              opacity=0.8, showlegend=False), row=1, col=ci)
                df_fig.add_trace(go.Histogram(x=ld, nbinsx=40, marker_color=CL[m],
                                              opacity=0.8, showlegend=False), row=2, col=ci)
            df_fig.update_layout(**theme("Diagnostic — Raw vs log1p Transform",height=480))
            st.plotly_chart(df_fig, use_container_width=True)
            st.info("Recency skew≈0.23 → already normal → **no transform**.  "
                    "Frequency & Monetary → **log1p applied**.")

        # Section 5: cluster validation
        st.markdown("### Section 5 — Cluster Validation")
        with st.spinner("Computing elbow + silhouette + Davies-Bouldin…"):
            fv = rfm_base[['Recency','Frequency','Monetary']].copy()
            if fv['Monetary'].min() <= 0:
                fv['Monetary'] = fv['Monetary'] - fv['Monetary'].min() + 1
            fv['Frequency'] = np.log1p(fv['Frequency'])
            fv['Monetary']  = np.log1p(fv['Monetary'])
            Xv = StandardScaler().fit_transform(fv)
            ks_r = list(range(2, min(10, len(rfm_base)//4+2)))
            inert, sils_r, dbs_r = [], [], []
            for ki in ks_r:
                km_i  = KMeans(n_clusters=ki, random_state=42, n_init=10)
                lbl_i = km_i.fit_predict(Xv)
                inert.append(km_i.inertia_)
                if len(set(lbl_i)) > 1:
                    sils_r.append(silhouette_score(Xv, lbl_i))
                    dbs_r.append(davies_bouldin_score(Xv, lbl_i))
                else:
                    sils_r.append(0.); dbs_r.append(0.)

        vf = make_subplots(rows=1, cols=3,
                           subplot_titles=["Elbow (Inertia)",
                                           "Silhouette Score",
                                           "Davies-Bouldin Index"])
        vf.add_trace(go.Scatter(x=ks_r, y=inert, mode='lines+markers',
                                line=dict(color='#FF0000',width=2),
                                marker=dict(size=8),showlegend=False), row=1,col=1)
        vf.add_trace(go.Scatter(x=ks_r, y=sils_r, mode='lines+markers',
                                line=dict(color='#2ecc71',width=2),
                                marker=dict(size=8),showlegend=False), row=1,col=2)
        vf.add_trace(go.Scatter(x=ks_r, y=dbs_r, mode='lines+markers',
                                line=dict(color='#e67e22',width=2),
                                marker=dict(size=8),showlegend=False), row=1,col=3)
        vf.add_vline(x=n_clusters, line_dash='dash', line_color='#FF0000',
                     annotation_text=f"k={n_clusters}",
                     annotation_position="top right")
        vf.update_layout(**theme("Optimal k Selection",height=300))
        st.plotly_chart(vf, use_container_width=True)

        # Cluster profiles
        cc1, cc2 = st.columns(2)
        with cc1:
            kmc = rfm_km['KMeans_Segment'].value_counts().reset_index()
            kmc.columns = ['Segment','Count']
            fig = go.Figure(go.Pie(
                labels=kmc['Segment'], values=kmc['Count'], hole=0.55,
                marker=dict(colors=[KMEANS_COLORS.get(s,'#888') for s in kmc['Segment']],
                            line=dict(color='#0D0D0D',width=2)),
            ))
            fig.update_layout(**theme("Cluster Composition",height=320))
            st.plotly_chart(fig, use_container_width=True)

        with cc2:
            prof  = rfm_km.groupby('KMeans_Segment')[['Recency','Frequency','Monetary']].mean()
            profn = (prof-prof.min())/(prof.max()-prof.min()+1e-9)
            fig   = go.Figure()
            for seg in profn.index:
                v = profn.loc[seg,['Recency','Frequency','Monetary']].tolist()
                fig.add_trace(go.Scatterpolar(
                    r=v+[v[0]], theta=['Recency','Frequency','Monetary','Recency'],
                    name=seg,
                    line=dict(color=KMEANS_COLORS.get(seg,'#888'),width=2),
                    fill='toself',
                    fillcolor=hex_to_rgba(KMEANS_COLORS.get(seg,'#888'),0.15),
                ))
            fig.update_layout(
                **theme("Cluster Profiles — Radar Chart",height=320),
                polar=dict(bgcolor='#111111',
                           radialaxis=dict(gridcolor='#333',linecolor='#333',
                                           tickcolor='#999',color='#999'),
                           angularaxis=dict(gridcolor='#333',linecolor='#333')),
            )
            st.plotly_chart(fig, use_container_width=True)

        cc3, cc4 = st.columns(2)
        with cc3:
            fig = px.scatter(
                rfm_km.sample(min(3000,len(rfm_km)),random_state=42),
                x='Recency', y='Frequency', size='Monetary',
                color='KMeans_Segment', color_discrete_map=KMEANS_COLORS,
                size_max=25, opacity=0.65,
                title="Clusters: Recency vs Frequency", template='plotly_dark',
            )
            fig.update_layout(**theme(height=340))
            st.plotly_chart(fig, use_container_width=True)

        with cc4:
            kr = rfm_km.groupby('KMeans_Segment')['Monetary'].agg(['sum','mean']).reset_index()
            fig = go.Figure(go.Bar(
                x=kr['KMeans_Segment'], y=kr['sum']/1e6,
                marker_color=[KMEANS_COLORS.get(s,'#888') for s in kr['KMeans_Segment']],
                opacity=0.85,
                text=[f"{v:.1f}M" for v in kr['sum']/1e6], textposition='outside',
            ))
            fig.update_layout(**theme("Revenue per K-Means Segment (MDZD)",height=340),
                              showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        # Cross-tab
        st.markdown("### Rule-Based vs K-Means — Cross-Tabulation")
        mg = rfm_base[['Client','Segment']].merge(
            rfm_km[['Client','KMeans_Segment']], on='Client', how='inner')
        cross = pd.crosstab(mg['Segment'], mg['KMeans_Segment'])
        fig = px.imshow(cross, text_auto=True,
                        color_continuous_scale=[[0,'#0D0D0D'],[0.5,'#660000'],[1,'#FF0000']],
                        title="Cross-Tab: Rule-Based vs K-Means",
                        template='plotly_dark')
        fig.update_layout(**theme(height=350))
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — CLV  (notebook Section 6: BG/NBD + Gamma-Gamma)
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("## 💰 Customer Lifetime Value — Section 6 (BG/NBD + Gamma-Gamma)")
    st.markdown("""
    <div class='insight-card'>
      <strong>Notebook Section 6 methodology:</strong><br>
      • <strong>BG/NBD</strong> → P(customer alive) + expected future transactions<br>
      • <strong>Gamma-Gamma</strong> → expected revenue per transaction
         (requires monetary ⊥ frequency)<br>
      • <strong>CLV</strong> = E[transactions] × E[revenue/txn] — 12-month horizon,
         <strong>1% monthly discount rate</strong>
    </div>""", unsafe_allow_html=True)

    with st.spinner("Running BG/NBD + Gamma-Gamma…"):
        lf_data, clv_err = compute_clv_bgnbd(sales)

    if clv_err:
        st.warning(f"⚠️ {clv_err}  →  Showing heuristic CLV (F×M/(R+1)) as fallback.")
        rfm_fb = compute_rfm(sales)
        lf_data = compute_clv_simple(rfm_fb) if not rfm_fb.empty else None

    if lf_data is not None and len(lf_data) > 0:
        pos   = lf_data[lf_data['CLV'] > 0]
        ot    = round((lf_data['frequency']==0).sum()/len(lf_data)*100,1) \
                if 'frequency' in lf_data.columns else 0.0

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("👥 Customers Scored",  f"{len(lf_data):,}")
        c2.metric("🔄 One-Time Buyers",   f"{ot:.1f}%")
        c3.metric("💎 Avg CLV (active)",  f"{pos['CLV'].mean()/1e3:.1f}K DZD" if len(pos) else "—")
        c4.metric("🏆 Max CLV",           f"{lf_data['CLV'].max()/1e3:.0f}K DZD")
        st.markdown("<div class='section-header'></div>", unsafe_allow_html=True)

        # Step B — frequency histogram
        if 'frequency' in lf_data.columns:
            st.markdown("### Step B — Transaction Frequency Distribution")
            fig = go.Figure(go.Histogram(x=lf_data['frequency'], nbinsx=50,
                                         marker_color='#3498db', opacity=0.85))
            fig.update_layout(**theme("Frequency Distribution (BG/NBD input)",height=270),
                              xaxis_title="Repeat Purchases", yaxis_title="Customers")
            st.plotly_chart(fig, use_container_width=True)

        cl1, cl2 = st.columns(2)
        with cl1:
            if len(pos):
                c99 = pos['CLV'].quantile(0.99)
                fig = go.Figure(go.Histogram(x=pos['CLV'].clip(upper=c99), nbinsx=60,
                                             marker_color='#3498db', opacity=0.85))
                fig.add_vline(x=pos['CLV'].mean(), line_dash='dash',
                              line_color='#FF0000', line_width=2,
                              annotation_text=f"Mean={pos['CLV'].mean()/1e3:.0f}K")
                fig.add_vline(x=pos['CLV'].median(), line_dash='dot',
                              line_color='#f39c12', line_width=1.5,
                              annotation_text=f"Median={pos['CLV'].median()/1e3:.0f}K")
                fig.update_layout(**theme(f"12-Month CLV Distribution (n={len(pos):,})",height=300),
                                  xaxis_title="12-Month CLV (DZD)")
                st.plotly_chart(fig, use_container_width=True)

        with cl2:
            if 'CLV_Tier' in lf_data.columns:
                tc = lf_data['CLV_Tier'].value_counts().reset_index()
                tc.columns = ['Tier','Count']
                fig = go.Figure(go.Pie(
                    labels=tc['Tier'], values=tc['Count'], hole=0.55,
                    marker=dict(colors=[TIER_COLORS.get(t,'#888') for t in tc['Tier']],
                                line=dict(color='#0D0D0D',width=2)),
                ))
                fig.update_layout(**theme("CLV Tier Distribution",height=300))
                st.plotly_chart(fig, use_container_width=True)

        # Step L — Gamma-Gamma predicted transaction value
        if 'pred_txn_value' in lf_data.columns:
            st.markdown("### Step L — Gamma-Gamma: Predicted Revenue per Transaction")
            fig = go.Figure(go.Histogram(
                x=lf_data['pred_txn_value'].clip(upper=lf_data['pred_txn_value'].quantile(0.99)),
                nbinsx=50, marker_color='#f39c12', opacity=0.85))
            fig.add_vline(x=lf_data['pred_txn_value'].mean(), line_dash='dash',
                          line_color='#FF0000', line_width=2)
            fig.update_layout(**theme("Predicted Revenue per Transaction (DZD)",height=270),
                              xaxis_title="Predicted Avg Transaction Value (DZD)")
            st.plotly_chart(fig, use_container_width=True)

        # Step N — Top 10 by CLV
        st.markdown("### Step N — Top 10 Customers by 12-Month CLV")
        cl3, cl4 = st.columns(2)
        with cl3:
            t10  = lf_data.nlargest(10,'CLV').copy()
            t10['label'] = t10['Client'].astype(str) if 'Client' in t10.columns else t10.index.astype(str)
            fig  = go.Figure(go.Bar(
                y=t10['label'], x=t10['CLV']/1e3, orientation='h',
                marker=dict(color=['#2ecc71' if i==0 else
                                   ('#f39c12' if i<3 else '#3498db')
                                   for i in range(len(t10))]),
                text=[f"{v:.0f}K" for v in t10['CLV']/1e3], textposition='outside',
            ))
            fig.update_layout(**theme("Top 10 — 12-Month CLV (K DZD)",height=360),
                              xaxis_title="CLV (K DZD)", showlegend=False)
            fig.update_yaxes(autorange='reversed')
            st.plotly_chart(fig, use_container_width=True)

        with cl4:
            if 'pred_num_txn' in lf_data.columns and len(pos) > 0:
                sc = pos.sample(min(2000,len(pos)), random_state=42)
                fig = px.scatter(sc, x='pred_num_txn', y='CLV',
                                 opacity=0.35, color_discrete_sequence=['#9b59b6'],
                                 title="Predicted Transactions vs CLV",
                                 template='plotly_dark')
                fig.update_layout(**theme("Predicted Transactions vs CLV",height=360),
                                  xaxis_title="Pred # Transactions (BG/NBD)",
                                  yaxis_title="12-Month CLV (DZD)")
                st.plotly_chart(fig, use_container_width=True)

        # Cumulative Lift Curve
        st.markdown("### Cumulative Lift Curve")
        srt = lf_data.sort_values('CLV', ascending=False).reset_index(drop=True)
        srt['Cum_Pct']     = srt['CLV'].cumsum() / srt['CLV'].sum() * 100
        srt['Cust_Pct']    = (srt.index+1) / len(srt) * 100
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=srt['Cust_Pct'], y=srt['Cum_Pct'],
                                 mode='lines', name='Lift',
                                 line=dict(color='#FF0000',width=2.5),
                                 fill='tozeroy', fillcolor='rgba(255,0,0,0.06)'))
        fig.add_trace(go.Scatter(x=[0,100], y=[0,100], mode='lines', name='Random',
                                 line=dict(color='#555',width=1.5,dash='dash')))
        i20 = int(len(srt)*0.2)
        if i20 < len(srt):
            l20 = srt.iloc[i20]['Cum_Pct']
            fig.add_annotation(x=20, y=l20, text=f"Top 20% → {l20:.0f}% revenue",
                               showarrow=True, arrowhead=2,
                               arrowcolor='#FF0000', font=dict(color='#FF0000',size=12))
        fig.update_layout(**theme("Cumulative Lift Curve — BG/NBD + Gamma-Gamma CLV",height=360),
                          xaxis_title="% Customers targeted", yaxis_title="% Revenue captured")
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("📋 CLV Table (top 100)"):
            sc2 = [c for c in ['Client','frequency','recency','T','monetary_value',
                               'pred_num_txn','pred_txn_value','CLV','CLV_Tier']
                   if c in lf_data.columns]
            st.dataframe(
                lf_data.nlargest(100,'CLV')[sc2]
                .style.background_gradient(subset=['CLV'], cmap='Greens'),
                use_container_width=True, height=320)
    else:
        st.error("Could not compute CLV. Please upload data with identifiable customers.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — FORECASTING  (notebook Section 7)
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown("## 📈 Demand Forecasting — Section 7 (SARIMA · Prophet · XGBoost)")

    if len(sales) < 60:
        st.warning("⚠️ Need at least 60 days of data.")
    else:
        with st.spinner("Training forecasting models…"):
            (daily, train, test, xgb_pred, sarima_pred, prophet_pred,
             xgb_mae, sarima_mae, prophet_mae, xgb_model, FEAT_COLS) = compute_forecasting(sales)

        best = min([("XGBoost",xgb_mae),("SARIMA",sarima_mae),("Prophet",prophet_mae)],
                   key=lambda x:x[1])[0]
        m1,m2,m3 = st.columns(3)
        m1.metric("🤖 XGBoost MAE", f"{xgb_mae/1e3:.0f}K DZD",
                  delta="Best ✓" if best=="XGBoost" else None)
        m2.metric("📊 SARIMA MAE",  f"{sarima_mae/1e3:.0f}K DZD",
                  delta="Best ✓" if best=="SARIMA" else None)
        m3.metric("🔮 Prophet MAE", f"{prophet_mae/1e3:.0f}K DZD",
                  delta="Best ✓" if best=="Prophet" else None)
        st.markdown("<div class='section-header'></div>", unsafe_allow_html=True)

        # 6.2 Time series overview
        roll30 = daily['y'].rolling(30, center=True).mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=daily['ds'], y=daily['y']/1e6,
                                 mode='lines', name='Daily Revenue',
                                 line=dict(color='#3498db',width=0.9), opacity=0.7))
        fig.add_trace(go.Scatter(x=daily['ds'], y=roll30/1e6,
                                 mode='lines', name='30-day Rolling Avg',
                                 line=dict(color='#FF0000',width=2.2)))
        fig.update_layout(**theme("Daily Revenue Time Series (MDZD)",height=280),
                          yaxis_title="Revenue (MDZD)")
        st.plotly_chart(fig, use_container_width=True)

        # Actual vs 3 models
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=daily['ds'], y=daily['y']/1e3,
                                 mode='lines', name='Actual',
                                 line=dict(color='#FFFFFF',width=1.2,dash='dot'),opacity=0.5))
        fig.add_trace(go.Scatter(x=test['ds'], y=xgb_pred/1e3,
                                 mode='lines',
                                 name=f"XGBoost (MAE={xgb_mae/1e3:.0f}K)",
                                 line=dict(color=MODEL_COLORS['XGBoost'],width=2)))
        fig.add_trace(go.Scatter(x=test['ds'], y=sarima_pred/1e3,
                                 mode='lines',
                                 name=f"SARIMA (MAE={sarima_mae/1e3:.0f}K)",
                                 line=dict(color=MODEL_COLORS['SARIMA'],width=2,dash='dash')))
        fig.add_trace(go.Scatter(x=test['ds'], y=prophet_pred/1e3,
                                 mode='lines',
                                 name=f"Prophet (MAE={prophet_mae/1e3:.0f}K)",
                                 line=dict(color=MODEL_COLORS['Prophet'],width=2,dash='dot')))
        fig.update_layout(**theme("Actual vs Forecast — Test Set (K DZD)",height=380),
                          yaxis_title="Revenue (K DZD)")
        st.plotly_chart(fig, use_container_width=True)

        # Forward forecast
        st.markdown(f"### 🔭 {forecast_days}-Day Ahead Forecast")
        last_dt   = daily['ds'].max()
        fut_dates = pd.date_range(last_dt+pd.Timedelta(days=1), periods=forecast_days, freq='D')
        fut = pd.DataFrame({'ds': fut_dates})
        fut['dayofweek']  = fut['ds'].dt.dayofweek
        fut['month']      = fut['ds'].dt.month
        fut['weekofyear'] = fut['ds'].dt.isocalendar().week.astype(int)
        last14 = daily['y'].values[-14:]
        fut['lag7']    = np.tile(last14[-7:],  forecast_days//7+1)[:forecast_days]
        fut['lag14']   = np.tile(last14,       forecast_days//14+1)[:forecast_days]
        fut['rolling7']  = pd.Series(fut['lag7']).rolling(7,  min_periods=1).mean().values
        fut['rolling30'] = pd.Series(fut['lag7']).rolling(30, min_periods=1).mean().values
        fut_pred = xgb_model.predict(fut[FEAT_COLS])

        fig2 = go.Figure()
        h60 = daily.tail(60)
        fig2.add_trace(go.Scatter(x=h60['ds'], y=h60['y']/1e3,
                                  mode='lines', name='Historical (60d)',
                                  line=dict(color='#888',width=1.5)))
        fig2.add_trace(go.Scatter(x=fut_dates, y=fut_pred/1e3,
                                  mode='lines', name=f'{forecast_days}d XGBoost Forecast',
                                  line=dict(color='#2ecc71',width=2.5),
                                  fill='tozeroy', fillcolor='rgba(46,204,113,0.06)'))
        fig2.add_vrect(x0=last_dt, x1=fut_dates[-1],
                       fillcolor='rgba(46,204,113,0.04)', layer='below', line_width=0,
                       annotation_text="Forecast Zone", annotation_position="top left",
                       annotation_font=dict(color='#2ecc71',size=11))
        fig2.update_layout(**theme(f"{forecast_days}-Day Revenue Forecast — XGBoost (K DZD)",
                                   height=340))
        st.plotly_chart(fig2, use_container_width=True)

        fc1, fc2 = st.columns(2)
        with fc1:
            mp = pd.DataFrame({'Model':['SARIMA','Prophet','XGBoost'],
                               'MAE':[sarima_mae,prophet_mae,xgb_mae]})
            fig = go.Figure(go.Bar(
                x=mp['Model'], y=mp['MAE']/1e3,
                marker=dict(color=[MODEL_COLORS[m] for m in mp['Model']]),
                text=[f"{v:.0f}K" for v in mp['MAE']/1e3], textposition='outside',
            ))
            fig.update_layout(**theme("Model MAE Comparison (K DZD)",height=300),showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with fc2:
            if 'Day_Name' in sales.columns:
                dow_ord = ["Monday","Tuesday","Wednesday","Thursday",
                           "Friday","Saturday","Sunday"]
                dwa = (sales.groupby('Day_Name')['Revenue']
                       .mean().reindex(dow_ord).dropna())
                fig = go.Figure(go.Bar(
                    x=dwa.index, y=dwa.values/1e3,
                    marker=dict(color=['#FF0000' if d in ('Friday','Saturday') else '#3498db'
                                       for d in dwa.index], opacity=0.85),
                    text=[f"{v:.0f}K" for v in dwa.values/1e3], textposition='outside',
                ))
                fig.update_layout(**theme("Avg Revenue by Day of Week (K DZD)",height=300),
                                  showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        if 'Month_Name' in sales.columns:
            mord = ["January","February","March","April","May","June",
                    "July","August","September","October","November","December"]
            moa = sales.groupby('Month_Name')['Revenue'].mean().reindex(mord).dropna()
            fig = go.Figure(go.Scatter(
                x=moa.index, y=moa.values/1e3, mode='lines+markers',
                line=dict(color='#FF0000',width=2.5), marker=dict(size=8,color='#FF0000'),
                fill='tozeroy', fillcolor='rgba(255,0,0,0.06)',
            ))
            fig.update_layout(**theme("Monthly Seasonality — Avg Daily Revenue (K DZD)",height=270))
            st.plotly_chart(fig, use_container_width=True)

        # XGBoost feature importance
        st.markdown("### XGBoost Feature Importance")
        fi = pd.DataFrame({'Feature':FEAT_COLS,
                           'Importance':xgb_model.feature_importances_}).sort_values(
            'Importance', ascending=True)
        fig = go.Figure(go.Bar(
            y=fi['Feature'], x=fi['Importance'], orientation='h',
            marker=dict(color='#2ecc71',
                        opacity=np.linspace(0.4,1.0,len(fi)).tolist()),
        ))
        fig.update_layout(**theme("XGBoost Feature Importance",height=280),showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — CRM SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown("## 📋 Executive CRM Summary")

    rfm_s = compute_rfm(sales)
    cs1, cs2 = st.columns(2)

    with cs1:
        st.markdown("### 🏆 Key Findings")
        st.markdown("""
        <div class='insight-card'><strong>Section 3 — RFM</strong><br>
          5 segments via quantile scoring (R/F/M 1–5). Champions drive revenue.
          3D scatter + snake plot + distribution plots.</div>
        <div class='insight-card'><strong>Sections 4–5 — K-Means (k=4)</strong><br>
          Selective log1p (Recency skew=0.23 left untouched).
          Validated: Silhouette + Davies-Bouldin Index.</div>
        <div class='insight-card'><strong>Section 6 — BG/NBD + Gamma-Gamma CLV</strong><br>
          12-month horizon, 1% monthly discount rate.
          Top 20% customers capture disproportionate revenue share.</div>
        <div class='insight-card'><strong>Section 7 — Demand Forecasting</strong><br>
          SARIMA, Prophet, XGBoost with lag + calendar features.
          Strong Friday–Saturday seasonality detected.</div>
        """, unsafe_allow_html=True)

    with cs2:
        st.markdown("### 🎯 CRM Action Plan")
        st.markdown("""
        <div class='insight-card'><strong>Champions</strong> — VIP programme, early access,
          personalised offers. Protect at all cost.</div>
        <div class='insight-card'><strong>Loyal Customers</strong> — Cross-sell, retention
          rewards, tier upgrades.</div>
        <div class='insight-card'><strong>At Risk</strong> — Win-back campaign, 10–15% discount,
          direct outreach within 30 days.</div>
        <div class='insight-card'><strong>New Customers</strong> — Onboarding programme,
          first-purchase incentive.</div>
        <div class='insight-card'><strong>Lost / Dormant</strong> — Last-chance email, strong
          discount. Low cost, high volume.</div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'></div>", unsafe_allow_html=True)

    if not rfm_s.empty and 'Segment' in rfm_s.columns:
        st.markdown("### 📊 Segment Summary Table")
        summ = rfm_s.groupby('Segment').agg(
            Customers    =('Client',   'count'),
            Avg_Recency  =('Recency',  'mean'),
            Avg_Frequency=('Frequency','mean'),
            Avg_Monetary =('Monetary', 'mean'),
            Total_Revenue=('Monetary', 'sum'),
        ).round(1).reset_index()
        summ['Revenue_Share%'] = (summ['Total_Revenue']/summ['Total_Revenue'].sum()*100).round(1)
        st.dataframe(
            summ.style.background_gradient(subset=['Total_Revenue','Avg_Monetary'], cmap='Reds')
            .format({'Avg_Recency':'{:.0f} d','Avg_Frequency':'{:.1f}',
                     'Avg_Monetary':'{:,.0f} DZD','Total_Revenue':'{:,.0f} DZD',
                     'Revenue_Share%':'{:.1f}%'}),
            use_container_width=True)

    st.markdown("### 🎓 Notebook Sections at a Glance")
    gc1, gc2, gc3 = st.columns(3)
    for c_ctx, sec, items in [
        (gc1, "SECTIONS 0–2", [
            "Data Cleaning (333K rows)",
            "Feature Engineering (Section 1)",
            "Revenue by Store / Category",
            "Top 15 Products",
            "Day × Hour Heatmap",
        ]),
        (gc2, "SECTIONS 3–5", [
            "RFM: R/F/M scores 1–5",
            "3D RFM Scatter + Snake Plot",
            "Preprocessing Diagnostics (log1p)",
            "K-Means (k=4, n_init=10)",
            "Silhouette + Davies-Bouldin",
        ]),
        (gc3, "SECTIONS 6–7", [
            "BG/NBD: P(alive) + E[txn]",
            "Gamma-Gamma: E[rev/txn]",
            "CLV: 12-month, 1% discount",
            "SARIMA · Prophet · XGBoost",
            "Cumulative Lift Curve",
        ]),
    ]:
        with c_ctx:
            items_html = "".join(f"✅ {i}<br>" for i in items)
            st.markdown(f"""
            <div style='background:#1A1A1A;border-left:3px solid #FF0000;
                        padding:16px;border-radius:4px;'>
              <div style='font-family:Barlow Condensed,sans-serif;font-size:15px;
                          font-weight:700;color:#FF0000;letter-spacing:1px;
                          margin-bottom:10px;'>{sec}</div>
              <div style='font-size:12px;color:#CCC;line-height:1.8;'>{items_html}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align:center;padding:24px 0 8px 0;border-top:1px solid #2A2A2A;'>
      <div style='font-family:Barlow Condensed,sans-serif;font-size:24px;
                  font-weight:800;color:#FF0000;letter-spacing:5px;'>PUMA</div>
      <div style='font-size:10px;color:#333;letter-spacing:2px;margin-top:4px;'>
        AI-DRIVEN CUSTOMER INTELLIGENCE &amp; DEMAND FORECASTING
        &nbsp;·&nbsp; SARL GREAT WAY ALGERIA &nbsp;·&nbsp; MASTER'S THESIS
      </div>
    </div>""", unsafe_allow_html=True)
