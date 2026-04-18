# Architecture — PUMA Algeria CRM Dashboard

## Project Layout

```
puma_crm/
│
├── VENTES_PUMA_2025.xlsx        ← raw data (do not modify)
│
├── config.py                    ← all paths and constants (single source of truth)
├── train_models.py              ← run once to produce all artifacts
├── app.py                       ← streamlit dashboard (reads artifacts, never trains)
│
├── requirements.txt             ← pinned dependencies
│
└── artifacts/                   ← auto-created by train_models.py
    ├── models/
    │   ├── kmeans_model.joblib      ← fitted KMeans(k=4)
    │   ├── kmeans_scaler.joblib     ← fitted StandardScaler
    │   ├── bgf_model.pkl            ← fitted BetaGeoFitter
    │   ├── ggf_model.pkl            ← fitted GammaGammaFitter
    │   ├── sarima_fit.pkl           ← fitted SARIMAXResults
    │   ├── prophet_model.pkl        ← fitted Prophet
    │   └── xgb_forecast.pkl         ← fitted XGBRegressor
    │
    └── data/
        ├── df_clean.parquet         ← cleaned full dataframe (all rows)
        ├── sales_clean.parquet      ← clean sales only (no returns, Revenue > 0)
        ├── rfm_scored.parquet       ← RFM table + R/F/M scores + Segment (5 classes)
        ├── rfm_clustered.parquet    ← rfm_scored + KMeans_Cluster + KMeans_Segment
        ├── clv_table.parquet        ← lifetimes summary + CLV + CLV_Tier per customer
        ├── daily_sales.parquet      ← daily revenue series (ds, y) — full range
        ├── forecast_test.parquet    ← test period (ds, y, sarima_pred, prophet_pred, xgb_pred)
        ├── kmeans_meta.json         ← silhouette, DB score, cluster mapping
        ├── clv_meta.json            ← CLV summary statistics
        └── forecast_meta.json       ← model orders, metrics, best model, feature cols
```

---

## config.py

```python
from pathlib import Path

# ══ Paths ══════════════════════════════════════════════════════════════════════
ROOT         = Path(__file__).parent
DATA_FILE    = ROOT / 'VENTES_PUMA_2025.xlsx'
ARTIFACTS    = ROOT / 'artifacts'
MODELS_DIR   = ARTIFACTS / 'models'
DATA_DIR     = ARTIFACTS / 'data'

# Data files
DF_CLEAN         = DATA_DIR / 'df_clean.parquet'
SALES_CLEAN      = DATA_DIR / 'sales_clean.parquet'
RFM_SCORED       = DATA_DIR / 'rfm_scored.parquet'
RFM_CLUSTERED    = DATA_DIR / 'rfm_clustered.parquet'
CLV_TABLE        = DATA_DIR / 'clv_table.parquet'
DAILY_SALES      = DATA_DIR / 'daily_sales.parquet'
FORECAST_TEST    = DATA_DIR / 'forecast_test.parquet'
KMEANS_META      = DATA_DIR / 'kmeans_meta.json'
CLV_META         = DATA_DIR / 'clv_meta.json'
FORECAST_META    = DATA_DIR / 'forecast_meta.json'

# Model files
KMEANS_MODEL     = MODELS_DIR / 'kmeans_model.joblib'
KMEANS_SCALER    = MODELS_DIR / 'kmeans_scaler.joblib'
BGF_MODEL        = MODELS_DIR / 'bgf_model.pkl'
GGF_MODEL        = MODELS_DIR / 'ggf_model.pkl'
SARIMA_FIT       = MODELS_DIR / 'sarima_fit.pkl'
PROPHET_MODEL    = MODELS_DIR / 'prophet_model.pkl'
XGB_FORECAST     = MODELS_DIR / 'xgb_forecast.pkl'

# ══ Constants ═════════════════════════════════════════════════════════════════
SNAPSHOT_DATE    = '2026-01-02'    # Fixed reference date for RFM recency
N_CLUSTERS       = 4
CLV_HORIZON_WKS  = 52             # 12 months in weeks
CLV_DISCOUNT_WK  = 0.0023        # weekly discount: (1.01)^(1/4.333)-1
FORECAST_TEST_DAYS = 60

# ══ Segment palettes (single definition) ══════════════════════════════════════
SEGMENT_COLORS_5 = {
    'Champions'      : '#2ecc71',
    'Loyal Customers': '#3498db',
    'New Customers'  : '#f1c40f',
    'At Risk'        : '#e67e22',
    'Lost'           : '#e74c3c',
}
KMEANS_COLORS = {
    'Champions': '#2ecc71',
    'At Risk'  : '#e67e22',
    'Promising': '#3498db',
    'Dormant'  : '#e74c3c',
}
TIER_COLORS = {
    'High CLV'  : '#2ecc71',
    'Medium CLV': '#f39c12',
    'Low CLV'   : '#e74c3c',
}
MODEL_COLORS = {
    'SARIMA' : '#e74c3c',
    'Prophet': '#3498db',
    'XGBoost': '#2ecc71',
}

# ══ Segment orders (for consistent axis ordering) ═════════════════════════════
SEG_ORDER_5     = ['Champions', 'Loyal Customers', 'New Customers', 'At Risk', 'Lost']
SEG_ORDER_KM    = ['Champions', 'At Risk', 'Promising', 'Dormant']
TIER_ORDER      = ['High CLV', 'Medium CLV', 'Low CLV']
MODEL_ORDER     = ['SARIMA', 'Prophet', 'XGBoost']
```

> **Rule**: `train_models.py` and `app.py` both import `config.py` at the top. No hardcoded paths or constants anywhere else.

---

## train_models.py Structure

```
train_models.py
│
├── imports
├── create artifact directories
│
├── ── SECTION 0: Load & Clean ────────────────────────────────────────────────
│   ├── load_raw()           → raw DataFrame
│   ├── clean_data()         → df_clean, sales_clean
│   └── save parquets
│
├── ── SECTION 1: RFM ─────────────────────────────────────────────────────────
│   ├── compute_rfm()        → rfm_scored DataFrame
│   └── save parquet
│
├── ── SECTION 2: K-Means ─────────────────────────────────────────────────────
│   ├── fit_kmeans()         → rfm_clustered, scaler, model, meta
│   └── save parquets + models + JSON
│
├── ── SECTION 3: CLV ─────────────────────────────────────────────────────────
│   ├── fit_clv()            → clv_table, bgf, ggf, meta
│   └── save parquet + models + JSON
│
├── ── SECTION 4: Forecasting ─────────────────────────────────────────────────
│   ├── build_daily_series() → daily_sales DataFrame
│   ├── fit_sarima()         → sarima_fit, predictions
│   ├── fit_prophet()        → prophet_model, predictions
│   ├── fit_xgboost()        → xgb_model, predictions
│   ├── evaluate_models()    → metrics dict
│   └── save parquets + models + JSON
│
└── ── SUMMARY ────────────────────────────────────────────────────────────────
    └── print summary table
```

Each section function signature:
```python
def load_raw(path: Path) -> pd.DataFrame:
    """Load raw Excel file and return unmodified DataFrame."""

def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean raw data. Returns (df_clean, sales_clean)."""

def compute_rfm(sales: pd.DataFrame, snapshot: pd.Timestamp) -> pd.DataFrame:
    """Compute RFM scores and rule-based segments. Returns rfm_scored."""

def fit_kmeans(rfm: pd.DataFrame, k: int) -> tuple[pd.DataFrame, dict]:
    """Fit KMeans clustering. Returns (rfm_clustered, meta_dict)."""

def fit_clv(sales: pd.DataFrame, snapshot: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
    """Fit BG/NBD + Gamma-Gamma. Returns (clv_table, meta_dict)."""

def build_daily_series(sales: pd.DataFrame) -> pd.DataFrame:
    """Build gap-filled daily revenue series."""

def build_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag and rolling features to daily series. Drops NaN rows."""

def fit_sarima(train: pd.DataFrame) -> tuple[object, np.ndarray, dict]:
    """Fit SARIMA via auto_arima. Returns (fitted_result, test_pred, meta)."""

def fit_prophet(train: pd.DataFrame, n_test: int) -> tuple[object, np.ndarray]:
    """Fit Prophet. Returns (model, test_pred)."""

def fit_xgboost(train_lag, test_lag, feat_cols) -> tuple[object, np.ndarray]:
    """Fit XGBoost. Returns (model, test_pred)."""

def xgb_recursive_forecast(model, history_y, feat_cols, base_date, n_days) -> list:
    """Recursive step-by-step forecast for XGBoost. Correct lag propagation."""
```

---

## app.py Structure

```
app.py
│
├── imports + page config
├── CSS (dark PUMA theme — copy from original app, keep as-is)
│
├── ── Artifact Loading ───────────────────────────────────────────────────────
│   └── load_artifacts()     → returns dict of all dataframes + models + metas
│       (decorated with @st.cache_resource)
│       Calls st.stop() with error if any file missing
│
├── ── Sidebar ────────────────────────────────────────────────────────────────
│   ├── date range picker
│   ├── store multiselect
│   ├── forecast horizon slider
│   └── "Trained on: {date}" note
│
├── ── Filter Application ─────────────────────────────────────────────────────
│   └── Apply date + store filter to sales_clean → filtered_sales
│
├── ── Header KPIs ────────────────────────────────────────────────────────────
│   └── 6 × st.metric (computed from filtered_sales)
│
├── ── Tabs ───────────────────────────────────────────────────────────────────
│   ├── Tab 1: render_eda(filtered_sales)
│   ├── Tab 2: render_rfm(rfm_scored)
│   ├── Tab 3: render_kmeans(rfm_clustered, kmeans_meta)
│   ├── Tab 4: render_clv(clv_table, clv_meta)
│   ├── Tab 5: render_forecasting(daily, forecast_test, forecast_meta,
│   │                              sarima_fit, prophet_model, xgb_model,
│   │                              horizon_days)
│   └── Tab 6: render_crm_summary(rfm_scored)
│
└── ── Helper Functions ───────────────────────────────────────────────────────
    ├── puma_theme(title, height)
    └── xgb_recursive_forecast(model, history_y, feat_cols, base_date, n_days)
```

Each tab is a standalone `render_*` function with a clear signature. This keeps `app.py` readable and each tab independently testable.

---

## Data Flow Diagram

```
VENTES_PUMA_2025.xlsx
        │
        ▼
   train_models.py
        │
        ├──► df_clean.parquet ──────────────────────────► Tab 1 (EDA)
        │
        ├──► sales_clean.parquet ──► compute_rfm ──────► Tab 2 (RFM)
        │                                  │
        │                                  ▼
        │                            rfm_scored.parquet
        │                                  │
        │                            fit_kmeans
        │                                  │
        │                                  ▼
        │                           rfm_clustered.parquet ─► Tab 3
        │
        ├──► sales_clean.parquet ──► fit_clv ──────────► Tab 4 (CLV)
        │                                │
        │                           clv_table.parquet
        │
        └──► sales_clean.parquet ──► build_daily ───────► Tab 5 (Forecast)
                                          │
                                    daily_sales.parquet
                                          │
                                    fit_sarima ─► sarima_fit.pkl
                                    fit_prophet ─► prophet_model.pkl
                                    fit_xgboost ─► xgb_forecast.pkl
                                          │
                                    forecast_test.parquet
```

---

## Running the Project

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train all models (run once, takes ~5–15 min depending on machine)
python train_models.py

# 3. Launch dashboard
streamlit run app.py
```

---

## requirements.txt

```
streamlit>=1.32
pandas>=2.0
numpy>=1.26
plotly>=5.20
scikit-learn>=1.4
xgboost>=2.0
statsmodels>=0.14
pmdarima>=2.0
prophet>=1.1
lifetimes>=0.11
joblib>=1.3
openpyxl>=3.1
```

---

## Artifact Versioning Note

`train_models.py` writes a `artifacts/data/training_log.json` at the end:

```json
{
  "trained_at": "2026-01-02T14:30:00",
  "data_file": "VENTES_PUMA_2025.xlsx",
  "n_rows_raw": 333843,
  "n_rows_sales": 310000,
  "snapshot_date": "2026-01-02"
}
```

`app.py` reads this file and displays "Models trained on: {trained_at}" in the sidebar. If this file does not exist, it shows the error and stops.
