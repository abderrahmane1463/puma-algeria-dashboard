from pathlib import Path

# ══ Paths ══════════════════════════════════════════════════════════════════════
ROOT         = Path(__file__).parent
DATA_FILE    = ROOT / 'VENTES PUMA 2025.xlsx'
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
CLV_VALIDATION   = DATA_DIR / 'clv_validation.parquet'
TRAINING_LOG     = DATA_DIR / 'training_log.json'

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
CLV_DISCOUNT_WK  = 0.0023         # weekly discount: (1.01)^(1/4.333)-1
CLV_PENALIZER    = 0.01           # regularization for BG/NBD and Gamma-Gamma
CLV_HOLDOUT_DAYS = 90             # Days to hold back for model validation
FORECAST_TEST_DAYS = 60

# ══ Column rename map ═════════════════════════════════════════════════════════
RENAME_MAP = {
    'Heure de création'      : 'Date',
    'Etablissement'          : 'Store',
    'Libellé article'        : 'Product',
    'Code article'           : 'Article_Code',
    'Category ligne doc'     : 'Category',
    'Category Age ligne doc' : 'Age_Cat',
    'Breakout ligne doc'     : 'Breakout',
    'LIBDIM1'                : 'Dimension',
    'Saison'                 : 'Season',
}

COLS_TO_DROP = [
    'Date création doc.',
    'Etablissement du doc.',
    'Désignation ligne',
    'Unnamed: 18',
    'Prix unitaire TTC net ligne',
]

# ══ Segment palettes (single definition) ══════════════════════════════════════
SEGMENT_COLORS_5 = {
    'Platinum': '#2ecc71',
    'Gold'    : '#3498db',
    'Silver'  : '#f1c40f',
    'Bronze'  : '#e67e22',
    'Lost'    : '#e74c3c',
}
KMEANS_COLORS = {
    'Platinum': '#2ecc71',
    'Gold'    : '#3498db',
    'Silver'  : '#f1c40f',
    'Bronze'  : '#e67e22',
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
SEG_ORDER_5     = ['Platinum', 'Gold', 'Silver', 'Bronze', 'Lost']
SEG_ORDER_KM    = ['Platinum', 'Gold', 'Silver', 'Bronze']
TIER_ORDER      = ['High CLV', 'Medium CLV', 'Low CLV']
MODEL_ORDER     = ['SARIMA', 'Prophet', 'XGBoost']

# ══ K-Means segment names (ranked by mean RFM_Score descending) ═══════════════
KMEANS_SEGMENT_NAMES = ['Platinum', 'Gold', 'Silver', 'Bronze']


# ══ XGBoost feature columns ═══════════════════════════════════════════════════
FEAT_COLS = ['dayofweek', 'month', 'weekofyear', 'lag7', 'lag14', 'lag28',
             'rolling7', 'rolling14']
