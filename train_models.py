# ═══════════════════════════════════════════════════════════════════════════════
#  PUMA Algeria — Model Training Pipeline
#  Master's Thesis | SARL Great Way | Statistics & Data Science
#
#  Run once:  python train_models.py
#  Produces all artifacts consumed by app.py
# ═══════════════════════════════════════════════════════════════════════════════

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import warnings
warnings.filterwarnings("ignore")

import json
import pickle
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             mean_absolute_error, mean_squared_error)
from xgboost import XGBRegressor

import config as cfg

# ── Create artifact directories ───────────────────────────────────────────────
cfg.MODELS_DIR.mkdir(parents=True, exist_ok=True)
cfg.DATA_DIR.mkdir(parents=True, exist_ok=True)


# ══ SECTION 0: Load & Clean ══════════════════════════════════════════════════

def load_raw(path: Path) -> pd.DataFrame:
    """Load raw Excel file and return unmodified DataFrame."""
    print(f" -- Loading {path.name}...")
    df = pd.read_excel(path)
    print(f"    Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


def clean_data(df: pd.DataFrame) -> tuple:
    """Clean raw data. Returns (df_clean, sales_clean)."""
    print(" -- Cleaning data...")
    df = df.copy()

    # Drop redundant columns
    df.drop(columns=cfg.COLS_TO_DROP, errors='ignore', inplace=True)

    # Parse datetime — primary date column
    df['Heure de création'] = pd.to_datetime(df['Heure de création'], errors='coerce')

    # Extract time features
    df['Year']       = df['Heure de création'].dt.year
    df['Month']      = df['Heure de création'].dt.month
    df['Week']       = df['Heure de création'].dt.isocalendar().week.astype(int)
    df['Hour']       = df['Heure de création'].dt.hour
    df['Day_Name']   = df['Heure de création'].dt.day_name()
    df['Month_Name'] = df['Heure de création'].dt.strftime('%B')
    df['Quarter']    = df['Heure de création'].dt.quarter
    df['Date_Only']  = df['Heure de création'].dt.date

    # Clean Revenue — French-formatted string (e.g. "11 950")
    ttc_col = 'Total TTC ligne'
    if ttc_col in df.columns:
        if df[ttc_col].dtype == object:
            df[ttc_col] = df[ttc_col].apply(_clean_currency)
        df[ttc_col] = pd.to_numeric(df[ttc_col], errors='coerce')
    df.rename(columns={ttc_col: 'Revenue'}, inplace=True)

    # Quantity and returns
    if 'Quantité' in df.columns:
        df.rename(columns={'Quantité': 'Qty'}, inplace=True)
    df['Is_Return'] = df['Qty'] < 0

    # Apply column renames
    df.rename(columns=cfg.RENAME_MAP, inplace=True)

    # Client classification
    df['Client_Type'] = df['Client'].apply(_classify_client)

    # Metadata imputation
    if 'Article_Code' in df.columns:
        for col in ['Category', 'Season', 'Age_Cat']:
            if col in df.columns:
                df[col] = df.groupby('Article_Code')[col].transform(
                    lambda x: x.ffill().bfill())

    # Drop rows missing essential data
    df = df[df['Revenue'].notna() & df['Date'].notna()].copy()

    # Clean sales subset
    sales = df[(df['Is_Return'] == False) & (df['Revenue'] > 0)].copy()

    print(f"    Full dataset: {len(df):,} rows")
    print(f"    Clean sales:  {len(sales):,} rows")
    print(f"    Returns:      {df['Is_Return'].sum():,}")
    return df, sales


def _clean_currency(value):
    """Strip non-numeric chars from French-formatted currency string."""
    if isinstance(value, str):
        s = "".join(ch for ch in value if ch.isdigit() or ch in ".-")
        return s if s not in ("", "-", ".") else np.nan
    return value


def _classify_client(cid):
    """Classify client ID into business-meaningful type."""
    c = str(cid).strip().upper()
    if c == 'SC000004':
        return 'Passage'
    elif c.startswith('P'):
        return 'Personnel'
    elif c.startswith('AMB'):
        return 'Partner'
    else:
        return 'Fidélité'



# ══ SECTION 1: RFM ═══════════════════════════════════════════════════════════

def compute_rfm(sales: pd.DataFrame, snapshot: pd.Timestamp) -> pd.DataFrame:
    """Compute RFM scores and rule-based segments. Returns rfm_scored."""
    print(" -- Computing RFM analysis...")

    # Identifiable customers only
    rfm_customers = sales[sales['Client_Type'].isin(
        ['Fidélité', 'Personnel', 'Partner'])].copy()

    rfm = rfm_customers.groupby('Client').agg(
        Recency   = ('Date', lambda x: (snapshot - x.max()).days),
        Frequency = ('Date', lambda x: x.dt.date.nunique()),
        Monetary  = ('Revenue', 'sum'),
    ).reset_index()
    
    # Calculate dominant category per customer
    cat_pref = sales.groupby(['Client', 'Category'])['Revenue'].sum().reset_index()
    top_cat = cat_pref.sort_values(['Client', 'Revenue'], ascending=[True, False]).drop_duplicates('Client')
    rfm = rfm.merge(top_cat[['Client', 'Category']], on='Client', how='left')

    # Calculate dominant Age Category per customer
    age_pref = sales.groupby(['Client', 'Age_Cat'])['Revenue'].sum().reset_index()
    top_age = age_pref.sort_values(['Client', 'Revenue'], ascending=[True, False]).drop_duplicates('Client')
    rfm = rfm.merge(top_age[['Client', 'Age_Cat']], on='Client', how='left')

    # Calculate dominant Breakout per customer
    brk_pref = sales.groupby(['Client', 'Breakout'])['Revenue'].sum().reset_index()
    top_brk = brk_pref.sort_values(['Client', 'Revenue'], ascending=[True, False]).drop_duplicates('Client')
    rfm = rfm.merge(top_brk[['Client', 'Breakout']], on='Client', how='left')

    # Calculate dominant Dimension per customer
    dim_pref = sales.groupby(['Client', 'Dimension'])['Revenue'].sum().reset_index()
    top_dim = dim_pref.sort_values(['Client', 'Revenue'], ascending=[True, False]).drop_duplicates('Client')
    rfm = rfm.merge(top_dim[['Client', 'Dimension']], on='Client', how='left')

    rfm = rfm[rfm['Monetary'] > 0].copy()

    # Business-Logic Threshold Scoring
    # 1. Recency (Days since last purchase)
    # 5: 0-2mo, 4: 2-4mo, 3: 4-7mo, 2: 7-9mo, 1: 9mo+
    rfm['R_Score'] = pd.cut(
        rfm['Recency'], 
        bins=[-1, 60, 120, 210, 270, 99999], 
        labels=[5, 4, 3, 2, 1]
    ).astype(int)

    # 2. Frequency (Transaction count)
    # 1: 1, 2: 2, 3: 3, 4: 4-5, 5: 6+
    rfm['F_Score'] = pd.cut(
        rfm['Frequency'], 
        bins=[0, 1, 2, 3, 5, 99999], 
        labels=[1, 2, 3, 4, 5]
    ).astype(int)

    # 3. Monetary (Revenue quantiles: 30-20-20-20-10)
    rfm['M_Score'] = pd.qcut(
        rfm['Monetary'].rank(method='first'), 
        q=[0, 0.3, 0.5, 0.7, 0.9, 1.0], 
        labels=[1, 2, 3, 4, 5]
    ).astype(int)

    rfm['RFM_Score'] = rfm[['R_Score', 'F_Score', 'M_Score']].sum(axis=1)
    rfm['RFM_Code'] = (rfm['R_Score'].astype(str)
                        + rfm['F_Score'].astype(str)
                        + rfm['M_Score'].astype(str))

    # Rule-based segmentation (5 segments)
    rfm['Segment'] = rfm.apply(_rfm_segment, axis=1)

    print(f"    Customers scored: {len(rfm):,}")
    print(f"    Segment distribution:")
    for seg, cnt in rfm['Segment'].value_counts().items():
        print(f"      {seg:20s}: {cnt:,}")
    return rfm


def _rfm_segment(row):
    """
    Assign Platinum/Gold/Silver/Bronze/Lost tiers based on temporal hierarchy.
    """
    r = row['R_Score']
    f = row['F_Score']
    m = row['M_Score']

    if r >= 4 and f >= 4:
        return 'Platinum'
    if r >= 3 and f >= 3:
        return 'Gold'
    if r >= 4:
        return 'Silver'
    if r >= 2:
        return 'Bronze'
    return 'Lost'



# ══ SECTION 2: K-Means ═══════════════════════════════════════════════════════

def fit_kmeans(rfm: pd.DataFrame, k: int) -> tuple:
    """Fit KMeans clustering. Returns (rfm_clustered, meta_dict)."""
    print(f" -- Fitting K-Means (k={k})...")

    # Feature preparation — use raw continuous variables for geometric distance
    feats = rfm[['Recency', 'Frequency', 'Monetary']].copy()
    
    # 1. Log Transform to correct extreme right-skewness (Whales/Outliers)
    # We use np.log1p (log(1+x)) to gracefully handle 0 values in Recency/Monetary
    feats_log = np.log1p(feats)

    # 2. Scale and fit to balance units (days vs thousands of DZD)
    scaler = StandardScaler()
    X = scaler.fit_transform(feats_log)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)

    # Validation metrics
    sil = silhouette_score(X, labels)
    db  = davies_bouldin_score(X, labels)

    # Segment naming — rank clusters by mean RFM_Score descending
    rfm_out = rfm.copy()
    rfm_out['KMeans_Cluster'] = labels
    means = (rfm_out.groupby('KMeans_Cluster')['RFM_Score']
             .mean().sort_values(ascending=False))
    cluster_to_segment = {
        c: name for c, name in zip(means.index, cfg.KMEANS_SEGMENT_NAMES)
    }
    rfm_out['KMeans_Segment'] = rfm_out['KMeans_Cluster'].map(cluster_to_segment)

    # Save models
    joblib.dump(km, cfg.KMEANS_MODEL)
    joblib.dump(scaler, cfg.KMEANS_SCALER)

    meta = {
        'silhouette': round(float(sil), 4),
        'davies_bouldin': round(float(db), 4),
        'k': k,
        'segment_names': cfg.KMEANS_SEGMENT_NAMES,
        'cluster_to_segment': {int(k_): v for k_, v in cluster_to_segment.items()},
    }

    print(f"    Silhouette: {sil:.4f}")
    print(f"    Davies-Bouldin: {db:.4f}")
    for seg, cnt in rfm_out['KMeans_Segment'].value_counts().items():
        print(f"      {seg:15s}: {cnt:,}")
    return rfm_out, meta


# ══ SECTION 3: CLV — BG/NBD + Gamma-Gamma ════════════════════════════════════

def fit_clv(sales: pd.DataFrame, snapshot: pd.Timestamp) -> tuple:
    """Fit BG/NBD + Gamma-Gamma. Returns (clv_table, meta_dict)."""
    print(" -- Fitting CLV models (BG/NBD + Gamma-Gamma)...")
    from lifetimes import BetaGeoFitter, GammaGammaFitter
    from lifetimes.utils import summary_data_from_transaction_data

    # Identifiable customers, positive revenue
    cust = sales[sales['Client_Type'].isin(
        ['Fidélité', 'Personnel', 'Partner'])].copy()
    lt_row = cust[cust['Revenue'] > 0][['Client', 'Date', 'Revenue']].copy()

    # CRITICAL: Aggregate to transaction level (one row per Client + Date)
    # lifetimes assumes one row per transaction, not per item.
    lt_df = lt_row.groupby(['Client', 'Date'])['Revenue'].sum().reset_index()
    print(f"    Aggregated to {len(lt_df):,} transactions (from {len(lt_row):,} items)")


    # Build lifetimes summary — weekly frequency
    lf = summary_data_from_transaction_data(
        lt_df,
        customer_id_col='Client',
        datetime_col='Date',
        monetary_value_col='Revenue',
        observation_period_end=snapshot,
        freq='W',
    )
    # lf = lf[lf['frequency'] > 0].copy() # Existing logic
    
    # NEW: Clip Monetary Outliers to prevent extreme spenders from distorting predictions
    # We clip at the 99th percentile to keep 99% of data but remove 'Wholesale/Error' spikes
    if not lf.empty:
        up_limit = lf['monetary_value'].quantile(0.99)
        lf['monetary_value'] = lf['monetary_value'].clip(upper=up_limit)
        print(f"    Clipped monetary outliers at {up_limit:,.0f} DZD")
    
    lf = lf[lf['frequency'] > 0].copy()
    print(f"    Customers with repeat purchases: {len(lf):,}")

    # Gamma-Gamma assumption check
    corr = lf['frequency'].corr(lf['monetary_value'], method='spearman')
    if abs(corr) > 0.30:
        print(f"    WARNING: Spearman corr(frequency, monetary_value) = {corr:.3f}"
              " -- Gamma-Gamma assumption weakened")
    else:
        print(f"    Spearman corr(frequency, monetary_value) = {corr:.3f} -- OK")

    # ── Model Validation: Expected vs Actual ──────────────────────────────────
    from lifetimes.utils import calibration_and_holdout_data
    
    # Separate model for validation (training on calibration period)
    cal_end = snapshot - pd.Timedelta(days=cfg.CLV_HOLDOUT_DAYS)
    ch = calibration_and_holdout_data(
        lt_df, 'Client', 'Date',
        calibration_period_end=cal_end,
        observation_period_end=snapshot,
        freq='W'
    )
    # Only customers with at least one repeat purchase in calibration
    ch_val = ch[ch['frequency_cal'] > 0].copy()
    
    if len(ch_val) > 0:
        bgf_val = BetaGeoFitter(penalizer_coef=cfg.CLV_PENALIZER)
        bgf_val.fit(ch_val['frequency_cal'], ch_val['recency_cal'], ch_val['T_cal'])
        
        # Predict transactions for the holdout period
        ch_val['pred_holdout'] = bgf_val.predict(
            ch_val['duration_holdout'], ch_val['frequency_cal'], 
            ch_val['recency_cal'], ch_val['T_cal']
        )
        
        # Aggregate by frequency in calibration period (standard BTYD check)
        # Limit to frequency <= 10 for cleaner plot
        val_agg = (ch_val[ch_val['frequency_cal'] <= 10]
                  .groupby('frequency_cal')[['frequency_holdout', 'pred_holdout']]
                  .mean()
                  .reset_index())
        val_agg.to_parquet(cfg.CLV_VALIDATION, index=False)
        print(f"    Saved CLV validation data (holdout={cfg.CLV_HOLDOUT_DAYS} days)")

    # ── Final Model: Fit on all data for LTV predictions ─────────────────────
    # Fit BG/NBD
    bgf = BetaGeoFitter(penalizer_coef=cfg.CLV_PENALIZER)
    bgf.fit(lf['frequency'], lf['recency'], lf['T'])
    lf['pred_num_txn'] = bgf.conditional_expected_number_of_purchases_up_to_time(
        cfg.CLV_HORIZON_WKS, lf['frequency'], lf['recency'], lf['T']
    ).round(4)
    
    # NEW: Calculate Probability of Being Alive (Churn Risk)
    lf['P_Alive'] = bgf.conditional_probability_alive(
        lf['frequency'], lf['recency'], lf['T']
    ).round(4)

    # Fit Gamma-Gamma
    ggf = GammaGammaFitter(penalizer_coef=cfg.CLV_PENALIZER)
    ggf.fit(lf['frequency'], lf['monetary_value'])
    lf['pred_txn_value'] = ggf.conditional_expected_average_profit(
        lf['frequency'], lf['monetary_value']
    ).round(2)

    # Compute 12-month CLV
    # lifetimes.customer_lifetime_value expects 'time' in MONTHS.
    # We pass freq='W' to correctly handle our weekly-fitted model units.
    # discount_rate should be a MONTHLY rate (standard 1% = 0.01).
    monthly_discount = (1 + cfg.CLV_DISCOUNT_WK)**4.345 - 1
    
    lf['CLV'] = ggf.customer_lifetime_value(
        bgf,
        lf['frequency'], lf['recency'], lf['T'], lf['monetary_value'],
        time=12,
        discount_rate=monthly_discount,
        freq='W'
    ).clip(lower=0).round(2)


    # CLV tiers — percentile-based (top 20% / next 30% / bottom 50%)
    p50 = lf['CLV'].quantile(0.50)
    p80 = lf['CLV'].quantile(0.80)
    lf['CLV_Tier'] = np.where(
        lf['CLV'] >= p80, 'High CLV',
        np.where(lf['CLV'] >= p50, 'Medium CLV', 'Low CLV')
    )

    # Save models using lifetimes built-in (handles internal lambdas)
    bgf.save_model(str(cfg.BGF_MODEL))
    ggf.save_model(str(cfg.GGF_MODEL))

    # Reset index so Client is a column
    lf_out = lf.reset_index()

    # Compute metadata
    sorted_clv = lf_out['CLV'].sort_values(ascending=False)
    n20 = max(1, int(len(sorted_clv) * 0.20))
    top20_share = round(float(sorted_clv.head(n20).sum() / sorted_clv.sum() * 100), 2)

    meta = {
        'n_customers': int(len(lf_out)),
        'horizon_weeks': cfg.CLV_HORIZON_WKS,
        'discount_rate_weekly': cfg.CLV_DISCOUNT_WK,
        'spearman_corr_freq_monetary': round(float(corr), 4),
        'clv_mean': round(float(lf_out['CLV'].mean()), 2),
        'clv_median': round(float(lf_out['CLV'].median()), 2),
        'clv_p80': round(float(lf_out['CLV'].quantile(0.80)), 2),
        'top20_revenue_share': top20_share,
    }

    print(f"    Median CLV: {meta['clv_median']:,.0f} DZD")
    print(f"    Mean CLV:   {meta['clv_mean']:,.0f} DZD")
    print(f"    Top-20% revenue share: {top20_share:.1f}%")
    return lf_out, meta


# ══ SECTION 4: Forecasting ════════════════════════════════════════════════════

def build_daily_series(sales: pd.DataFrame) -> pd.DataFrame:
    """Build gap-filled daily revenue series."""
    print(" -- Building daily revenue series...")
    daily = (sales[sales['Revenue'] > 0]
             .groupby('Date_Only')['Revenue'].sum()
             .reset_index()
             .rename(columns={'Date_Only': 'ds', 'Revenue': 'y'}))
    daily['ds'] = pd.to_datetime(daily['ds'])
    daily = daily.sort_values('ds').reset_index(drop=True)

    # Fill gaps (missing days = 0 revenue)
    full_range = pd.date_range(daily['ds'].min(), daily['ds'].max(), freq='D')
    daily = (daily.set_index('ds')
             .reindex(full_range, fill_value=0)
             .reset_index()
             .rename(columns={'index': 'ds'}))
    print(f"    Series length: {len(daily)} days "
          f"({daily['ds'].min().date()} to {daily['ds'].max().date()})")
    return daily


def build_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag and rolling features to daily series. Drops NaN rows."""
    d = df.copy()
    d['dayofweek']  = d['ds'].dt.dayofweek
    d['month']      = d['ds'].dt.month
    d['weekofyear'] = d['ds'].dt.isocalendar().week.astype(int)
    d['lag7']       = d['y'].shift(7)
    d['lag14']      = d['y'].shift(14)
    d['lag28']      = d['y'].shift(28)
    d['rolling7']   = d['y'].shift(1).rolling(7).mean()
    d['rolling14']  = d['y'].shift(1).rolling(14).mean()
    return d.dropna()


def fit_sarima(train: pd.DataFrame) -> tuple:
    """Fit SARIMA via auto_arima. Returns (fitted_result, test_pred, meta)."""
    print(" -- Fitting SARIMA...")
    from pmdarima import auto_arima
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    am = auto_arima(
        train['y'].values,
        seasonal=True, m=7,
        stepwise=True,
        max_p=2, max_q=2,
        max_P=1, max_Q=1,
        information_criterion='aic',
        suppress_warnings=True,
        error_action='ignore',
    )
    print(f"    Auto-selected order: {am.order}, seasonal: {am.seasonal_order}")

    sarima_model = SARIMAX(
        train['y'].values,
        order=am.order,
        seasonal_order=am.seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    sarima_fit = sarima_model.fit(disp=False)

    meta = {
        'sarima_order': list(am.order),
        'sarima_seasonal_order': list(am.seasonal_order),
    }
    return sarima_fit, meta


def fit_prophet(train: pd.DataFrame, n_test: int) -> tuple:
    """Fit Prophet. Returns (model, test_pred)."""
    print(" -- Fitting Prophet...")
    from prophet import Prophet

    prophet_model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
    )
    prophet_model.fit(train[['ds', 'y']])
    future = prophet_model.make_future_dataframe(periods=n_test)
    prophet_fc = prophet_model.predict(future)
    prophet_pred_test = prophet_fc.tail(n_test)['yhat'].clip(lower=0).values
    print(f"    Prophet fitted, {n_test} test predictions generated")
    return prophet_model, prophet_pred_test


def fit_xgboost(train_lag, test_lag, feat_cols) -> tuple:
    """Fit XGBoost. Returns (model, test_pred)."""
    print(" -- Fitting XGBoost...")
    xgb_model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=4,
        random_state=42,
        verbosity=0,
    )
    xgb_model.fit(train_lag[feat_cols], train_lag['y'])
    xgb_pred_test = xgb_model.predict(test_lag[feat_cols])
    xgb_pred_test = np.clip(xgb_pred_test, 0, None)
    print(f"    XGBoost fitted, {len(xgb_pred_test)} test predictions generated")
    return xgb_model, xgb_pred_test


def xgb_recursive_forecast(model, history_y, feat_cols, base_date, n_days) -> list:
    """Recursive step-by-step forecast for XGBoost. Correct lag propagation."""
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


# ══ MAIN ══════════════════════════════════════════════════════════════════════

def main():
    """Run the full training pipeline."""
    start_time = datetime.datetime.now()
    print("=========================================================")
    print("  PUMA ALGERIA -- Model Training Pipeline")
    print("=========================================================\n")

    snapshot = pd.Timestamp(cfg.SNAPSHOT_DATE)

    # ── Section 0: Load & Clean ───────────────────────────────────────────
    df_raw = load_raw(cfg.DATA_FILE)
    df_clean, sales_clean = clean_data(df_raw)

    df_clean.to_parquet(cfg.DF_CLEAN, index=False)
    sales_clean.to_parquet(cfg.SALES_CLEAN, index=False)
    print(f"    Saved: {cfg.DF_CLEAN.name}, {cfg.SALES_CLEAN.name}\n")

    # ── Section 1: RFM ───────────────────────────────────────────────────
    rfm_scored = compute_rfm(sales_clean, snapshot)
    rfm_scored.to_parquet(cfg.RFM_SCORED, index=False)
    print(f"    Saved: {cfg.RFM_SCORED.name}\n")

    # ── Section 2: K-Means ───────────────────────────────────────────────
    rfm_clustered, kmeans_meta = fit_kmeans(rfm_scored, cfg.N_CLUSTERS)
    rfm_clustered.to_parquet(cfg.RFM_CLUSTERED, index=False)
    with open(cfg.KMEANS_META, 'w') as f:
        json.dump(kmeans_meta, f, indent=2)
    print(f"    Saved: {cfg.RFM_CLUSTERED.name}, kmeans_model.joblib, "
          f"kmeans_scaler.joblib, {cfg.KMEANS_META.name}\n")

    # ── Section 3: CLV ───────────────────────────────────────────────────
    clv_table, clv_meta = fit_clv(sales_clean, snapshot)
    clv_table.to_parquet(cfg.CLV_TABLE, index=False)
    with open(cfg.CLV_META, 'w') as f:
        json.dump(clv_meta, f, indent=2)
    print(f"    Saved: {cfg.CLV_TABLE.name}, bgf_model.pkl, ggf_model.pkl, "
          f"{cfg.CLV_META.name}\n")

    # ── Section 4: Forecasting ───────────────────────────────────────────
    daily = build_daily_series(sales_clean)
    daily.to_parquet(cfg.DAILY_SALES, index=False)

    # Train/test split — last 60 days
    train_end = daily['ds'].max() - pd.Timedelta(days=cfg.FORECAST_TEST_DAYS)
    train = daily[daily['ds'] <= train_end].copy()
    test  = daily[daily['ds'] > train_end].copy()
    print(f"    Train: {len(train)} days, Test: {len(test)} days")
    print(f"    Train end: {train_end.date()}, Test start: {test['ds'].min().date()}")

    # SARIMA
    sarima_fit_obj, sarima_meta = fit_sarima(train)
    sarima_pred_test = sarima_fit_obj.get_forecast(steps=len(test)).predicted_mean
    sarima_pred_test = np.clip(sarima_pred_test, 0, None)
    with open(cfg.SARIMA_FIT, 'wb') as f:
        pickle.dump(sarima_fit_obj, f)

    # Prophet
    prophet_model, prophet_pred_test = fit_prophet(train, len(test))
    with open(cfg.PROPHET_MODEL, 'wb') as f:
        pickle.dump(prophet_model, f)

    # XGBoost
    train_lag = build_lag_features(train)
    full_lag  = build_lag_features(daily)
    test_lag  = full_lag[full_lag['ds'].isin(test['ds'])].copy()
    xgb_model, xgb_pred_test = fit_xgboost(train_lag, test_lag, cfg.FEAT_COLS)
    with open(cfg.XGB_FORECAST, 'wb') as f:
        pickle.dump(xgb_model, f)

    # Evaluate
    y_true = test['y'].values
    def rmse(a, b):
        return np.sqrt(mean_squared_error(a, b))

    results = {
        'SARIMA':  {'MAE': float(mean_absolute_error(y_true, sarima_pred_test)),
                    'RMSE': float(rmse(y_true, sarima_pred_test))},
        'Prophet': {'MAE': float(mean_absolute_error(y_true, prophet_pred_test)),
                    'RMSE': float(rmse(y_true, prophet_pred_test))},
        'XGBoost': {'MAE': float(mean_absolute_error(y_true, xgb_pred_test)),
                    'RMSE': float(rmse(y_true, xgb_pred_test))},
    }
    best_model = min(results, key=lambda m: results[m]['MAE'])

    # Save forecast test parquet
    forecast_test = test[['ds', 'y']].copy()
    forecast_test['sarima_pred']  = sarima_pred_test
    forecast_test['prophet_pred'] = prophet_pred_test
    forecast_test['xgb_pred']     = xgb_pred_test
    forecast_test.to_parquet(cfg.FORECAST_TEST, index=False)

    # Save forecast meta
    forecast_meta = {
        'train_end': str(train_end.date()),
        'test_start': str(test['ds'].min().date()),
        'n_test_days': int(len(test)),
        **sarima_meta,
        'feat_cols': cfg.FEAT_COLS,
        'metrics': results,
        'best_model': best_model,
    }
    with open(cfg.FORECAST_META, 'w') as f:
        json.dump(forecast_meta, f, indent=2)

    # Save training log
    training_log = {
        'trained_at': datetime.datetime.now().isoformat(),
        'data_file': cfg.DATA_FILE.name,
        'n_rows_raw': int(len(df_raw)),
        'n_rows_sales': int(len(sales_clean)),
        'snapshot_date': cfg.SNAPSHOT_DATE,
    }
    with open(cfg.TRAINING_LOG, 'w') as f:
        json.dump(training_log, f, indent=2)

    print(f"\n    Saved: daily_sales.parquet, forecast_test.parquet, "
          f"forecast_meta.json")
    print(f"    Saved: sarima_fit.pkl, prophet_model.pkl, xgb_forecast.pkl")
    print(f"    Saved: training_log.json")

    # ── Summary ──────────────────────────────────────────────────────────
    elapsed = (datetime.datetime.now() - start_time).total_seconds()
    s_order = sarima_meta.get('sarima_order', [])
    s_sorder = sarima_meta.get('sarima_seasonal_order', [])

    print(f"""
=========================================================
  PUMA ALGERIA -- Model Training Complete
=========================================================
  Customers (RFM scope)  : {len(rfm_scored):,}
  K-Means k={cfg.N_CLUSTERS}            : Silhouette={kmeans_meta['silhouette']:.3f}  DB={kmeans_meta['davies_bouldin']:.3f}
  CLV customers          : {clv_meta['n_customers']:,}  |  Median CLV: {clv_meta['clv_median']:,.0f} DZD
  SARIMA order           : {tuple(s_order)}{tuple(s_sorder)}
  Forecast MAE           : SARIMA={results['SARIMA']['MAE']:,.0f}  Prophet={results['Prophet']['MAE']:,.0f}  XGBoost={results['XGBoost']['MAE']:,.0f}
  Best model             : {best_model}
  Artifacts saved to     : ./artifacts/
  Elapsed time           : {elapsed:.0f}s
=========================================================""")


if __name__ == '__main__':
    main()
