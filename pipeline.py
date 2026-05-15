"""
pipeline.py — Self-contained in-memory analytics pipeline.

Called by app.py when the user uploads the raw Excel file via Streamlit.
No disk I/O — every artifact is returned in a plain dict and stored in
st.session_state.  train_models.py (local script) still works independently
and saves artifacts to disk for local development.
"""

import warnings
warnings.filterwarnings("ignore")

import io
import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                              mean_absolute_error, mean_squared_error)
from xgboost import XGBRegressor

import config as cfg


# ══ Helpers ══════════════════════════════════════════════════════════════════

def _clean_currency(value):
    if isinstance(value, str):
        s = "".join(ch for ch in value if ch.isdigit() or ch in ".-")
        return s if s not in ("", "-", ".") else np.nan
    return value


def _classify_client(cid):
    c = str(cid).strip().upper()
    if c == 'SC000004':
        return 'Passage'
    elif c.startswith('P'):
        return 'Personnel'
    elif c.startswith('AMB'):
        return 'Partner'
    else:
        return 'Fidélité'


def _rfm_segment(row):
    r, f = row['R_Score'], row['F_Score']
    if r >= 4 and f >= 4:
        return 'Champions'
    elif r >= 4 and f >= 3:
        return 'Loyal Customers'
    elif r >= 4 and f <= 2:
        return 'New Customers'
    elif (r == 3 or r == 2) and f >= 2:
        return 'At Risk'
    elif r == 1 and f >= 3:
        return 'At Risk'
    else:
        return 'Lost'


# ══ Step 0: Load & Clean ══════════════════════════════════════════════════════

def load_and_clean(file_bytes: bytes):
    """Parse raw Excel bytes, apply all cleaning steps.
    Returns (df_clean, sales_clean)."""
    df = pd.read_excel(io.BytesIO(file_bytes))
    df = df.copy()

    df.drop(columns=cfg.COLS_TO_DROP, errors='ignore', inplace=True)

    df['Heure de création'] = pd.to_datetime(df['Heure de création'], errors='coerce')
    df['Year']       = df['Heure de création'].dt.year
    df['Month']      = df['Heure de création'].dt.month
    df['Week']       = df['Heure de création'].dt.isocalendar().week.astype(int)
    df['Hour']       = df['Heure de création'].dt.hour
    df['Day_Name']   = df['Heure de création'].dt.day_name()
    df['Month_Name'] = df['Heure de création'].dt.strftime('%B')
    df['Quarter']    = df['Heure de création'].dt.quarter
    df['Date_Only']  = df['Heure de création'].dt.date

    ttc_col = 'Total TTC ligne'
    if ttc_col in df.columns:
        if df[ttc_col].dtype == object:
            df[ttc_col] = df[ttc_col].apply(_clean_currency)
        df[ttc_col] = pd.to_numeric(df[ttc_col], errors='coerce')
    df.rename(columns={ttc_col: 'Revenue'}, inplace=True)

    if 'Quantité' in df.columns:
        df.rename(columns={'Quantité': 'Qty'}, inplace=True)
    df['Is_Return'] = df['Qty'] < 0

    df.rename(columns=cfg.RENAME_MAP, inplace=True)
    df['Client_Type'] = df['Client'].apply(_classify_client)

    if 'Article_Code' in df.columns:
        for col in ['Category', 'Season', 'Age_Cat']:
            if col in df.columns:
                df[col] = df.groupby('Article_Code')[col].transform(
                    lambda x: x.ffill().bfill())

    df = df[df['Revenue'].notna() & df['Date'].notna()].copy()
    sales = df[(df['Is_Return'] == False) & (df['Revenue'] > 0)].copy()
    return df, sales


# ══ Step 1: RFM ═══════════════════════════════════════════════════════════════

def compute_rfm(sales: pd.DataFrame, snapshot: pd.Timestamp) -> pd.DataFrame:
    rfm_customers = sales[sales['Client_Type'].isin(
        ['Fidélité', 'Personnel', 'Partner'])].copy()

    rfm = rfm_customers.groupby('Client').agg(
        Recency   = ('Date', lambda x: (snapshot - x.max()).days),
        Frequency = ('Date', lambda x: x.dt.date.nunique()),
        Monetary  = ('Revenue', 'sum'),
    ).reset_index()

    for grp_col in ['Category', 'Age_Cat', 'Breakout', 'Dimension']:
        if grp_col in sales.columns:
            pref = sales.groupby(['Client', grp_col])['Revenue'].sum().reset_index()
            top  = (pref.sort_values(['Client', 'Revenue'], ascending=[True, False])
                    .drop_duplicates('Client'))
            rfm  = rfm.merge(top[['Client', grp_col]], on='Client', how='left')

    rfm = rfm[rfm['Monetary'] > 0].copy()

    rfm['R_Score'] = pd.cut(rfm['Recency'],
        bins=[-1, 60, 120, 210, 270, 99999], labels=[5, 4, 3, 2, 1]).astype(int)
    rfm['F_Score'] = pd.cut(rfm['Frequency'],
        bins=[0, 1, 2, 3, 5, 99999], labels=[1, 2, 3, 4, 5]).astype(int)
    rfm['M_Score'] = pd.qcut(rfm['Monetary'].rank(method='first'),
        q=[0, 0.3, 0.5, 0.7, 0.9, 1.0], labels=[1, 2, 3, 4, 5]).astype(int)

    rfm['RFM_Score'] = rfm[['R_Score', 'F_Score', 'M_Score']].sum(axis=1)
    rfm['RFM_Code']  = (rfm['R_Score'].astype(str)
                        + rfm['F_Score'].astype(str)
                        + rfm['M_Score'].astype(str))
    rfm['Segment']   = rfm.apply(_rfm_segment, axis=1)
    return rfm


# ══ Step 2: K-Means ═══════════════════════════════════════════════════════════

def fit_kmeans(rfm: pd.DataFrame):
    feats     = rfm[['Recency', 'Frequency', 'Monetary']].copy()
    feats_log = np.log1p(feats)
    scaler    = StandardScaler()
    X         = scaler.fit_transform(feats_log)

    sweep_meta = {}
    for k_c in range(2, 7):
        km_c   = KMeans(n_clusters=k_c, random_state=42, n_init=10)
        lbl_c  = km_c.fit_predict(X)
        sweep_meta[k_c] = {
            'silhouette':    round(float(silhouette_score(X, lbl_c)), 4),
            'davies_bouldin': round(float(davies_bouldin_score(X, lbl_c)), 4),
            'inertia':       round(float(km_c.inertia_), 1),
        }

    km     = KMeans(n_clusters=cfg.N_CLUSTERS, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    sil    = silhouette_score(X, labels)
    db     = davies_bouldin_score(X, labels)

    rfm_out = rfm.copy()
    rfm_out['KMeans_Cluster'] = labels
    means   = (rfm_out.groupby('KMeans_Cluster')['RFM_Score']
               .mean().sort_values(ascending=False))
    c2seg   = {c: name for c, name in zip(means.index, cfg.KMEANS_SEGMENT_NAMES)}
    rfm_out['KMeans_Segment'] = rfm_out['KMeans_Cluster'].map(c2seg)

    meta = {
        'silhouette':         round(float(sil), 4),
        'davies_bouldin':     round(float(db), 4),
        'k':                  cfg.N_CLUSTERS,
        'segment_names':      cfg.KMEANS_SEGMENT_NAMES,
        'cluster_to_segment': {int(k_): v for k_, v in c2seg.items()},
        'k_sweep':            sweep_meta,
    }
    return rfm_out, meta


# ══ Step 3: CLV — BG/NBD + Gamma-Gamma ═══════════════════════════════════════

def fit_clv(sales: pd.DataFrame, snapshot: pd.Timestamp):
    from lifetimes import BetaGeoFitter, GammaGammaFitter
    from lifetimes.utils import (summary_data_from_transaction_data,
                                  calibration_and_holdout_data)

    cust   = sales[sales['Client_Type'].isin(
        ['Fidélité', 'Personnel', 'Partner'])].copy()
    lt_row = cust[cust['Revenue'] > 0][['Client', 'Date', 'Revenue']].copy()
    lt_df  = lt_row.groupby(['Client', 'Date'])['Revenue'].sum().reset_index()

    lf = summary_data_from_transaction_data(
        lt_df, customer_id_col='Client', datetime_col='Date',
        monetary_value_col='Revenue', observation_period_end=snapshot, freq='W')

    if not lf.empty:
        up_limit = lf['monetary_value'].quantile(0.99)
        lf['monetary_value'] = lf['monetary_value'].clip(upper=up_limit)

    lf   = lf[lf['frequency'] > 0].copy()
    corr = lf['frequency'].corr(lf['monetary_value'], method='spearman')

    # Holdout validation
    cal_end    = snapshot - pd.Timedelta(days=cfg.CLV_HOLDOUT_DAYS)
    ch         = calibration_and_holdout_data(lt_df, 'Client', 'Date',
                     calibration_period_end=cal_end,
                     observation_period_end=snapshot, freq='W')
    ch_val     = ch[ch['frequency_cal'] > 0].copy()
    clv_validation = None
    if len(ch_val) > 0:
        bgf_val = BetaGeoFitter(penalizer_coef=cfg.CLV_PENALIZER)
        bgf_val.fit(ch_val['frequency_cal'], ch_val['recency_cal'], ch_val['T_cal'])
        ch_val['pred_holdout'] = bgf_val.predict(
            ch_val['duration_holdout'], ch_val['frequency_cal'],
            ch_val['recency_cal'], ch_val['T_cal'])
        clv_validation = (ch_val[ch_val['frequency_cal'] <= 10]
                          .groupby('frequency_cal')[['frequency_holdout', 'pred_holdout']]
                          .mean().reset_index())

    # Full-data models
    bgf = BetaGeoFitter(penalizer_coef=cfg.CLV_PENALIZER)
    bgf.fit(lf['frequency'], lf['recency'], lf['T'])
    lf['pred_num_txn'] = bgf.conditional_expected_number_of_purchases_up_to_time(
        cfg.CLV_HORIZON_WKS, lf['frequency'], lf['recency'], lf['T']).round(4)
    lf['P_Alive'] = bgf.conditional_probability_alive(
        lf['frequency'], lf['recency'], lf['T']).round(4)

    ggf = GammaGammaFitter(penalizer_coef=cfg.CLV_PENALIZER)
    ggf.fit(lf['frequency'], lf['monetary_value'])
    lf['pred_txn_value'] = ggf.conditional_expected_average_profit(
        lf['frequency'], lf['monetary_value']).round(2)

    monthly_discount = (1 + cfg.CLV_DISCOUNT_WK) ** 4.345 - 1
    lf['CLV'] = ggf.customer_lifetime_value(
        bgf, lf['frequency'], lf['recency'], lf['T'], lf['monetary_value'],
        time=12, discount_rate=monthly_discount, freq='W').clip(lower=0).round(2)

    p50 = lf['CLV'].quantile(0.50)
    p80 = lf['CLV'].quantile(0.80)
    lf['CLV_Tier'] = np.where(lf['CLV'] >= p80, 'High CLV',
                     np.where(lf['CLV'] >= p50, 'Medium CLV', 'Low CLV'))
    lf_out = lf.reset_index()

    sorted_clv  = lf_out['CLV'].sort_values(ascending=False)
    n20         = max(1, int(len(sorted_clv) * 0.20))
    top20_share = round(float(sorted_clv.head(n20).sum() / sorted_clv.sum() * 100), 2)

    meta = {
        'n_customers':                  int(len(lf_out)),
        'horizon_weeks':                cfg.CLV_HORIZON_WKS,
        'discount_rate_weekly':         cfg.CLV_DISCOUNT_WK,
        'spearman_corr_freq_monetary':  round(float(corr), 4),
        'clv_mean':                     round(float(lf_out['CLV'].mean()), 2),
        'clv_median':                   round(float(lf_out['CLV'].median()), 2),
        'clv_p80':                      round(float(lf_out['CLV'].quantile(0.80)), 2),
        'top20_revenue_share':          top20_share,
    }

    if clv_validation is None:
        clv_validation = pd.DataFrame(
            columns=['frequency_cal', 'frequency_holdout', 'pred_holdout'])

    return lf_out, bgf, clv_validation, meta


# ══ Step 4: Forecasting ═══════════════════════════════════════════════════════

def _build_daily_series(sales: pd.DataFrame) -> pd.DataFrame:
    daily = (sales[sales['Revenue'] > 0]
             .groupby('Date_Only')['Revenue'].sum()
             .reset_index()
             .rename(columns={'Date_Only': 'ds', 'Revenue': 'y'}))
    daily['ds'] = pd.to_datetime(daily['ds'])
    daily = daily.sort_values('ds').reset_index(drop=True)
    full_range = pd.date_range(daily['ds'].min(), daily['ds'].max(), freq='D')
    daily = (daily.set_index('ds')
             .reindex(full_range, fill_value=0)
             .reset_index()
             .rename(columns={'index': 'ds'}))
    return daily


def _build_lag_features(df: pd.DataFrame) -> pd.DataFrame:
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


def fit_forecasting_models(sales: pd.DataFrame):
    from pmdarima import auto_arima
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from prophet import Prophet
    from sklearn.ensemble import RandomForestRegressor

    daily     = _build_daily_series(sales)
    n_test    = cfg.FORECAST_TEST_DAYS
    train_end = daily['ds'].max() - pd.Timedelta(days=n_test)
    train     = daily[daily['ds'] <= train_end].copy()
    test      = daily[daily['ds'] > train_end].copy()

    # SARIMA
    am = auto_arima(
        train['y'].values, seasonal=True, m=7, stepwise=True,
        max_p=2, max_q=2, max_P=1, max_Q=1,
        information_criterion='aic',
        suppress_warnings=True, error_action='ignore')
    sarima_model = SARIMAX(train['y'].values, order=am.order,
                           seasonal_order=am.seasonal_order,
                           enforce_stationarity=False, enforce_invertibility=False)
    sarima_fit        = sarima_model.fit(disp=False)
    sarima_pred_test  = np.clip(
        sarima_fit.get_forecast(steps=len(test)).predicted_mean, 0, None)

    # Prophet
    prophet_model = Prophet(yearly_seasonality=True, weekly_seasonality=True,
                            daily_seasonality=False, changepoint_prior_scale=0.05)
    prophet_model.fit(train[['ds', 'y']])
    future            = prophet_model.make_future_dataframe(periods=len(test))
    prophet_pred_test = (prophet_model.predict(future)
                         .tail(len(test))['yhat'].clip(lower=0).values)

    # XGBoost + Random Forest
    train_lag  = _build_lag_features(train)
    full_lag   = _build_lag_features(daily)
    test_lag   = full_lag[full_lag['ds'].isin(test['ds'])].copy()
    feat_cols  = cfg.FEAT_COLS

    xgb = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=4,
                       random_state=42, verbosity=0)
    xgb.fit(train_lag[feat_cols], train_lag['y'])
    xgb_pred_test = np.clip(xgb.predict(test_lag[feat_cols]), 0, None)

    rf = RandomForestRegressor(n_estimators=300, max_features='sqrt',
                               min_samples_leaf=2, random_state=42)
    rf.fit(train_lag[feat_cols], train_lag['y'])
    rf_pred_test = np.clip(rf.predict(test_lag[feat_cols]), 0, None)

    # Evaluate all models
    y_true = test['y'].values
    def _rmse(a, b): return float(np.sqrt(mean_squared_error(a, b)))

    results = {
        'SARIMA':  {'MAE': float(mean_absolute_error(y_true, sarima_pred_test)),
                    'RMSE': _rmse(y_true, sarima_pred_test)},
        'Prophet': {'MAE': float(mean_absolute_error(y_true, prophet_pred_test)),
                    'RMSE': _rmse(y_true, prophet_pred_test)},
        'XGBoost': {'MAE': float(mean_absolute_error(y_true, xgb_pred_test)),
                    'RMSE': _rmse(y_true, xgb_pred_test)},
        'RF':      {'MAE': float(mean_absolute_error(y_true, rf_pred_test)),
                    'RMSE': _rmse(y_true, rf_pred_test)},
    }
    best_model = min(results, key=lambda m: results[m]['MAE'])

    forecast_test = test[['ds', 'y']].copy()
    forecast_test['sarima_pred']  = sarima_pred_test
    forecast_test['prophet_pred'] = prophet_pred_test
    forecast_test['xgb_pred']     = xgb_pred_test
    forecast_test['rf_pred']      = rf_pred_test

    forecast_meta = {
        'train_end':            str(train_end.date()),
        'test_start':           str(test['ds'].min().date()),
        'n_test_days':          int(len(test)),
        'sarima_order':         list(am.order),
        'sarima_seasonal_order': list(am.seasonal_order),
        'feat_cols':            feat_cols,
        'metrics':              results,
        'best_model':           best_model,
    }

    return daily, forecast_test, sarima_fit, prophet_model, xgb, rf, forecast_meta


# ══ Master entry point ════════════════════════════════════════════════════════

def run_full_pipeline(file_bytes: bytes, on_step=None) -> dict:
    """
    Run the complete pipeline from raw Excel bytes.

    Parameters
    ----------
    file_bytes : bytes
        Raw content of the uploaded .xlsx file.
    on_step : callable(label: str, pct: float) | None
        Progress callback — called after each major step.

    Returns
    -------
    dict
        All dashboard artifacts in memory (same keys as disk-based artifacts).
    """

    def _step(label, pct):
        if on_step:
            on_step(label, pct)

    snapshot = pd.Timestamp(cfg.SNAPSHOT_DATE)

    _step("Loading and cleaning data…", 0.05)
    df_clean, sales_clean = load_and_clean(file_bytes)

    _step("Computing RFM scores…", 0.18)
    rfm_scored = compute_rfm(sales_clean, snapshot)

    _step("Running K-Means clustering (k=2…6 sweep)…", 0.32)
    rfm_clustered, kmeans_meta = fit_kmeans(rfm_scored)

    _step("Fitting BG/NBD + Gamma-Gamma CLV models…", 0.48)
    clv_table, bgf_model, clv_validation, clv_meta = fit_clv(sales_clean, snapshot)

    _step("Fitting SARIMA (auto_arima search)…", 0.62)
    _step("Fitting Prophet, XGBoost, Random Forest…", 0.75)
    (daily_sales, forecast_test, sarima_fit,
     prophet_model, xgb_model, rf_model,
     forecast_meta) = fit_forecasting_models(sales_clean)

    _step("Wrapping up…", 0.95)
    training_log = {
        'trained_at':    datetime.datetime.now().isoformat(),
        'data_file':     'uploaded_file.xlsx',
        'n_rows_raw':    int(len(df_clean)),
        'n_rows_sales':  int(len(sales_clean)),
        'snapshot_date': cfg.SNAPSHOT_DATE,
    }

    _step("Done!", 1.0)

    return {
        'df_clean':      df_clean,
        'sales_clean':   sales_clean,
        'rfm_scored':    rfm_scored,
        'rfm_clustered': rfm_clustered,
        'clv_table':     clv_table,
        'daily_sales':   daily_sales,
        'forecast_test': forecast_test,
        'clv_validation': clv_validation,
        'kmeans_meta':   kmeans_meta,
        'clv_meta':      clv_meta,
        'forecast_meta': forecast_meta,
        'training_log':  training_log,
        'bgf_model':     bgf_model,
        'sarima_fit':    sarima_fit,
        'prophet_model': prophet_model,
        'xgb_model':     xgb_model,
        'rf_model':      rf_model,
    }
