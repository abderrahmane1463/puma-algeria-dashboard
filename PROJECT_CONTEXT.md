# Project Context — PUMA Algeria CRM Dashboard

## Business Context

This is the analytics deliverable for a Master's thesis in Statistics and Data Science:
**"AI-Driven Customer Intelligence and Demand Forecasting for Retail Decision Support"**
Case study: SARL Great Way, the official PUMA distributor in Algeria.

The dashboard is presented to company management. It must be professional, correct, and explainable.

---

## Dataset

**File**: `VENTES_PUMA_2025.xlsx`
**Rows**: ~333,000 transactions
**Period**: January 2025 – January 2026
**Stores**: 13 branches across Algeria

### Raw Column Schema (18 columns)

| Raw Column Name | Type | Description |
|----------------|------|-------------|
| `Date création doc.` | datetime | Document creation date — **DROP** (less precise than Heure) |
| `Etablissement` | str | Store/branch name → rename to `Store` |
| `Code article` | int | Product code → rename to `Article_Code` |
| `Libellé article` | str | Product name → rename to `Product` |
| `Quantité` | int | Units sold (negative = return) → rename to `Qty` |
| `Total TTC ligne` | str | Line revenue, French-formatted string (e.g. "11 950") → clean & rename to `Revenue` |
| `Client` | str | Customer ID (SC000004=walk-in, P*=employee, AMB*=partner, else=loyal) |
| `Prix unitaire TTC net ligne` | int | Unit price — **DROP** |
| `Category ligne doc` | str | Product category → rename to `Category` |
| `Brand ligne doc` | str | Brand (all PUMA in this dataset) |
| `Category Age ligne doc` | str | Age category (Men/Women/Kids) → rename to `Age_Cat` |
| `Saison` | str | Season (Spring-Summer / Fall-Winter) → rename to `Season` |
| `Heure de création` | datetime | **PRIMARY DATE COLUMN** — full timestamp with hour → rename to `Date` |
| `Etablissement du doc.` | int | Internal document establishment ID — **DROP** |
| `Breakout ligne doc` | str | Size/variant breakout |
| `LIBDIM1` | str | Size label (S/M/L/XL etc.) |
| `Collection ligne doc` | str | Collection name |
| `Désignation ligne` | str | Long product designation — **DROP** |

Note: `'Unnamed: 18'` does NOT exist in this file (exactly 18 columns). Drop attempt is safe but unnecessary.

### Key Data Properties

- `Total TTC ligne` arrives as a **string with French thousand-separators** (spaces, not commas): `"11 950"`, `"4 500"`. Strip all non-digit/non-period/non-minus chars before converting to float.
- `'Date création doc.'` is document date only (no time). `'Heure de création'` has the full timestamp and is the correct date column.
- Negative `Quantité` = customer return. These rows have negative `Total TTC ligne` too.
- `SC000004` is the generic anonymous cashier code for walk-in purchases (majority of transactions). These customers have no trackable identity across visits and must be **excluded from RFM and CLV analysis**.

---

## Client Segmentation Rules

```
Client ID = 'SC000004'    → Walk-in       (anonymous, excluded from RFM/CLV)
Client ID starts with 'P' → Employee      (included in RFM/CLV)
Client ID starts with 'AMB' → Partner (CRB) (included in RFM/CLV)
All other unique IDs       → Loyal (Fidelity) (included in RFM/CLV)
```

---

## Snapshot Date

**Fixed**: `2026-01-02`

This is the reference date for all RFM recency calculations. It must never be derived dynamically from `data.max()` to ensure reproducibility across dashboard runs.

---

| Segment | Logic (Priority Order) | Business Meaning |
|---------|------|-----------------|
| **Champions** | R (4-5) AND F (4-5) | Recent elite, high value |
| **Loyal Customers** | (R 3-5 AND F 3-5) OR (R 4-5 AND M 4-5) | Reliable core or recent big spenders |
| **New Customers** | R (4-5) AND F (1-3) | Recent trialists |
| **At Risk** | R (2-3) | Customers starting to drift away |
| **Lost** | R (1) | Ancient churn (no purchase in 9+ months) |

**Color palette (SEGMENT_COLORS_5)**:
```
Champions       → #2ecc71  (green)
Loyal Customers → #3498db  (blue)
New Customers   → #f1c40f  (yellow)
At Risk         → #e67e22  (orange)
Lost            → #e74c3c  (red)
```

---

## K-Means Segment Definitions (4 Segments)

Clusters are named by ranking cluster mean RFM_Score descending:

| Rank | Name | Business Meaning |
|------|------|-----------------|
| 1st (highest RFM) | Champions | Best customers |
| 2nd | At Risk | Good customers, showing churn signals |
| 3rd | Promising | Developing, not yet loyal |
| 4th (lowest RFM) | Dormant | Inactive / lost |

**Color palette (KMEANS_COLORS)**:
```
Champions → #2ecc71  (green)
At Risk   → #e67e22  (orange)
Promising → #3498db  (blue)
Dormant   → #e74c3c  (red)
```

This ordering and palette are **fixed**. Do not change them.

---

## CLV Model Specification

- **Model**: BG/NBD (transaction frequency) + Gamma-Gamma (revenue per transaction)
- **Library**: `lifetimes`
- **Customer scope**: Identifiable customers only (no Walk-in), positive revenue, `frequency > 0`
- **Time unit**: Weekly (`freq='W'` in `summary_data_from_transaction_data`)
- **Horizon**: 52 weeks = 12 months
- **Discount rate**: 0.0023 per week (weekly equivalent of 1% per month)
  - Derivation: `(1.01)^(1/4.333) - 1 ≈ 0.0023`
- **Penalizer**: 0.01 for both BG/NBD and Gamma-Gamma (regularization)
- **Gamma-Gamma assumption check**: log a warning if `Spearman(freq, monetary) > 0.30`

**CLV Tiers**:
```
High CLV   → top tercile    → #2ecc71
Medium CLV → middle tercile → #f39c12
Low CLV    → bottom tercile → #e74c3c
```

---

## Forecasting Model Specification

### Series
- Unit: daily total revenue (DZD)
- Fill zero for days with no sales (store closures, holidays)
- **Train/test split**: last 60 days = test, everything before = train

### SARIMA
- Auto-selection via `pmdarima.auto_arima`
- Seasonal: `m=7` (weekly pattern)
- Search bounds: `max_p=2, max_q=2, max_P=1, max_Q=1` (fast baseline)
- Clip negative predictions to 0

### Prophet
- Yearly + weekly seasonality, no daily
- `changepoint_prior_scale=0.05` (conservative)
- Clip negative predictions to 0

### XGBoost
- Lag features: `lag7`, `lag14`, `lag28`, `rolling7`, `rolling14`
- Calendar features: `dayofweek`, `month`, `weekofyear`
- Baseline params: `n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42`
- **Forward forecast must use recursive prediction** (extend history one step at a time, not tile)
- Clip negative predictions to 0

### Primary metric: MAE (not MAPE — MAPE is unstable when daily revenue near zero)

---

## Known Bugs Fixed in This Refactor

| Bug | Location | Fix Applied |
|-----|---------|-------------|
| SARIMA replaced by LinearRegression | `puma_dashb.py:430` | Real SARIMA via pmdarima + statsmodels |
| Prophet replaced by lag-7 naive copy | `puma_dashb.py:434` | Real Prophet fit |
| CLV `time=12` = 12 days not 12 months | `puma_dashb.py:347` | `freq='W'`, `time=52` |
| K-Means names: "Promising" ranked 2nd | `puma_dashb.py:287` | Correct order: Champions→At Risk→Promising→Dormant |
| Forward forecast uses `np.tile` for lags | `puma_dashb.py:1245` | Recursive XGBoost forecast |
| `penalizer_coef=0.0` — no regularization | `puma_dashb.py:325,336` | `penalizer_coef=0.01` |
| Dynamic snapshot date (not reproducible) | `puma_dashb.py:241` | Fixed `SNAPSHOT_DATE = '2026-01-02'` |
| No method label when fallback CLV used | dashboard | Dashboard always uses BG/NBD (no fallback needed — trained artifact loaded) |
| `pred_num_txn` window = 10 (unexplained) | `puma_dashb.py:330` | Changed to 52 (weeks = 12-month horizon) |

---

## Dependencies

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

All packages are pip-installable. Prophet requires `pystan` or `cmdstanpy` as a backend — ensure one is installed.

---

## Thesis Methodology Alignment

The dashboard is a visual companion to the notebook (`puma_rfm_clv.py`). The methodologies must match exactly:

| Notebook Section | Dashboard Tab | Method |
|-----------------|--------------|--------|
| Section 2 | Tab 1 — EDA | Descriptive statistics |
| Section 3 | Tab 2 — RFM | **Business-Threshold scored RFM**, 5 rule-based segments |
| Sections 4–5 | Tab 3 — K-Means | k=4, log1p(F,M), StandardScaler |
| Section 6A–C | Tab 4 — CLV | BG/NBD + Gamma-Gamma, 12-month horizon |
| Section 7 | Tab 5 — Forecasting | SARIMA + Prophet + XGBoost |
