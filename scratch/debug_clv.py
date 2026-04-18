import pandas as pd
import numpy as np
from pathlib import Path

clv_path = Path("artifacts/data/clv_table.parquet")
if clv_path.exists():
    df = pd.read_parquet(clv_path)
    # Check the specific customers from the screenshot
    target_ids = ['6010007501', '0580003952', '0800006539']
    targets = df[df['Client'].isin(target_ids)]
    
    print("--- TARGET CUSTOMERS ---")
    print(targets[['Client', 'frequency', 'recency', 'T', 'monetary_value', 'P_Alive', 'CLV']])
    
    print("\n--- MONETARY STATS ---")
    print(df['monetary_value'].describe(percentiles=[0.5, 0.9, 0.95, 0.99]))
    
    print("\n--- P_ALIVE vs CLV ---")
    # See how many churned customers have high CLV
    churned_high_clv = df[(df['P_Alive'] < 0.05) & (df['CLV'] > 50000)]
    print(f"Count of customers with P_Alive < 5% and CLV > 50k: {len(churned_high_clv)}")
    if len(churned_high_clv) > 0:
        print(churned_high_clv[['Client', 'monetary_value', 'P_Alive', 'CLV']].head(10))
else:
    print("CLV table not found at", clv_path)
