import pandas as pd
from pathlib import Path

clv_path = Path("artifacts/data/clv_table.parquet")
if clv_path.exists():
    df = pd.read_parquet(clv_path)
    # Check the top customers
    top = df.nlargest(10, 'CLV')
    
    print("--- TOP CUSTOMERS DEBUG ---")
    # Add a sanity check column: pred_num_txn * pred_txn_value
    top['Expected_Revenue_1Y'] = top['pred_num_txn'] * top['pred_txn_value']
    top['Ratio'] = top['CLV'] / top['Expected_Revenue_1Y']
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(top[['Client', 'frequency', 'monetary_value', 'pred_num_txn', 'pred_txn_value', 'CLV', 'Expected_Revenue_1Y', 'Ratio']])
    
    print("\nIf Ratio is ~4.33, the model is predicting for 52 MONTHS instead of 52 WEEKS.")
    print("If Ratio is ~1.0, the model is correct.")
else:
    print("CLV table not found.")
