import pandas as pd

df = pd.read_parquet('./artifacts/data/rfm_scored.parquet')

print("=== SEGMENT COUNTS ===")
print(df['Segment'].value_counts())

print("\n=== R_Score x F_Score CROSSTAB (customer count) ===")
print(pd.crosstab(df['R_Score'], df['F_Score'], margins=True))

print("\n=== SEGMENT per R/F COMBINATION (dominant segment) ===")
grp = df.groupby(['R_Score','F_Score'])['Segment'].agg(lambda x: x.value_counts().index[0])
print(grp.unstack())

print("\n=== CASES: R=3, F=2, M=5 (currently Lost?) ===")
s = df[(df['R_Score']==3) & (df['F_Score']==2) & (df['M_Score']==5)]
print(s[['Recency','Frequency','Monetary','R_Score','F_Score','M_Score','Segment']].head(5))

print("\n=== CASES: R=2, F=2, M=5 (currently?) ===")
s = df[(df['R_Score']==2) & (df['F_Score']==2) & (df['M_Score']==5)]
print(s[['Recency','Frequency','Monetary','R_Score','F_Score','M_Score','Segment']].head(5))

print("\n=== MONETARY thresholds ===")
for q, label in [(0.3,'p30'),(0.5,'p50'),(0.7,'p70'),(0.9,'p90')]:
    val = df['Monetary'].quantile(q)
    print(f"  {label}: {val:,.0f} DZD")

print("\n=== AVERAGE MONETARY by Segment ===")
print(df.groupby('Segment')['Monetary'].agg(['mean','min','max']).round(0).astype(int))

print("\n=== AVERAGE RECENCY by Segment ===")
print(df.groupby('Segment')['Recency'].agg(['mean','min','max']).round(0).astype(int))

print("\n=== PROBLEM CASES: High M_Score but labeled Lost ===")
prob = df[(df['Segment']=='Lost') & (df['M_Score']>=4)]
print(f"Count: {len(prob)}")
print(prob.groupby(['R_Score','F_Score','M_Score']).size().sort_values(ascending=False).head(10))
