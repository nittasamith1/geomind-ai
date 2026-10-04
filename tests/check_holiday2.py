import pandas as pd

train = pd.read_csv('data/processed/train.csv')
print("=== Holiday column sample ===")
print(train['holiday'].value_counts(dropna=False).head(20))
print()
print("First 20 rows:")
print(train[['date_time','holiday']].head(20).to_string())
print()
# What does NaN != 'None' evaluate to?
import numpy as np
nan_val = float('nan')
print(f"nan != 'None': {nan_val != 'None'}")
print(f"pd.isna NaN: {pd.isna(nan_val)}")
print()
# Check the raw data ingestion  
raw_files = ['data/raw/Metro_Interstate_Traffic_Volume.csv']
import os
for f in raw_files:
    if os.path.exists(f):
        raw = pd.read_csv(f)
        print(f"=== Raw data: {f} ===")
        print(raw['holiday'].value_counts(dropna=False).head(10))
        break
