import pandas as pd
import numpy as np

# 1. Load your master dataset (Make sure the filename matches yours!)
data = pd.read_csv('/Users/francescamaltoni/Desktop/ML-Project-2026/data/processed/eth_merged_6h_2021_to_latest.csv')

# 2. Make absolutely sure the data is sorted by time from oldest to newest
data['hour'] = pd.to_datetime(data['hour'])
data.set_index('hour', inplace=True)
data = data.sort_index()

# 3. Define where to draw the line (80% Training, 20% Testing)
split_percent = 0.80
split_index = int(len(data) * split_percent)

# 4. Perform the Chronological Split
# .iloc slices the dataframe based on the row numbers
train_df = data.iloc[:split_index].copy()
test_df = data.iloc[split_index:].copy()

print("--- Data Split Successful! ---")
print(f"Total Rows: {len(data)}")
print(f"Training Data (The Past): {len(train_df)} rows")
print(f"Testing Data (The Future): {len(test_df)} rows")

print("\nTraining set timeline:")
print(f"Starts: {train_df.index.min()} | Ends: {train_df.index.max()}")
print("\nTesting set timeline:")
print(f"Starts: {test_df.index.min()} | Ends: {test_df.index.max()}")