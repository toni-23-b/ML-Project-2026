import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Define your file paths (Input = Original, Output = New Clustered File)
input_file  = 'data/processed/eth_merged_6h_2017_to_latest.csv'
output_file = 'data/processed/eth_merged_6h_clustered_2017_to_latest.csv'

# 2. Load the original dataset
print(f"Loading data from: {input_file}")
data = pd.read_csv(input_file)
data['hour'] = pd.to_datetime(data['hour'])
data.set_index('hour', inplace=True)
data = data.sort_index()

# 3. Split the data chronologically (80% Train, 20% Test)
split_index = int(len(data) * 0.80)
train_df = data.iloc[:split_index].copy()
test_df = data.iloc[split_index:].copy()

# --- THE OUTLIER FIX (Clipping) ---
print("Squashing extreme outliers...")
# Find the 99.9th percentile of gas prices (the normal ceiling)
gas_ceiling = train_df['max_gas_gwei'].quantile(0.999)

# Cap both datasets so nothing goes above that ceiling
train_df['max_gas_gwei'] = train_df['max_gas_gwei'].clip(upper=gas_ceiling)
test_df['max_gas_gwei'] = test_df['max_gas_gwei'].clip(upper=gas_ceiling)

# 4. Perform K-Means Clustering on the 5 specific features
cluster_features = [
    'massive_whale_volume', 
    'max_gas_gwei', 
    'unique_large_senders', 
    'whale_contract_calls',
    'total_network_volume'
]

scaler = StandardScaler()
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)

# Train the AI on the past, let it predict the future
X_train_scaled = scaler.fit_transform(train_df[cluster_features])
train_df['Market_Regime'] = kmeans.fit_predict(X_train_scaled)

X_test_scaled = scaler.transform(test_df[cluster_features])
test_df['Market_Regime'] = kmeans.predict(X_test_scaled)

# 5. Save the new dataset (Recombine and Export)
final_df = pd.concat([train_df, test_df])
final_df.to_csv(output_file)
print(f"Success! Clustered dataset saved to: {output_file}")

# 6. Plot the Training Results
print("Generating graph...")
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=train_df, 
    x='massive_whale_volume', 
    y='max_gas_gwei',
    hue='Market_Regime', 
    palette='viridis', 
    alpha=0.7, 
    edgecolor='k', 
    s=60
)

plt.title('K-Means Market Regimes (Training Data)')
plt.xlabel('Massive Whale Volume (ETH)')
plt.ylabel('Max Gas Paid (Gwei)')
plt.legend(title='Market Regime')
plt.tight_layout()
os.makedirs('results', exist_ok=True)
plt.savefig('results/cluster_plot.png')
plt.close()
print("Cluster plot saved to results/cluster_plot.png")