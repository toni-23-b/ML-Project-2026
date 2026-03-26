import matplotlib.pyplot as plt
import seaborn as sns
from splitter import train_df, test_df


print("--- Visualizing the K-Means Market Regimes ---")

# Set up the size and style of the graph
plt.figure(figsize=(12, 7))
sns.set_theme(style="whitegrid")

# Create the scatter plot
# We use train_df because that's the data the AI learned from
scatter = sns.scatterplot(
    data=train_df,
    x='massive_whale_volume',
    y='max_gas_gwei',
    hue='Market_Regime', # Colors the dots based on the AI's cluster choice
    palette='viridis',   # A professional color scheme (blues/greens/yellows)
    alpha=0.7,           # Makes the dots slightly transparent to see overlaps
    edgecolor='k',
    s=60                 # Size of the dots
)

# Add titles and labels for the presentation
plt.title('K-Means Market Regimes: Whale Volume vs. Market Panic', fontsize=16, fontweight='bold')
plt.xlabel('Massive Whale Volume (ETH)', fontsize=12)
plt.ylabel('Max Gas Paid (Gwei - Panic Indicator)', fontsize=12)

# Customize the legend
plt.legend(title='Market Regime (Cluster ID)', title_fontsize='13', fontsize='11')

# Optional: Limit the axes if massive outliers make the graph hard to read
# plt.xlim(0, train_df['massive_whale_volume'].quantile(0.99))
# plt.ylim(0, train_df['max_gas_gwei'].quantile(0.99))

plt.tight_layout()
plt.show()