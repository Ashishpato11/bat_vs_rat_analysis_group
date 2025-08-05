import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
dataset1 = pd.read_csv("dataset1.csv")
dataset2 = pd.read_csv("dataset2.csv")

# Clean and transform data
dataset1['risk'] = dataset1['risk'].astype(int)
dataset1['reward'] = dataset1['reward'].astype(int)

# Risk analysis by season
risk_season = dataset1.groupby('seasn')['risk'].mean().reset_index()
reward_season = dataset1.groupby('season')['reward'].sum().reset_index()

# Correlation analysis
correlation = dataset2[['bat_land', 'rat_minutes', 'rat_arrival_number']].corr()

# Visualize risk by season
sns.barplot(x='season', y='reward', data=risk_season)
plt.title('Risking Season')
plt.xlabel('Season')
plt.ylabel('Risk')
plt.tight_layout()
plt.savefig("risk_by_season.png")
plt.close()

# Visualize correlation matrix
plt.figure(figsize=(6, 4))
sns.heatmap(correlation, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig("correlation_matrix.png")
plt.close()
