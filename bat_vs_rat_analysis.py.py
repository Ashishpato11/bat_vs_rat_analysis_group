import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#importing for t test
from scipy import stats

# Import datasets csv
dataset1 = pd.read_csv("dataset1.csv")
dataset2 = pd.read_csv("dataset2.csv")


#testing the import of data need to clean this after
print(dataset1.columns)
print(dataset2)


# Data Cleaning and transformation
#selects column ris from the data frame as int converts all the values to integer
dataset1['risk'] = dataset1['risk'].astype(int)
#selects column reward from the data frame
dataset1['reward'] = dataset1['reward'].astype(int)


# Risk analysis by season
risk_season = dataset1.groupby('season')['risk'].mean().reset_index()
reward_season = dataset1.groupby('season')['reward'].mean().reset_index()

print(risk_season)
print(reward_season)

# Correlation analysis
correlation = dataset2[['bat_landing_number', 'rat_minutes', 'rat_arrival_number']].corr()


# Plotting the correlation matrix
plt.figure(figsize=(6, 4))
sns.heatmap(correlation, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig("correlation_matrix.png")
plt.close()

# # Plotting average risk by season
plt.figure(figsize=(6, 4))
sns.barplot(x='season', y='reward', data=reward_season)
plt.title('Average Risk-Taking Behaviour by Season')
plt.xlabel('Season')
plt.ylabel('Average Risk')
plt.tight_layout()
plt.savefig("risk_by_season.png")
plt.close()

# proportion of reward vs risk (no rat vs rat)
plt.figure(figsize=(6,4))
sns.barplot(
    data=dataset1, x="risk", y="reward", ci=None, estimator=lambda x: sum(x)/len(x)
)
plt.xticks([0,1], ["No Rat", "Rat Present"])
plt.title("Proportion of Rewards vs risk")
plt.ylabel("Proportion of Rewards")
plt.xlabel("Risk")
plt.ylim(0,1)
plt.show()


"""Formulating and testing the hypothesis 
Null hypothesis :behaviour does not change
alternate hypothesis behviour change when rat are present"""


#creating two groups
#taking the data from dataset1,after any rats arrived
bats_with_rats=dataset1[dataset1['seconds_after_rat_arrival']>0]['bat_landing_to_food']


#taking the data from dataset1 ,before any rats arrived
bats_without_rats=dataset1[dataset1['seconds_after_rat_arrival']==0]['bat_landing_to_food']


#applying t test

#more clarilty is required in thisgit 
t_stat, p_val = stats.ttest_ind(bats_with_rats.dropna(), bats_without_rats.dropna())
print("t-statistic:", t_stat, "p-value:", p_val)

if p_val < 0.05:
    print("Reject H0 → Rat presence significantly affects bat behaviour")
else:
    print("Fail to reject H0 → No significant difference")
