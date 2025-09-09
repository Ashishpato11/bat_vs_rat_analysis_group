import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#for z score analysis
from scipy.stats import zscore



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

#updated the data handling 
dataset1['risk'] = pd.to_numeric(dataset1['risk'], errors='coerce')
dataset1['reward'] = pd.to_numeric(dataset1['reward'], errors='coerce')


num_cols_1 = ['bat_landing_to_food', 'seconds_after_rat_arrival', 'hours_after_sunset']
num_cols_2 = ['bat_landing_number', 'food_availability', 'rat_minutes', 'rat_arrival_number', 'hours_after_sunset']

for col in num_cols_1:
    dataset1[col] = pd.to_numeric(dataset1[col], errors='coerce')

for col in num_cols_2:
    dataset2[col] = pd.to_numeric(dataset2[col], errors='coerce')




#Removing the column that has important missing values
dataset1 = dataset1.dropna(subset=['bat_landing_to_food', 'seconds_after_rat_arrival', 'risk', 'reward'])
dataset2 = dataset2.dropna(subset=['bat_landing_number', 'rat_minutes', 'rat_arrival_number'])

"""Descriptive analysis of data sets 1

"""


"function to create the descriptive analysis"
def descriptive_analysis(data, col_name, title_prefix):
    mean_val = data[col_name].mean()
    median_val = data[col_name].median()
    mode_val = data[col_name].mode()[0]
    var_val = data[col_name].var()
    std_val = data[col_name].std()
    Q1 = data[col_name].quantile(0.25)
    Q3 = data[col_name].quantile(0.75)
    IQR = Q3 - Q1

    

    # Histogram visulaization for the 
    plt.figure(figsize=(8,5))
    sns.histplot(data[col_name], bins=30, kde=True, color='skyblue')
    plt.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
    plt.axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.2f}')
    plt.axvline(mode_val, color='orange', linestyle='--', label=f'Mode: {mode_val}')
    plt.title(f"{title_prefix} - Histogram")
    plt.xlabel(col_name)
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Boxplot
    plt.figure(figsize=(6,4))
    sns.boxplot(x=data[col_name], color='lightcoral')
    plt.title(f"{title_prefix} - Boxplot")
    plt.xlabel(col_name)
    plt.tight_layout()
    plt.show()

    # Barplot for mean, median, mode
    stats_summary = {'Mean': mean_val, 'Median': median_val, 'Mode': mode_val}
    plt.figure(figsize=(6,4))
    sns.barplot(x=list(stats_summary.keys()), y=list(stats_summary.values()), palette='viridis')
    plt.title(f"{title_prefix} - Descriptive Stats")
    plt.ylabel(col_name)
    plt.tight_layout()
    plt.show()


# Descriptive analysis for dataset1 (bat landing times)
descriptive_analysis(dataset1, 'bat_landing_to_food', 'Dataset1 - Bat Landing to Food')

# Descriptive analysis for dataset2 (bat landings, rat minutes, rat arrivals)
for col in ['bat_landing_number', 'rat_minutes', 'rat_arrival_number']:
    descriptive_analysis(dataset2, col, f'Dataset2 - {col}')

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


"""Formulating and testing the hypothesis 
Null hypothesis :behaviour does not change
alternate hypothesis behviour change when rat are present"""


#creating two groups
#taking the data from dataset1,after any rats arrived
bats_with_rats=dataset1[dataset1['seconds_after_rat_arrival']>0]['bat_landing_to_food']


#taking the data from dataset1 ,before any rats arrived
bats_without_rats=dataset1[dataset1['seconds_after_rat_arrival']==0]['bat_landing_to_food']


#applying t test for significace analysis 
#more clarilty is required in thisgit 
t_stat, p_val = stats.ttest_ind(bats_with_rats.dropna(), bats_without_rats.dropna())
print("t-statistic:", t_stat, "p-value:", p_val)



#comparing p-value test difference in bat behaviour
if p_val < 0.05:
    print("Rat presence significantly affects bat behaviour")
else:
    print("No significant difference")





dataset1['bat_landing_z'] = zscore(dataset1['bat_landing_to_food'])
dataset1['seconds_after_rat_arrival_z'] = zscore(dataset1['seconds_after_rat_arrival'])



# Visualize Z-scores for bat_landing_to_food
plt.figure(figsize=(8,5))
sns.histplot(dataset1['bat_landing_z'], bins=30, kde=True, color='skyblue')
plt.title("Z-score Distribution of Bat Landing to Food")
plt.xlabel("Z-score")
plt.ylabel("Frequency")
plt.axvline(0, color='red', linestyle='--', label='Mean = 0')
plt.legend()
plt.tight_layout()
plt.show()

# Visualize Z-scores for seconds_after_rat_arrival
plt.figure(figsize=(8,5))
sns.histplot(dataset1['seconds_after_rat_arrival_z'], bins=30, kde=True, color='lightgreen')
plt.title("Z-score Distribution of Seconds After Rat Arrival")
plt.xlabel("Z-score")
plt.ylabel("Frequency")
plt.axvline(0, color='red', linestyle='--', label='Mean = 0')
plt.legend()
plt.tight_layout()
plt.show()
