import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set plot style and figure size
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)


# LOAD & PREPARE DATA
data = pd.read_csv("C:\\Users\\BHAVESH\\Downloads\\7102_source_data.csv")

# Rename columns
data.rename(columns={
    'srcStateName': 'State',
    'srcYear': 'FinancialYear',
    'srcMonth': 'FinancialMonth',
    'GST ( Goods and Service Tax ) Return Type': 'GST_Return_Type',
    'Payer eligible for GST ( Goods and Service Tax ) registration': 'Eligible_Payers',
    'GST ( Goods and Service Tax ) Payers registered before due date': 'Payers_Before_Due',
    'GST ( Goods and Service Tax ) Payers registered after due date': 'Payers_After_Due',
    'YearCode': 'YearCode',
    'Year': 'YearLabel',
    'MonthCode': 'MonthCode',
    'Month': 'MonthLabel'
}, inplace=True)

# Strip column names
data.columns = data.columns.str.strip()

# CLEANING & FEATURE ENGINEERING
# Convert relevant columns to numeric and handle errors
numeric_cols = ['Eligible_Payers', 'Payers_Before_Due', 'Payers_After_Due']
for col in numeric_cols:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# FILL rows with missing critical numeric values
#data.dropna(subset=numeric_cols, inplace=True)
data['Payers_Before_Due']=data['Payers_Before_Due'].fillna(data['Payers_Before_Due'].mode()[0])
print(data['Payers_Before_Due'].isna().sum())

data['Payers_After_Due']=data['Payers_After_Due'].fillna(data['Payers_After_Due'].mode()[0])
print(data['Payers_After_Due'].isna().sum())






# Replace zero Eligible_Payers to avoid division by zero
data = data[data['Eligible_Payers'] > 0]

# Derived features
data['Total_Filers'] = data['Payers_Before_Due'] + data['Payers_After_Due']
data['OnTime_Percentage'] = (data['Payers_Before_Due'] / data['Eligible_Payers']) * 100
data['Late_Percentage'] = (data['Payers_After_Due'] / data['Eligible_Payers']) * 100

# Drop NaNs in derived columns if any
data.dropna(subset=['OnTime_Percentage', 'Late_Percentage'], inplace=True)

# BASIC STATS
print("\n=== BASIC INFO ===")
print(data.info())
print("\nSummary Stats:")
print(data.describe())


# PAIR PLOT
pairplot_data = data[['Eligible_Payers', 'Payers_Before_Due', 'Payers_After_Due',
                      'Total_Filers', 'OnTime_Percentage', 'Late_Percentage']]

sns.pairplot(pairplot_data.dropna())
plt.suptitle("Pair Plot of GST Filing Metrics", y=1.02)
plt.show()


# CORRELATION HEATMAP
correlation = pairplot_data.corr()
plt.figure(figsize=(10, 6))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap of GST Filing Metrics")
plt.show()


# BOXPLOT: On-Time % by Return Type
plt.figure(figsize=(10, 6))
sns.boxplot(x='GST_Return_Type', y='OnTime_Percentage', data=data)
plt.title("On-Time Filing Percentage by GST Return Type")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# DISTRIBUTION PLOT
plt.figure(figsize=(10, 6))
sns.histplot(data['OnTime_Percentage'].dropna(), kde=True, bins=30, color='teal')
plt.title("Distribution of On-Time Filing Percentage")
plt.xlabel("On-Time Filing Percentage")
plt.ylabel("Frequency")
plt.show()


# TREND OVER TIME
yearly_trend = data.groupby('FinancialYear')['OnTime_Percentage'].mean()
plt.figure(figsize=(10, 5))
sns.lineplot(x=yearly_trend.index, y=yearly_trend.values, marker='o')
plt.title("Average On-Time Filing % Over Financial Years")
plt.xlabel("Financial Year")
plt.ylabel("On-Time Filing (%)")
plt.show()


# TOP STATES: ON-TIME FILING %
top_states = data.groupby('State')['OnTime_Percentage'].mean().sort_values(ascending=False).head(10)
plt.figure(figsize=(12, 6))
sns.barplot(x=top_states.values, y=top_states.index)
plt.title("Top 10 States by Average On-Time Filing (%)")
plt.xlabel("Average On-Time Filing %")
plt.ylabel("State")
plt.show()


# HEATMAP: STATE vs YEAR
heatmap_data = data.pivot_table(index='State', columns='FinancialYear', values='OnTime_Percentage', aggfunc='mean')
plt.figure(figsize=(14, 10))
sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="YlGnBu")
plt.title("On-Time GST Filing % by State and Financial Year")
plt.tight_layout()
plt.show()

from scipy import stats

def perform_z_test(group1, group2, group1_name, group2_name):
    """
    Perform Z-test comparing on-time filing proportions.
    """
    x1 = group1['Payers_Before_Due'].sum()
    n1 = group1['Eligible_Payers'].sum()
    x2 = group2['Payers_Before_Due'].sum()
    n2 = group2['Eligible_Payers'].sum()

    # Proportions
    p1 = x1 / n1
    p2 = x2 / n2

    # Pooled proportion
    p_pool = (x1 + x2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
    z = (p1 - p2) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    print(f"\n===== Z-TEST: {group1_name} vs {group2_name} =====")
    print(f"Proportion {group1_name}: {p1:.4f} ({x1:,} on-time / {n1:,} eligible)")
    print(f"Proportion {group2_name}: {p2:.4f} ({x2:,} on-time / {n2:,} eligible)")
    print(f"Z-Score: {z:.4f}")
    print(f"P-Value: {p_value:.4f}")
    if p_value < 0.05:
        print("Conclusion: Significant difference (p < 0.05)")
    else:
        print("Conclusion: No significant difference (p ≥ 0.05)")


# Z-TEST 1: Maharashtra vs Gujarat
state1 = 'Maharashtra'
state2 = 'Gujarat'
group1 = data[data['State'] == state1]
group2 = data[data['State'] == state2]
perform_z_test(group1, group2, state1, state2)
