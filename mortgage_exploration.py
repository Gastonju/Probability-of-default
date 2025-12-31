# %%
# Import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# Loading data from csv
data = pd.read_csv("mortgage_sample.csv")
# Check number of rows and columns
print("Data shape:", data.shape)
print("Columns:", data.columns)

# %%
# Display first 5 rows
print(data.head())


# %%
# Check data types of variables
print("\nData types:\n", data.dtypes)

# %%
# Check duplicates
print("\nNumber of duplicate rows:\n", data.duplicated().sum())


# %%
# Check missing values
print("\nMissing values:\n", data.isnull().sum())

# Count of rows with missing values
null_rows = data.isnull().any(axis=1)
print("\nNumber of rows with missing values:", null_rows.sum())

# %%
# Basic descriptive statistics

# First only numeric variables
num_vars = data.select_dtypes(include=["float64", "int64"])

print("\nDescriptive statistics:", num_vars.describe())

# %%
# We can also calculate each statistic individually

# Count
print("\nCounts:\n", num_vars.count())

# Mean
print("\nMeans:\n", num_vars.mean())

# Standard deviation
print("\nStandard deviations:\n", num_vars.std())

# Minimum
print("\nMinimum values:\n", num_vars.min())

# Maximum
print("\nMaximum values:\n", num_vars.max())

# Quantile
print("\nQuantiles (0.25):\n", num_vars.quantile(q=0.25))


# %%
# We can also calculate statistics for individual columns

mean_uer_time = data["uer_time"].mean()
std_uer_time = data["uer_time"].std()
min_uer_time = data["uer_time"].min()

print(f"\nMean of uer_time: {mean_uer_time:.2f}")
print(f"Std deviation of uer_time: {std_uer_time:.2f}")
print(f"Minimum of uer_time: {min_uer_time:.2f}")

# %%
# Visualization of selected variables
# Histogram
fig, ax = plt.subplots(figsize=(8, 6))
ax.hist(data["FICO_orig_time"], bins=100)
ax.set_xlabel("FICO score")
ax.set_ylabel("Frequency")
plt.show()

# %%
# Scatter plot
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(data["hpi_time"], data["gdp_time"], alpha=0.5)
ax.set_xlabel("House price index")
ax.set_ylabel("GDP growth")
plt.show()

# %%
# Histograms for all numerical variables

num_vars.hist(figsize=(12, 10), bins=20)
plt.suptitle("Distributions of numeric variables")
plt.show()

# %%
# Boxplot for LTV_orig_time
sns.boxplot(x=data["balance_time"])
plt.title("Boxplot of balance_time")
plt.show()

sns.boxplot(x=data["LTV_orig_time"])
plt.title("Boxplot of LTV_orig_time")
plt.show()

sns.boxplot(x=data["LTV_time"])
plt.title("Boxplot of LTV_time")
plt.show()

sns.boxplot(x=data["interest_rate_time"])
plt.title("Boxplot of interest_rate_time")
plt.show()

sns.boxplot(x=data["hpi_time"])
plt.title("Boxplot of hpi_time")
plt.show()

sns.boxplot(x=data["gdp_time"])
plt.title("Boxplot of gdp_time")
plt.show()

sns.boxplot(x=data["balance_orig_time"])
plt.title("Boxplot of balance_orig_time")
plt.show()

sns.boxplot(x=data["FICO_orig_time"])
plt.title("Boxplot of FICO_orig_time")
plt.show()

sns.boxplot(x=data["Interest_Rate_orig_time"])
plt.title("Boxplot of Interest_Rate_orig_time")
plt.show()

sns.boxplot(x=data["hpi_orig_time"])
plt.title("Boxplot of hpi_orig_time")
plt.show()

sns.boxplot(x=data["uer_time"])
plt.title("Boxplot of uer_time")
plt.show()
# %%
# Correlation

# Correlation matrix for all numeric variables
corr_matrix = num_vars.corr()

# Visualization of correlation matrix as heatmap
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr_matrix, cmap="coolwarm", annot=False)
ax.set_title("Correlation Matrix Heatmap")
plt.show()

# %%
# Data aggregation

# Group by home type and calculate average FICO score in each type
home_type_fico_mean = data.groupby(["REtype_SF_orig_time"])[
    "FICO_orig_time"].mean()
print(home_type_fico_mean)

# %%
# Data filtering by conditions

# Select only loans with FICO score greater than 700
high_fico_loans = data[data["FICO_orig_time"] > 700]


print(high_fico_loans.head())

# %%
# Missing values

# Fill missing values with the mean of the given variable
data_mean_imputed = num_vars.fillna(num_vars.mean())
print(data_mean_imputed.isnull().sum())

# %%
# Remove rows with missing values
data_dropped = data.dropna()
print(data_dropped.isnull().sum())

# Q: Other thoughts how to handle missing data?

# %%
# Simple function for variable modification


# Define function to classify loans by FICO score
def fico_class(row):
    if row["FICO_orig_time"] >= 750:
        return "excellent"
    elif row["FICO_orig_time"] >= 700:
        return "good"
    else:
        return "poor"


# Apply function to FICO score column
data["FICO_class"] = data.apply(fico_class, axis=1)
FICO_class_counts = data.groupby(["FICO_class"])["FICO_class"].count()
print(FICO_class_counts)


# %%
# Boxplot for outliers for FICO_class
sns.boxplot(x="FICO_class", y="LTV_orig_time", data=data)
plt.title("LTV by FICO class")
plt.show()

# %%
