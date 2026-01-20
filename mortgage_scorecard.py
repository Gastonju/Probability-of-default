# %% 0. Settings
# imports
import pandas as pd
import numpy as np
import statsmodels.api as sm
from optbinning import BinningProcess
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor

# %% 1. Data Import
data = pd.read_csv("mortgage_sample.csv")

# %% 2. Sample preparation and checks
# Although mortgages have long contractual maturities
# (e.g., 20 years), we estimate a 12-month probability of default (PD-12M).
# fix cohort approach were going to use all loans originated up to 12 months ago

# TODO: Need to use fixed or flexible cohort approach!

data["max_time_per_loan"] = data.groupby("id")["time"].transform("max")
data["orig_time_per_loan"] = data.groupby("id")["orig_time"].transform("first")
data["age"] = data.groupby("id").cumcount() + 1
data["max_age"] = data.groupby("id")["age"].transform("max")

data = data


data = data[(data["max_age"] >= 12)]

# TODO: Need to add split into train and test!
sample = data[(data["sample"] == "public")]

loan_recency = (
    sample.groupby("id")["time"]
    .max()
    .reset_index()
    .rename(columns={"time": "last_time"})
)

loan_recency = loan_recency.sort_values("last_time")

index = int(0.8 * len(loan_recency))

train_ids = loan_recency.iloc[:index]["id"]
test_ids = loan_recency.iloc[index:]["id"]

train = data[data["id"].isin(train_ids)]
test = data[data["id"].isin(test_ids)]

default_date = train[train["default_time"] == 1].groupby("id")["time"].min()


def compute_target_correct(row, df_defaults):
    current_id = row["id"]
    current_time = row["time"]

    if current_id not in df_defaults:
        return 0

    time_of_default = df_defaults[current_id]
    months_until_default = time_of_default - current_time

    if 0 <= months_until_default <= 12:
        return 1
    else:
        return 0


train["TARGET"] = train.apply(compute_target_correct, args=(default_date,), axis=1)
test["TARGET"] = test.apply(compute_target_correct, args=(default_date,), axis=1)

# %% 3
# training_set = sample

# TODO: Exploratory data analysis & treatment of missings + outliers

print("number clients in training before", train["id"].nunique())
print("missing values in training set before:", train.isnull().sum())

# remove missing value in the training set
train_remove_ids = train[train["LTV_time"].isnull()]["id"].unique()
test_remove_ids = test[test["LTV_time"].isnull()]["id"].unique()

train = train[~train["id"].isin(train_remove_ids)]
test = test[~test["id"].isin(test_remove_ids)]

sample = train.copy()
# %% 3. Predictor preparation


# %%
# TODO: All available should be explored to maximize model predictive power
cols_to_drop = [
    "id",
    "time",
    "TARGET",
    "default_time",
    "payoff_time",
    "status_time",
    "sample",
    "max_time_per_loan",
    "max_age",
]

predictors = [col for col in train.columns if col not in cols_to_drop]
binning_process = BinningProcess(
    variable_names=predictors,
    selection_criteria={"iv": {"min": 0.02}},
    binning_fit_params={var: {"solver": "mip"} for var in predictors},
)

X = train[predictors]
y_train = train["TARGET"]
y_test = test["TARGET"]
test_ids = test["id"]
train_ids = train["id"]

binning_process.fit(X, y_train)
binning_process.summary()

train = binning_process.transform(train[predictors], metric="woe")
test = binning_process.transform(test[predictors], metric="woe")

train["TARGET"] = y_train
test["TARGET"] = y_test

test.to_csv("mortgage_test_woe.csv", index=False)
train.to_csv("mortgage_train_woe.csv", index=False)

# TODO: Multivariate (correlation) check

# remove hpi_origin_time due to high correlation with orig_time_per_loan
train = train.drop(columns=["hpi_orig_time", "orig_time"])
test = test.drop(columns=["hpi_orig_time", "orig_time"])

corr_matrix = train.corr()
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr_matrix, cmap="coolwarm", annot=False)
ax.set_title("Correlation Matrix Heatmap")
plt.show()

# Calculation VIF
vif_data = pd.DataFrame()
vif_data["Variable"] = train.columns
vif_data["VIF"] = [
    variance_inflation_factor(train.values, i) for i in range(train.shape[1])
]

print(vif_data.sort_values("VIF", ascending=False))

# %% 4. Modelling
# TODO: Think carefully about predictors we want to add
vars_to_remove = ["TARGET"]
X = train.drop(columns=vars_to_remove, errors="ignore")
X = sm.add_constant(X)

# %% Simple regression estimation
logit_mod = sm.Logit(endog=y_train, exog=X)
estimated_model = logit_mod.fit(disp=0)
estimated_model.summary()


# %% Stepwise selection can give us an idea about significant predictors of risk
# Implement forward regression, start with intercept
selected = ["const"]
not_selected = [key for key in predictors if key not in selected]
p_value = 0.01

while True:
    # model changed?
    changed = False
    selected_prev = selected.copy()  # save previoust list

    # dataframe with p-values
    p_values = pd.Series(index=not_selected, dtype="float64")

    ## forward step
    # loop over not selected columns
    for key in not_selected:
        # estimate model
        logit_mod = sm.Logit(endog=y, exog=X[selected + [key]])
        estimated_model = logit_mod.fit(disp=0)

        # extract p_value
        p_values[key] = estimated_model.pvalues[key]

    # select best predictor with best p-value (if it is below threshold)
    min_pval = p_values.min()
    if min_pval < p_value:
        best_feature = p_values.idxmin()
        selected.append(best_feature)
        not_selected.remove(best_feature)
        print("Add  {:30} with p-value {:.6}".format(best_feature, min_pval))
    else:
        print("No predictor added")

    # check if model changed
    if set(selected) != set(selected_prev):
        changed = True

    if len(selected) == len(X.columns):
        print("\n No more predictors to test!")
        break

    if not changed:
        break

# %% 5. Estimate final model
# TODO: Make sure to create several candidate models to compare
logit_mod = sm.Logit(endog=y, exog=X[selected])
estimated_model = logit_mod.fit(disp=0)
y_pred = estimated_model.predict()

# TODO: Check the final model quality - p-values? Coefficient signs?
estimated_model.summary()

# %% 6. Train Performance assessment
# TODO: GINI is the most common metric for assessing predictive power
fpr, tpr, _ = metrics.roc_curve(y, y_pred)
auc = metrics.roc_auc_score(y, y_pred)
plt.plot(fpr, tpr, label="GINI=" + str(2 * auc - 1))
plt.legend(loc=4)
plt.show()

# TODO: Other model assessment dimensions


# %% 7. Test Performance assessment
# TODO: Make sure to check for overfitting!
