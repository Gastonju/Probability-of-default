# %% 0. Settings
# imports
import pandas as pd
import numpy as np
import statsmodels.api as sm
from optbinning import OptimalBinning, BinningProcess
from sklearn import metrics
import matplotlib.pyplot as plt


# %% 1. Data Import
data = pd.read_csv("mortgage_sample.csv")

# %% 2. Sample preparation and checks
# Although mortgages have long contractual maturities
# (e.g., 20 years), we estimate a 12-month probability of default (PD-12M).
# fix cohort approach were going to use all loans originated up to 12 months ago

# TODO: Need to use fixed or flexible cohort approach!

data["max_time_per_loan"] = data.groupby("id")["time"].transform("max")
data["orig_time_per_loan"] = data.groupby("id")["orig_time"].transform("first")
data["max_age"] = data.groupby("id")["age"].transform("max")

data_cohort = data
data_cohort["age"] = data_cohort.groupby("id").cumcount() + 1


data_cohort = data_cohort[(data_cohort["max_age"] >= 12)]

# TODO: Need to add split into train and test!
sample = data_cohort[(data_cohort["sample"] == "public")]

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

train = data_cohort[data_cohort["id"].isin(train_ids)]
test = data_cohort[data_cohort["id"].isin(test_ids)]

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

test.to_csv("mortgage_test.csv", index=False)
train.to_csv("mortgage_train.csv", index=False)

# %% 3
# training_set = sample

# TODO: Exploratory data analysis & treatment of missings + outliers

# print("missing values in test set:", test.isnull().sum())
print("number clients in training before", train["id"].nunique())
print("missing values in training set before:", train.isnull().sum())

# remove missing value in the training set
train_remove_ids = train[train["LTV_time"].isnull()]["id"].unique()
test_remove_ids = test[test["LTV_time"].isnull()]["id"].unique()

train = train[~train["id"].isin(train_remove_ids)]
test = test[~test["id"].isin(test_remove_ids)]

# sample = sample.dropna(
#    subset=["LTV_orig_time", "FICO_orig_time", "LTV_time", "default_time", "TARGET"]
# )
# %% 3. Predictor preparation


# %% Example OptimalBinning for one predictor
predictor = "LTV_orig_time"
x = sample[predictor]
y = sample["TARGET"]

optb = OptimalBinning(name=predictor)
optb.fit(x, y)
optb.binning_table.build()
# %%
optb.binning_table.analysis()

# %%
# TODO: All available should be explored to maximize model predictive power
predictors = ["FICO_orig_time", "LTV_time"]
binning_process = BinningProcess(
    variable_names=predictors,
    binning_fit_params={"LTV_time": {"special_codes": [92.75]}},
)

X = sample[predictors]
y = sample["default_time"]

binning_process.fit_transform(X, y)
binning_process.summary()

# TODO: Multivariate (correlation) check

# %% 4. Modelling
# TODO: Think carefully about predictors we want to add
X = sm.add_constant(X)

# %% Simple regression estimation
logit_mod = sm.Logit(endog=y, exog=X)
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
