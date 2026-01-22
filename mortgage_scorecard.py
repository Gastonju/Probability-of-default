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
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


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
data = data[(data["sample"] == "public")]

loan_recency = (
    data.groupby("id")["time"].max().reset_index().rename(columns={"time": "last_time"})
)

loan_recency = loan_recency.sort_values("last_time")

index = int(0.8 * len(loan_recency))

train_ids = loan_recency.iloc[:index]["id"]
test_ids = loan_recency.iloc[index:]["id"]

train = data[data["id"].isin(train_ids)]
test = data[data["id"].isin(test_ids)]

default_date = data[data["default_time"] == 1].groupby("id")["time"].min()


def compute_target_correct(row, df_defaults):
    current_id = row["id"]
    current_time = row["time"]

    if current_id not in df_defaults.index:
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

# remove missing value in the training set
train_remove_ids = train[train["LTV_time"].isnull()]["id"].unique()
test_remove_ids = test[test["LTV_time"].isnull()]["id"].unique()

train = train[~train["id"].isin(train_remove_ids)].copy()
test = test[~test["id"].isin(test_remove_ids)].copy()

sample = train.copy()


# %% 3. Predictor preparation
cluster_vars = [
    "LTV_mean",
    "LTV_max",
    "balance_mean",
    "balance_max",
    "uer_mean",
    "interest_rate_mean",
    "hpi_ratio_mean",
    "stress_index_mean",
    "interest_burden_mean",
]

train["hpi_ratio"] = train["hpi_time"] / train["hpi_orig_time"]
train["stress_index"] = train["uer_time"] * train["LTV_time"]
train["interest_burden"] = train["balance_time"] * train["interest_rate_time"]

test["hpi_ratio"] = test["hpi_time"] / test["hpi_orig_time"]
test["stress_index"] = test["uer_time"] * test["LTV_time"]
test["interest_burden"] = test["balance_time"] * test["interest_rate_time"]


client_features = (
    train.groupby("id")
    .agg(
        LTV_mean=("LTV_time", "mean"),
        LTV_max=("LTV_time", "max"),
        balance_mean=("balance_time", "mean"),
        balance_max=("balance_time", "max"),
        uer_mean=("uer_time", "mean"),
        interest_rate_mean=("interest_rate_time", "mean"),
        hpi_ratio_mean=("hpi_ratio", "mean"),
        stress_index_mean=("stress_index", "mean"),
        interest_burden_mean=("interest_burden", "mean"),
    )
    .reset_index()
)


sil_scores = []


scaler = StandardScaler()
X_cluster_train = scaler.fit_transform(client_features[cluster_vars])

kmeans = KMeans(n_clusters=4, random_state=42)
client_features["cluster_id"] = kmeans.fit_predict(X_cluster_train)

train = train.merge(client_features[["id", "cluster_id"]], on="id", how="left")

client_features_test = (
    test.groupby("id")
    .agg(
        LTV_mean=("LTV_time", "mean"),
        LTV_max=("LTV_time", "max"),
        balance_mean=("balance_time", "mean"),
        balance_max=("balance_time", "max"),
        uer_mean=("uer_time", "mean"),
        interest_rate_mean=("interest_rate_time", "mean"),
        hpi_ratio_mean=("hpi_ratio", "mean"),
        stress_index_mean=("stress_index", "mean"),
        interest_burden_mean=("interest_burden", "mean"),
    )
    .reset_index()
)

X_cluster_test = scaler.transform(client_features_test[cluster_vars])
client_features_test["cluster_id"] = kmeans.predict(X_cluster_test)

test = test.merge(client_features_test[["id", "cluster_id"]], on="id", how="left")

assert train.groupby("id")["cluster_id"].nunique().max() == 1
assert test.groupby("id")["cluster_id"].nunique().max() == 1

train["cluster_id"] = train["cluster_id"].astype("category")
test["cluster_id"] = test["cluster_id"].astype("category")

clients_per_cluster = train.groupby("cluster_id")["id"].nunique()
print(clients_per_cluster)

print(
    train.groupby(["cluster_id", "id"], observed=True)["TARGET"]
    .max()
    .groupby("cluster_id", observed=True)
    .mean()
)


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

y_train = train["TARGET"]
y_test = test["TARGET"]
test_ids = test["id"]
train_ids = train["id"]


binning_process.fit(train[predictors], y_train)
binning_process.summary()

train_woe = binning_process.transform(train[predictors], metric="woe")
test_woe = binning_process.transform(test[predictors], metric="woe")

test_woe.to_csv("mortgage_test_woe.csv", index=False)
train_woe.to_csv("mortgage_train_woe.csv", index=False)


# TODO: Multivariate (correlation) check

# remove hpi_origin_time due to high correlation with orig_time_per_loan
to_remove = ["orig_time_per_loan", "orig_time", "LTV_time", "balance_time"]
train_woe = train_woe.drop(columns=to_remove, errors="ignore")
test_woe = test_woe.drop(columns=to_remove, errors="ignore")

corr_matrix = train_woe.corr()
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr_matrix, cmap="coolwarm", annot=False)
ax.set_title("Correlation Matrix Heatmap")
plt.show()

# Calculation VIF
vif_data = pd.DataFrame()
vif_data["Variable"] = train_woe.columns
vif_data["VIF"] = [
    variance_inflation_factor(train_woe.values, i) for i in range(train_woe.shape[1])
]

print(vif_data.sort_values("VIF", ascending=False))


# %% 4. Modelling
# TODO: Think carefully about predictors we want to add
X = train_woe
X = sm.add_constant(X, has_constant="add")
test_woe = sm.add_constant(test_woe, has_constant="add")


# %% Simple regression estimation
logit_mod = sm.Logit(endog=y_train, exog=X)
estimated_model = logit_mod.fit(disp=0)
estimated_model.summary()


# %% Stepwise selection can give us an idea about significant predictors of risk
# Implement forward regression, start with intercept
selected = ["const"]
not_selected = [key for key in X.columns if key not in selected]
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
        logit_mod = sm.Logit(endog=y_train, exog=X[selected + [key]])
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
logit_mod = sm.Logit(endog=y_train, exog=X[selected])
estimated_model = logit_mod.fit(disp=0)
y_pred = estimated_model.predict(test_woe[selected])

# TODO: Check the final model quality - p-values? Coefficient signs?
estimated_model.summary()

pd.DataFrame({"TARGET": y_test, "PD": y_pred}).groupby("TARGET").mean()

tmp = pd.DataFrame({"TARGET": y_test.values, "PD": y_pred})
print(tmp.groupby("TARGET")["PD"].describe())

print("Médiane PD défaut:", tmp[tmp.TARGET == 1].PD.median())
print("Médiane PD non-défaut:", tmp[tmp.TARGET == 0].PD.median())

# %% 6. Train Performance assessment
# TODO: GINI is the most common metric for assessing predictive power
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)

auc = metrics.roc_auc_score(y_test, y_pred)

plt.plot(fpr, tpr, label="GINI=" + str(2 * auc - 1))
plt.legend(loc=4)
plt.show()

# TODO: Other model assessment dimensions


# %% 7. Test Performance assessment
# TODO: Make sure to check for overfitting!
