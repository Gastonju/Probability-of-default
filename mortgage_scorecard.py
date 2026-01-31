# %% 0. Settings
# imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from optbinning import BinningProcess
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# %% 1. Data Import
data = pd.read_csv("mortgage_sample.csv")


# %% 2. Sample preparation and checks
# Although mortgages have long contractual maturities
# (e.g., 20 years), we estimate a 12-month probability of default (PD-12M).
# fix cohort approach were going to use all loans originated up to 12 months ago

# TODO: Need to use fixed or flexible cohort approach!

data["max_time_per_loan"] = data.groupby("id")["time"].transform("max")
data["orig_time_per_loan"] = data.groupby("id")["orig_time"].transform("first")
data["age"] = data["time"] - data["orig_time"]
data["max_age"] = data.groupby("id")["age"].transform("max")
data = data[data["orig_time"] == 25]

data.to_csv("mortgage_full_sample.csv", index=False)

# %%

# TODO: Need to add split into train and test!
data = data[(data["sample"] == "public")]


loan_end = data.groupby("id").agg(
    final_status=("status_time", "max"), end_time=("time", "max")
)
data = data.merge(loan_end, on="id", how="left")


def compute_target_robust(row):
    horizon = row["time"] + 12
    # Si le prêt finit par un défaut
    if row["final_status"] == 1:
        # Et que le défaut est dans l'horizon -> 1
        if row["end_time"] <= horizon:
            return 1
        else:
            return 0
    # Si le prêt est remboursé (2) ou actif (0)
    else:
        # Si on a observé le prêt jusqu'à l'horizon -> 0 (Sûr)
        if row["end_time"] >= horizon:
            return 0
        # Sinon -> NaN (On ne sait pas, on exclut)
        else:
            return np.nan


data["TARGET"] = data.apply(compute_target_robust, axis=1)
data_model = data.dropna(subset=["TARGET"])

unique_ids = data["id"].unique()
train_ids, test_ids = train_test_split(unique_ids, test_size=0.2, random_state=42)

train = data[data["id"].isin(train_ids)]
test = data[data["id"].isin(test_ids)]

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

train = train.sort_values(["id", "time"]).copy()
test = test.sort_values(["id", "time"]).copy()

for df in (train, test):
    df["hpi_ratio"] = df["hpi_time"] / df["hpi_orig_time"]
    df["stress_index"] = df["uer_time"] * df["LTV_time"]
    df["interest_burden"] = df["balance_time"] * df["interest_rate_time"]

# variation (trend) features
for df in (train, test):
    # month-to-month changes
    df["d_LTV_1m"] = df.groupby("id")["LTV_time"].diff(1)
    df["d_bal_1m"] = df.groupby("id")["balance_time"].diff(1)
    df["d_rate_1m"] = df.groupby("id")["interest_rate_time"].diff(1)
    df["d_uer_1m"] = df.groupby("id")["uer_time"].diff(1)
    df["d_hpi_ratio_1m"] = df.groupby("id")["hpi_ratio"].diff(1)
    df["d_gdp_1m"] = df.groupby("id")["gdp_time"].diff(1)

    # smothing features
    df["LTV_roll_mean_3m"] = df.groupby("id")["LTV_time"].transform(
        lambda s: s.rolling(3, min_periods=1).mean()
    )
    df["LTV_roll_std_6m"] = df.groupby("id")["LTV_time"].transform(
        lambda s: s.rolling(6, min_periods=2).std()
    )

    df["gdp_roll_mean_3m"] = df.groupby("id")["gdp_time"].transform(
        lambda s: s.rolling(3, min_periods=1).mean()
    )
    df["gdp_roll_std_6m"] = df.groupby("id")["gdp_time"].transform(
        lambda s: s.rolling(6, min_periods=2).std()
    )

    df["bal_roll_mean_3m"] = df.groupby("id")["balance_time"].transform(
        lambda s: s.rolling(3, min_periods=1).mean()
    )
    df["bal_roll_std_6m"] = df.groupby("id")["balance_time"].transform(
        lambda s: s.rolling(6, min_periods=2).std()
    )

    df["stress_roll_mean_3m"] = df.groupby("id")["stress_index"].transform(
        lambda s: s.rolling(3, min_periods=1).mean()
    )
    df["stress_roll_std_6m"] = df.groupby("id")["stress_index"].transform(
        lambda s: s.rolling(6, min_periods=2).std()
    )

# Client level aggregation for clustering
client_features = (
    train.groupby("id")
    .agg(
        LTV_mean=("LTV_time", "mean"),
        LTV_max=("LTV_time", "max"),
        gdp_mean=("gdp_time", "mean"),
        gdp_max=("gdp_time", "max"),
        balance_mean=("balance_time", "mean"),
        balance_max=("balance_time", "max"),
        uer_mean=("uer_time", "mean"),
        interest_rate_mean=("interest_rate_time", "mean"),
        hpi_ratio_mean=("hpi_ratio", "mean"),
        stress_index_mean=("stress_index", "mean"),
        interest_burden_mean=("interest_burden", "mean"),
        gdp_d_max=("d_gdp_1m", "max"),
        LTV_std=("LTV_time", "std"),
        balance_std=("balance_time", "std"),
        stress_index_std=("stress_index", "std"),
        gdp_std=("gdp_time", "std"),
        LTV_last=("LTV_time", "last"),
        balance_last=("balance_time", "last"),
        uer_last=("uer_time", "last"),
        rate_last=("interest_rate_time", "last"),
        hpi_ratio_last=("hpi_ratio", "last"),
        LTV_recent_mean_3m=("LTV_roll_mean_3m", "last"),
        LTV_recent_std_6m=("LTV_roll_std_6m", "last"),
        bal_recent_mean_3m=("bal_roll_mean_3m", "last"),
        bal_recent_std_6m=("bal_roll_std_6m", "last"),
        stress_recent_mean_3m=("stress_roll_mean_3m", "last"),
        stress_recent_std_6m=("stress_roll_std_6m", "last"),
        gdp_recent_mean_3m=("gdp_roll_mean_3m", "last"),
        gdp_recent_std_6m=("gdp_roll_std_6m", "last"),
        d_LTV_mean=("d_LTV_1m", "mean"),
        d_bal_mean=("d_bal_1m", "mean"),
        d_rate_mean=("d_rate_1m", "mean"),
        d_uer_mean=("d_uer_1m", "mean"),
        d_hpi_ratio_mean=("d_hpi_ratio_1m", "mean"),
        d_gdp_mean=("d_gdp_1m", "mean"),
    )
    .reset_index()
)

client_features_test = (
    test.groupby("id")
    .agg(
        LTV_mean=("LTV_time", "mean"),
        LTV_max=("LTV_time", "max"),
        gdp_mean=("gdp_time", "mean"),
        gdp_max=("gdp_time", "max"),
        balance_mean=("balance_time", "mean"),
        balance_max=("balance_time", "max"),
        uer_mean=("uer_time", "mean"),
        interest_rate_mean=("interest_rate_time", "mean"),
        hpi_ratio_mean=("hpi_ratio", "mean"),
        stress_index_mean=("stress_index", "mean"),
        interest_burden_mean=("interest_burden", "mean"),
        gdp_d_max=("d_gdp_1m", "max"),
        LTV_std=("LTV_time", "std"),
        balance_std=("balance_time", "std"),
        stress_index_std=("stress_index", "std"),
        gdp_std=("gdp_time", "std"),
        LTV_last=("LTV_time", "last"),
        balance_last=("balance_time", "last"),
        uer_last=("uer_time", "last"),
        rate_last=("interest_rate_time", "last"),
        hpi_ratio_last=("hpi_ratio", "last"),
        LTV_recent_mean_3m=("LTV_roll_mean_3m", "last"),
        LTV_recent_std_6m=("LTV_roll_std_6m", "last"),
        bal_recent_mean_3m=("bal_roll_mean_3m", "last"),
        bal_recent_std_6m=("bal_roll_std_6m", "last"),
        stress_recent_mean_3m=("stress_roll_mean_3m", "last"),
        stress_recent_std_6m=("stress_roll_std_6m", "last"),
        gdp_recent_mean_3m=("gdp_roll_mean_3m", "last"),
        gdp_recent_std_6m=("gdp_roll_std_6m", "last"),
        d_LTV_mean=("d_LTV_1m", "mean"),
        d_bal_mean=("d_bal_1m", "mean"),
        d_rate_mean=("d_rate_1m", "mean"),
        d_uer_mean=("d_uer_1m", "mean"),
        d_hpi_ratio_mean=("d_hpi_ratio_1m", "mean"),
        d_gdp_mean=("d_gdp_1m", "mean"),
    )
    .reset_index()
)

# which variables to use for clustering
cluster_vars = [
    "LTV_mean",
    "LTV_max",
    "LTV_std",
    "LTV_last",
    "balance_mean",
    "balance_max",
    "balance_std",
    "balance_last",
    "uer_mean",
    "uer_last",
    "interest_rate_mean",
    "rate_last",
    "hpi_ratio_mean",
    "hpi_ratio_last",
    "stress_index_mean",
    "stress_index_std",
    "interest_burden_mean",
    "LTV_recent_mean_3m",
    "LTV_recent_std_6m",
    "stress_recent_mean_3m",
    "stress_recent_std_6m",
    "d_LTV_mean",
    "d_bal_mean",
    "d_rate_mean",
    "d_uer_mean",
    "d_hpi_ratio_mean",
    "d_gdp_mean",
    "gdp_d_max",
    "gdp_mean",
]

# Handle NaNs (std/diff/rolling may create NaNs)
client_features[cluster_vars] = (
    client_features[cluster_vars].replace([np.inf, -np.inf], np.nan).fillna(0)
)
client_features_test[cluster_vars] = (
    client_features_test[cluster_vars].replace([np.inf, -np.inf], np.nan).fillna(0)
)


scaler = StandardScaler()
X_train = scaler.fit_transform(client_features[cluster_vars])
X_test = scaler.transform(client_features_test[cluster_vars])


kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
client_features["cluster_id"] = kmeans.fit_predict(X_train)
client_features_test["cluster_id"] = kmeans.predict(X_test)

# Merge cluster id back to panel
train = train.merge(client_features[["id", "cluster_id"]], on="id", how="left")
test = test.merge(client_features_test[["id", "cluster_id"]], on="id", how="left")

# Sanity checks
assert train.groupby("id")["cluster_id"].nunique().max() == 1
assert test.groupby("id")["cluster_id"].nunique().max() == 1
assert train["cluster_id"].isna().sum() == 0
assert test["cluster_id"].isna().sum() == 0

train["cluster_id"] = train["cluster_id"].astype("category")
test["cluster_id"] = test["cluster_id"].astype("category")

# Quick cluster stats
clients_per_cluster = train.groupby("cluster_id")["id"].nunique()
print("Clients per cluster (train):")
print(clients_per_cluster)

print("\nClient-level default rate by cluster (train):")
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

# %%

# TODO: Multivariate (correlation) check

# remove hpi_origin_time due to high correlation with orig_time_per_loan
to_remove = [
    "orig_time_per_loan",
    "orig_time",
    "LTV_time",
    "balance_time",
    "gdp_time",
    "stress_index",
    "LTV_roll_mean_3m",
    "hpi_ratio",
]
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
p_value = 0.05

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
threshold = np.quantile(y_pred, 0.70)


# TODO: Check the final model quality - p-values? Coefficient signs?
estimated_model.summary()


# %% 6. Train Performance assessment
# TODO: GINI is the most common metric for assessing predictive power
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)

auc = metrics.roc_auc_score(y_test, y_pred)

plt.plot(fpr, tpr, label="GINI=" + str(2 * auc - 1))
plt.legend(loc=4)
plt.show()

# TODO: Other model assessment dimensions

y_pred_const = (y_pred >= 0.1).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_const).ravel()

benchmark = pd.DataFrame(
    columns=[
        "Model name",
        "GINI",
        "Pseudo R2",
        "Sensitivity",
        "Specificity",
        "Accuracy",
        "Precision",
    ]
)

benchmark.loc[len(benchmark)] = {
    "Model name": "Logit Clustered",
    "GINI": 2 * auc - 1,
    "Pseudo R2": estimated_model.prsquared,
    "Sensitivity": tp / (tp + fn) if (tp + fn) > 0 else np.nan,
    "Specificity": tn / (tn + fp) if (tn + fp) > 0 else np.nan,
    "Accuracy": (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else np.nan,
    "Precision": tp / (tp + fp) if (tp + fp) > 0 else np.nan,
}

print(benchmark)

nb_default_clients = train.groupby("id")["TARGET"].max().sum()


nb_default_clients_rate = test.groupby("id")["TARGET"].max().mean()
print(nb_default_clients_rate)

# %% 7. Test Performance assessment
# TODO: Make sure to check for overfitting!
