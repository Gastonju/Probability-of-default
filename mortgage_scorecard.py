# %% 0. Settings
# imports
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
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
from sklearn.linear_model import LogisticRegressionCV


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

    if row["final_status"] == 1:

        if row["end_time"] <= horizon:
            return 1
        else:
            return 0

    else:

        if row["end_time"] >= horizon:
            return 0
        # we don't know
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

train = train.copy()
test = test.copy()

numeric_cols = train.select_dtypes(include=["number"]).columns

for i in numeric_cols:
    median = train[i].median()
    train[i] = train[i].fillna(median)
    test[i] = test[i].fillna(median)

sample = train.copy()


# %% 3. Predictor preparation

train = train.sort_values(["id", "time"]).copy()
test = test.sort_values(["id", "time"]).copy()

for df in (train, test):
    df["hpi_ratio"] = df["hpi_time"] / df["hpi_orig_time"]
    df["stress_index"] = df["uer_time"] * df["LTV_time"]
    df["interest_burden"] = df["balance_time"] * df["interest_rate_time"]
    df["LTV_x_FICO"] = df["LTV_time"] * df["FICO_orig_time"]
    df["LTV_x_interest"] = df["LTV_time"] * df["interest_rate_time"]
    df["uer_lag3"] = df.groupby("id")["uer_time"].shift(3)
    df["hpi_lag3"] = df.groupby("id")["hpi_time"].shift(3)

# if user unemployment rate increase default now ?
for df in (train, test):
    df["uer_lag6"] = df.groupby("id")["uer_time"].shift(6)
    df["uer_lag3"] = df["uer_lag3"].fillna(df["uer_time"])
    df["hpi_lag3"] = df["hpi_lag3"].fillna(df["hpi_time"])
    df["uer_lag6"] = df["uer_lag6"].fillna(df["uer_time"])

# integration quadratic, features
for df in (train, test):
    df["LTV_time_sq"] = df["LTV_time"] ** 2
    df["balance_time_sq"] = df["balance_time"] ** 2
    df["interest_rate_time_sq"] = df["interest_rate_time"] ** 2
    df["uer_time_sq"] = df["uer_time"] ** 2
    df["hpi_ratio_sq"] = df["hpi_ratio"] ** 2
    df["gdp_time_sq"] = df["gdp_time"] ** 2
    df["stress_index_sq"] = df["stress_index"] ** 2


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
        LTV_orig_time=("LTV_orig_time", "first"),
        FICO_orig_time=("FICO_orig_time", "first"),
        balance_orig_time=("balance_orig_time", "first"),
        Interest_Rate_orig_time=("Interest_Rate_orig_time", "first"),
        investor_orig_time=("investor_orig_time", "first"),
    )
    .reset_index()
)

client_features_test = (
    test.groupby("id")
    .agg(
        LTV_orig_time=("LTV_orig_time", "first"),
        FICO_orig_time=("FICO_orig_time", "first"),
        balance_orig_time=("balance_orig_time", "first"),
        Interest_Rate_orig_time=("Interest_Rate_orig_time", "first"),
        investor_orig_time=("investor_orig_time", "first"),
    )
    .reset_index()
)

# which variables to use for clustering
cluster_vars = [
    "LTV_orig_time",
    "FICO_orig_time",
    "balance_orig_time",
    "Interest_Rate_orig_time",
    "investor_orig_time",
]


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


assert train.groupby("id")["cluster_id"].nunique().max() == 1
assert test.groupby("id")["cluster_id"].nunique().max() == 1
assert train["cluster_id"].isna().sum() == 0
assert test["cluster_id"].isna().sum() == 0

train["cluster_id"] = train["cluster_id"].astype("category")
test["cluster_id"] = test["cluster_id"].astype("category")


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
    "end_time",
    "final_status",
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
    "hpi_ratio_sq",
    "interest_rate_time_sq",
    "uer_time_sq",
    "bal_roll_mean_3m",
    "age",
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
estimated_model_full = logit_mod.fit(disp=0)
y_pred = estimated_model_full.predict(test_woe)

estimated_model_full.summary()

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
estimated_model_step_regression = logit_mod.fit(disp=0)
y_pred_step_regre = estimated_model_step_regression.predict(test_woe[selected])

# TODO: Check the final model quality - p-values? Coefficient signs?
estimated_model_step_regression.summary()


# %% 6. Train Performance assessment
# TODO: GINI is the most common metric for assessing predictive power
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_step_regre)

auc = metrics.roc_auc_score(y_test, y_pred_step_regre)

auc_full = metrics.roc_auc_score(y_test, y_pred)

plt.plot(fpr, tpr, label="GINI=" + str(2 * auc - 1))
plt.legend(loc=4)
plt.show()

y_pred_const_step_regression = (y_pred_step_regre >= 0.41).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_const_step_regression).ravel()

y_pred_const_full = (y_pred >= 0.41).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_const_full).ravel()

benchmark.loc[len(benchmark)] = {
    "Model name": "Logitistic Regression Full",
    "GINI": 2 * auc_full - 1,
    "Pseudo R2": estimated_model_full.prsquared,
    "Sensitivity": tp / (tp + fn) if (tp + fn) > 0 else np.nan,
    "Specificity": tn / (tn + fp) if (tn + fp) > 0 else np.nan,
    "Accuracy": (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else np.nan,
    "Precision": tp / (tp + fp) if (tp + fp) > 0 else np.nan,
}

benchmark.loc[len(benchmark)] = {
    "Model name": "Logitistic Regression stepwise",
    "GINI": 2 * auc - 1,
    "Pseudo R2": estimated_model_step_regression.prsquared,
    "Sensitivity": tp / (tp + fn) if (tp + fn) > 0 else np.nan,
    "Specificity": tn / (tn + fp) if (tn + fp) > 0 else np.nan,
    "Accuracy": (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else np.nan,
    "Precision": tp / (tp + fp) if (tp + fp) > 0 else np.nan,
}

# TODO: Other model assessment dimensions

# %% lasso regression estimation

if "const" in X.columns:
    X = X.drop(columns="const")
if "const" in test_woe.columns:
    test_woe = test_woe.drop(columns="const")

lasso_model = LogisticRegressionCV(
    Cs=20,
    cv=10,
    penalty="l1",
    solver="liblinear",
    scoring="roc_auc",
    random_state=42,
    max_iter=1000,
)

lasso_model.fit(X, y_train)
y_pred_lasso = lasso_model.predict_proba(test_woe)[:, 1]


# calculate prsquare
ll_model = np.sum(
    y_test * np.log(y_pred_lasso + 1e-15)
    + (1 - y_test) * np.log(1 - y_pred_lasso + 1e-15)
)

p_null = np.mean(y_test)
ll_null = np.sum(y_test * np.log(p_null) + (1 - y_test) * np.log(1 - p_null))
pseudo_r2 = 1 - ll_model / ll_null

fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_lasso)
auc = metrics.roc_auc_score(y_test, y_pred_lasso)
gini = 2 * auc - 1

y_pred_const_lasso = (y_pred_lasso >= 0.41).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_const_lasso).ravel()
benchmark.loc[len(benchmark)] = {
    "Model name": "Logitistic Regression Lasso",
    "GINI": 2 * auc - 1,
    "Pseudo R2": pseudo_r2,
    "Sensitivity": tp / (tp + fn) if (tp + fn) > 0 else np.nan,
    "Specificity": tn / (tn + fp) if (tn + fp) > 0 else np.nan,
    "Accuracy": (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else np.nan,
    "Precision": tp / (tp + fp) if (tp + fp) > 0 else np.nan,
}

# %% elastic bet regression estimation

if "const" in X.columns:
    X = X.drop(columns="const")
if "const" in test_woe.columns:
    test_woe = test_woe.drop(columns="const")

elastic_net_cv = LogisticRegressionCV(
    Cs=10,
    l1_ratios=[0.1, 0.5, 0.7, 0.9],  # Teste aussi différentes combinaisons L1/L2
    cv=10,
    penalty="elasticnet",
    solver="saga",
    scoring="roc_auc",
    random_state=42,
    max_iter=2000,
)
elastic_net_cv.fit(X, y_train)
y_pred_elas_net = elastic_net_cv.predict_proba(test_woe)[:, 1]

# calculate prsquare
ll_model = np.sum(
    y_test * np.log(y_pred_elas_net + 1e-15)
    + (1 - y_test) * np.log(1 - y_pred_elas_net + 1e-15)
)

p_null = np.mean(y_test)
ll_null = np.sum(y_test * np.log(p_null) + (1 - y_test) * np.log(1 - p_null))
pseudo_r2 = 1 - ll_model / ll_null

# calculate GINI
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_elas_net)
auc = metrics.roc_auc_score(y_test, y_pred_elas_net)
gini = 2 * auc - 1

y_pred_const_elas_net = (y_pred_elas_net >= 0.41).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_const_elas_net).ravel()
benchmark.loc[len(benchmark)] = {
    "Model name": "Logitistic Regression Elastic Net",
    "GINI": 2 * auc - 1,
    "Pseudo R2": pseudo_r2,
    "Sensitivity": tp / (tp + fn) if (tp + fn) > 0 else np.nan,
    "Specificity": tn / (tn + fp) if (tn + fp) > 0 else np.nan,
    "Accuracy": (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else np.nan,
    "Precision": tp / (tp + fp) if (tp + fp) > 0 else np.nan,
}
# %% 7. Test Performance assessment
# TODO: Make sure to check for overfitting!
print(benchmark)
nb_default_clients = train.groupby("id")["TARGET"].max().mean()
print("Overall default rate in training set:", nb_default_clients)

# %%
