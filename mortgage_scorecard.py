# %% 0. Settings
# imports
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from optbinning import BinningProcess
import inspect
import pandas as pd
import numpy as np
import statsmodels.api as sm

# Compatibility patch: scikit-learn >= 1.6 renamed force_all_finite -> ensure_all_finite


def _patch_optbinning_sklearn_compat() -> None:
    try:
        from sklearn.utils.validation import check_array as _sk_check_array
        import optbinning.binning.metrics as _ob_metrics
    except Exception:
        return

    sig = inspect.signature(_sk_check_array)
    has_force = "force_all_finite" in sig.parameters
    has_ensure = "ensure_all_finite" in sig.parameters

    if (not has_force) and has_ensure:
        def _check_array_compat(*args, **kwargs):
            if "force_all_finite" in kwargs and "ensure_all_finite" not in kwargs:
                kwargs["ensure_all_finite"] = kwargs.pop("force_all_finite")
            return _sk_check_array(*args, **kwargs)

        _ob_metrics.check_array = _check_array_compat


_patch_optbinning_sklearn_compat()


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
id_event = (
    data.groupby("id")["default_time"]
        .max()
        .reset_index()
        .rename(columns={"default_time": "ever_default"})
)

train_ids, test_ids = train_test_split(
    id_event["id"],
    test_size=0.2,
    random_state=42,
    stratify=id_event["ever_default"]
)

train = data[data["id"].isin(train_ids)].copy()
test = data[data["id"].isin(test_ids)].copy()


default_date = (
    data.loc[data["default_time"] == 1]
    .groupby("id")["time"]
    .min()
    .rename("time_of_default")
)


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


# PD-12M target at observation time t:
# TARGET=1 if first default occurs in (t, t+12] months.
# Also removes observations after default to prevent leakage.
def _add_target(df: pd.DataFrame, defaults: pd.Series) -> pd.DataFrame:
    out = df.merge(defaults, on="id", how="left")

    out = out[(out["time_of_default"].isna()) | (
        out["time"] <= out["time_of_default"])].copy()

    months_until_default = out["time_of_default"] - out["time"]

    out["TARGET"] = (
        out["time_of_default"].notna()
        & (months_until_default >= 1)
        & (months_until_default <= 12)
    ).astype(int)

    out.drop(columns=["time_of_default"], inplace=True)
    return out


train = _add_target(train, default_date)
test = _add_target(test, default_date)


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

train_pit = train.sort_values(["id", "time"]).groupby(
    "id", as_index=False).head(1)
test_pit = test.sort_values(["id", "time"]).groupby(
    "id", as_index=False).head(1)

client_features = (
    train_pit.groupby("id")
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
    test_pit.groupby("id")
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

test = test.merge(
    client_features_test[["id", "cluster_id"]], on="id", how="left")

assert train.groupby("id")["cluster_id"].nunique().max() == 1
assert test.groupby("id")["cluster_id"].nunique().max() == 1

train["cluster_id"] = train["cluster_id"].astype("category")
test["cluster_id"] = test["cluster_id"].astype("category")

clients_per_cluster = train.groupby(
    "cluster_id", observed=True)["id"].nunique()
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
if "cluster_id" in predictors:
    predictors.remove("cluster_id")

binning_process = BinningProcess(
    variable_names=predictors,
    selection_criteria={"iv": {"min": 0.02}},
    binning_fit_params={var: {"solver": "mip"} for var in predictors},
)

y_train = train["TARGET"].astype(int)
y_test = test["TARGET"].astype(int)
test_ids = test["id"]
train_ids = train["id"]


binning_process.fit(train[predictors], y_train)
binning_process.summary()

train_woe = binning_process.transform(train[predictors], metric="woe")
test_woe = binning_process.transform(test[predictors], metric="woe")

train_woe, test_woe = train_woe.align(
    test_woe, join="left", axis=1, fill_value=0)

train_cluster_d = pd.get_dummies(
    train["cluster_id"], prefix="cluster", drop_first=True)
test_cluster_d = pd.get_dummies(
    test["cluster_id"],  prefix="cluster", drop_first=True)

test_cluster_d = test_cluster_d.reindex(
    columns=train_cluster_d.columns, fill_value=0)

# Make dummies numeric explicitly
train_cluster_d = train_cluster_d.astype(float)
test_cluster_d = test_cluster_d.astype(float)

train_woe = pd.concat([train_woe, train_cluster_d], axis=1)
test_woe = pd.concat([test_woe,  test_cluster_d], axis=1)

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
# VIF requires numeric + finite
train_woe = train_woe.apply(pd.to_numeric, errors="coerce")
test_woe = test_woe.apply(pd.to_numeric, errors="coerce")

train_woe = train_woe.replace([np.inf, -np.inf], np.nan).fillna(0.0)
test_woe = test_woe.replace([np.inf, -np.inf], np.nan).fillna(0.0)

# Drop constant columns
const_cols = [
    c for c in train_woe.columns if train_woe[c].nunique(dropna=False) <= 1]
if const_cols:
    print("Dropping constant columns for VIF:", const_cols)
    train_woe = train_woe.drop(columns=const_cols)
    test_woe = test_woe.drop(columns=const_cols, errors="ignore")

vif_data = pd.DataFrame()
X_vif = sm.add_constant(train_woe, has_constant="add")

vif_data = pd.DataFrame({
    "Variable": X_vif.columns,
    "VIF": [
        variance_inflation_factor(X_vif.to_numpy(dtype=float), i)
        for i in range(X_vif.shape[1])
    ]
})

# Drop intercept from report
vif_data = vif_data[vif_data["Variable"] != "const"]
print(vif_data.sort_values("VIF", ascending=False))


# %% 4. Modelling
# TODO: Think carefully about predictors we want to add
# Ensure model matrices are strictly numeric + finite (statsmodels requirement)
train_woe = train_woe.apply(pd.to_numeric, errors="coerce")
test_woe = test_woe.apply(pd.to_numeric, errors="coerce")

train_woe = train_woe.replace([np.inf, -np.inf], np.nan).fillna(0.0)
test_woe = test_woe.replace([np.inf, -np.inf], np.nan).fillna(0.0)

# Align columns/order one last time (prevents train/test mismatch)
train_woe, test_woe = train_woe.align(
    test_woe, join="left", axis=1, fill_value=0.0)

# Build exog matrices
X = sm.add_constant(train_woe, has_constant="add")
X_test = sm.add_constant(test_woe, has_constant="add")

# Force float dtype explicitly
X = X.astype(float)
X_test = X_test.astype(float)

# %% Simple regression estimation
logit_mod = sm.Logit(endog=y_train.astype(int), exog=X)
estimated_model = logit_mod.fit_regularized(alpha=1.0, L1_wt=0.0)
print(estimated_model.summary())


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

    # forward step
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
y_pred = estimated_model.predict(X_test[selected])

# TODO: Check the final model quality - p-values? Coefficient signs?
estimated_model.summary()

# Attach scores to identifiers
test_scored = test[["id", "time", "TARGET"]].copy()
test_scored["PD"] = y_pred.values

# One score per loan: PD at first observation in test (PIT-at-start)
loan_score = (
    test_scored.sort_values(["id", "time"])
    .groupby("id", as_index=False)
    .first()[["id", "PD"]]
)

# One label per loan: did the loan EVER have TARGET=1 in the test panel?
loan_label = (
    test_scored.groupby("id", as_index=False)["TARGET"]
    .max()
    .rename(columns={"TARGET": "TARGET_loan"})
)

loan_level = loan_score.merge(loan_label, on="id", how="left")

print(loan_level.groupby("TARGET_loan")["PD"].describe())
print("Median PD default (loan):",
      loan_level.loc[loan_level.TARGET_loan == 1, "PD"].median())
print("Median PD non-default (loan):",
      loan_level.loc[loan_level.TARGET_loan == 0, "PD"].median())


# %% 6. Train Performance assessment
# TODO: GINI is the most common metric for assessing predictive power

y_true_loan = loan_level["TARGET_loan"].values
y_score_loan = loan_level["PD"].values

fpr, tpr, _ = metrics.roc_curve(y_true_loan, y_score_loan)
auc = metrics.roc_auc_score(y_true_loan, y_score_loan)

plt.plot(fpr, tpr, label="GINI=" + str(2 * auc - 1))
plt.legend(loc=4)
plt.show()

# TODO: Other model assessment dimensions


# %% 7. Test Performance assessment
# TODO: Make sure to check for overfitting!
