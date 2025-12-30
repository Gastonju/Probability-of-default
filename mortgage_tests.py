# %% 0. Settings
# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn import metrics
from scipy.stats import norm, somersd

# %% 1. Data Import
data = pd.read_csv("mortgage_sample.csv")

# %% 2. Subsampling of the original data

sample = data[(data["sample"] == "public")]

# Drop rows with missing values in key columns 
sample = sample.dropna(subset=["LTV_time", "FICO_orig_time", "default_time", "TARGET", "interest_rate_time"])

# %% 3. GINI calculation
# Simple model with two predictors and intercept
X_basic = sample[["LTV_time", "FICO_orig_time"]]
X_basic = sm.add_constant(X_basic)  # Add intercept term
logit_mod = sm.Logit(endog=sample["TARGET"], exog=X_basic)
estimated_model = logit_mod.fit(disp=0)
y_pred = estimated_model.predict()

# Print model summary
print(estimated_model.summary())

# GINI calculation
gini = 2 * metrics.roc_auc_score(sample["TARGET"], y_pred) - 1
print("GINI:", gini)

# Plot ROC curve
fpr, tpr, _ = metrics.roc_curve(sample["TARGET"], y_pred)
auc = metrics.roc_auc_score(sample["TARGET"], y_pred)
plt.plot(fpr, tpr, label="GINI=" + str(2 * auc - 1))
plt.plot([0, 1], [0, 1], linestyle="--")
plt.legend(loc=4)
plt.show()

# %% 4. Correlations
# Correlation matrix
corr_pearson = sample[["LTV_time", "FICO_orig_time", "interest_rate_time"]].corr(method="pearson")
print(corr_pearson)

corr_spearman = sample[["LTV_time", "FICO_orig_time", "interest_rate_time"]].corr(method="spearman")
print(corr_spearman)

# Scatter plot matrix
pd.plotting.scatter_matrix(sample[["LTV_time", "FICO_orig_time", "interest_rate_time"]], alpha=0.2, figsize=(10, 10), diagonal='kde')
plt.suptitle("Scatter Matrix")
plt.show()


# %% 5. Model performance - Sommer's D
# Calculation of Sommers D is really demanding, dont run it on large datasets
def sommers_d(y_true, y_scores):
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    n = len(y_true)
    num_concordant = 0
    num_discordant = 0
    num_tied = 0

    for i in range(n):
        for j in range(i + 1, n):
            if y_true[i] != y_true[j]:  
                if y_scores[i] > y_scores[j]:
                    if y_true[i] > y_true[j]:
                        num_concordant += 1
                    else:
                        num_discordant += 1
                elif y_scores[i] < y_scores[j]:
                    if y_true[i] < y_true[j]:
                        num_concordant += 1
                    else:
                        num_discordant += 1
                else: 
                    num_tied += 1

    total_pairs = num_concordant + num_discordant + num_tied
    if total_pairs == 0:
        return 0.0

    d = (num_concordant - num_discordant) / total_pairs
    return d

# Take top 1000 observations by model score (y_pred)
sample_small = sample.copy()
sample_small["y_pred"] = y_pred
sample_small = sample_small.sort_values("y_pred", ascending=False).head(1000)

# Convert to numpy arrays
y_true_array = sample_small["TARGET"].values
y_pred_array = sample_small["y_pred"].values

# Compute Somers' D (using your existing function or scipy)
d = sommers_d(y_true_array, y_pred_array)
print(f"Somers' D (top 1000): {d:.4f}")

# %% Build-in sommers d funciton

sd_res = somersd(y_true_array, y_pred_array)
somers_d_scipy = sd_res.statistic
print(f"Somers' D (scipy.stats.somersd): {somers_d_scipy:.6f}")

# https://stackoverflow.com/questions/59442544/is-there-an-efficient-python-implementation-for-somersd-for-ungrouped-variables

# %% 6. Model performance - binomial Z-test
sample_reset = sample.reset_index(drop=True)
y_pred_reset = pd.Series(y_pred).reset_index(drop=True)

cal = pd.DataFrame({
    "y": sample_reset["TARGET"],
    "pd": y_pred_reset
}).dropna()

# Decile bins
cal["bin"] = pd.qcut(cal["pd"], q=10, duplicates="drop")

# Group by bins and calculate statistics
cal_tbl = (cal.groupby("bin", observed=True)
             .agg({
                 "y": ["count", "sum", "mean"],
                 "pd": "mean"
             })
             .round(4))

# Flatten column names
cal_tbl.columns = ["n", "events", "DR", "PD"]
cal_tbl = cal_tbl.reset_index()

# Z-statistic and p-value for binomial test
cal_tbl["var"] = cal_tbl["PD"] * (1 - cal_tbl["PD"]) / cal_tbl["n"]
cal_tbl["Z"] = (cal_tbl["DR"] - cal_tbl["PD"]) / np.sqrt(cal_tbl["var"])
cal_tbl["p_value_norm"] = 2 * (1 - norm.cdf(np.abs(cal_tbl["Z"])))  # two-sided test

print("\nCalibration Table (Binomial Z-test):")
print(cal_tbl[["bin", "n", "events", "DR", "PD", "Z", "p_value_norm"]])

# Calibration plot: DR vs PD
plt.figure(figsize=(8, 6))
plt.plot(cal_tbl["PD"], cal_tbl["DR"], marker="o", linewidth=2, markersize=8, label="Observed DR")
plt.plot([0, cal_tbl["PD"].max()], [0, cal_tbl["PD"].max()], "--", color="red", alpha=0.7, label="Perfect Calibration")
plt.xlabel("Average Predicted Probability (PD)")
plt.ylabel("Observed Default Rate (DR)")
plt.title("Model Calibration: Observed vs Predicted Default Rates")
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% 7. Population Stability Index (PSI) - Proper Implementation

# Filter the two populations
df = sample.loc[sample["time"].isin([25, 26]), ["time", "LTV_time"]].dropna()

# Define expected (time 25) and actual (time 26)
expected = df.loc[df["time"] == 25, "LTV_time"]
actual   = df.loc[df["time"] == 26, "LTV_time"]

# Define quantile bins based on expected distribution
bins = np.quantile(expected, np.linspace(0, 1, 11))
bins = np.unique(bins)  # remove duplicates if any

# Bin the two distributions
exp_counts, _ = np.histogram(expected, bins=bins)
act_counts, _ = np.histogram(actual, bins=bins)

# Convert to proportions
exp_pct = exp_counts / np.sum(exp_counts)
act_pct = act_counts / np.sum(act_counts)

# Avoid division by zero
exp_pct = np.where(exp_pct == 0, 1e-6, exp_pct)
act_pct = np.where(act_pct == 0, 1e-6, act_pct)

# PSI calculation
psi_value = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
print(f"Population Stability Index (PSI) for LTV_time (time 25 vs 26): {psi_value:.4f}")


# Prepare table for plotting
psi_table = pd.DataFrame({
    "bin_left": bins[:-1],
    "bin_right": bins[1:],
    "expected_pct": exp_pct,
    "actual_pct": act_pct,
    "psi_contrib": (act_pct - exp_pct) * np.log(act_pct / exp_pct)
})
psi_table["bin_label"] = psi_table.apply(
    lambda r: f"{r.bin_left:.1f}–{r.bin_right:.1f}", axis=1
)

print("\nPSI Table:" )
print(psi_table[["bin_label", "expected_pct", "actual_pct", "psi_contrib"]])
# Histogram comparison
plt.figure(figsize=(7, 4))
plt.hist(expected, bins=bins, alpha=0.6, label="time 25", color="steelblue", density=True)
plt.hist(actual, bins=bins, alpha=0.6, label="time 26", color="salmon", density=True)
plt.xlabel("LTV_time")
plt.ylabel("Density")
plt.title("Distribution of LTV_time (time 25 vs 26)")
plt.legend()
plt.tight_layout()
plt.show()

# Quantile bar chart 
x = np.arange(len(psi_table))
width = 0.35

plt.figure(figsize=(8, 4))
plt.bar(x - width/2, psi_table["expected_pct"], width, label="time 25", color="steelblue")
plt.bar(x + width/2, psi_table["actual_pct"],   width, label="time 26", color="salmon")
plt.xticks(x, psi_table["bin_label"], rotation=45, ha="right")
plt.ylabel("Proportion")
plt.title("LTV_time – proportion in each decile (time 25 vs 26)")
plt.legend()
plt.tight_layout()
plt.show()

# PSI contribution chart 
plt.figure(figsize=(8, 4))
plt.bar(x, psi_table["psi_contrib"], color="darkorange")
plt.xticks(x, psi_table["bin_label"], rotation=45, ha="right")
plt.axhline(0, color="black", linewidth=0.8)
plt.ylabel("PSI contribution")
plt.title(f"PSI contributions by LTV_time decile (Total PSI = {psi_value:.4f})")
plt.tight_layout()
plt.show()


# %%
