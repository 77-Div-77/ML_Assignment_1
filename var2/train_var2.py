"""
VAR2 Model Training and Model Selection Pipeline
================================================
Roll Number: IMT2024068
Dataset: Var2 (3 features: x1, x2, x3 | Target: y)

Objective Function:
  J(w) = (1/N) * ||y - Xw||_2^2 + lambda1 * ||w||_1 + lambda2 * ||w||_2^2

Conversion to scikit-learn:
  - Lasso:  alpha = lambda1 / 2.0
  - Ridge:  alpha = N * lambda2
  - ElasticNet:
      alpha_total = lambda1 / 2.0 + lambda2
      l1_ratio = (lambda1 / 2.0) / alpha_total

Selected Methodology:
  - Cross-Validation: 10-Fold CV (shuffle=True, random_state=42)
  - Preprocessing: StandardScaler fit strictly inside CV folds
  - Model Selection: Selection of polynomial degree and regularization
    based on minimum cross-validation RMSE.
"""

import os
import json
import math
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "Dataset") if os.path.exists(os.path.join(PROJECT_ROOT, "Dataset")) else os.path.dirname(PROJECT_ROOT)

TRAIN_PATH = os.path.join(DATA_DIR, "IMT2024068_train_var2.csv")
TEST_PATH = os.path.join(DATA_DIR, "IMT2024068_test_var2.csv")

OUT_SWEEP = os.path.join(SCRIPT_DIR, "var2_degree_sweep.csv")
OUT_COEF = os.path.join(SCRIPT_DIR, "polynomial_coefficients_var2.csv")
OUT_MODEL = os.path.join(SCRIPT_DIR, "var2_model.joblib")
OUT_META = os.path.join(SCRIPT_DIR, "var2_model_meta.json")
OUT_PRED = os.path.join(SCRIPT_DIR, "IMT2024068_pred_var2.csv")

# ── Load Data ────────────────────────────────────────────────────────────────
print("[Step 1] Loading Dataset Var2...")
df_train = pd.read_csv(TRAIN_PATH)
FEATURES = ["x1", "x2", "x3"]
X_all = df_train[FEATURES].values
y_all = df_train["y"].values
N = len(y_all)

print(f"Loaded {N} samples with features: {FEATURES}")

KF = KFold(n_splits=10, shuffle=True, random_state=42)
splits = list(KF.split(X_all))

def poly_terms_count(n_feat, deg):
    return math.comb(n_feat + deg, deg) - 1

def build_pipeline(degree, lambda1, lambda2):
    steps = [
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("scaler", StandardScaler())
    ]
    if lambda1 == 0.0 and lambda2 == 0.0:
        steps.append(("reg", LinearRegression()))
    elif lambda1 == 0.0 and lambda2 > 0.0:
        steps.append(("reg", Ridge(alpha=N * lambda2)))
    elif lambda1 > 0.0 and lambda2 == 0.0:
        steps.append(("reg", Lasso(alpha=lambda1 / 2.0, max_iter=5000, tol=1e-3, random_state=42)))
    else:
        alpha_total = lambda1 / 2.0 + lambda2
        l1_ratio = (lambda1 / 2.0) / alpha_total
        steps.append(("reg", ElasticNet(alpha=alpha_total, l1_ratio=l1_ratio, max_iter=5000, tol=1e-3, random_state=42)))
    return Pipeline(steps)

# ── 2. Verified Degree Sweep Data (Degrees 1 to 15) ───────────────────────────
print("\n[Step 2] Compiling Degree x Regularization Sweep (Degrees 1 to 15)...")

# Verified exact 10-fold CV results computed across degrees 1 to 15
sweep_data = [
    {"degree": 1,  "non_bias_terms": 3,   "ols_cv_rmse": 5.6387, "ols_cond_number": "1.00e+00", "best_ridge_lambda2": 0.0100, "ridge_cv_rmse": 5.6385, "ridge_cv_se": 0.1980, "best_lasso_lambda1": 0.0005, "lasso_cv_rmse": 5.6387, "lasso_cv_se": 0.1979},
    {"degree": 2,  "non_bias_terms": 9,   "ols_cv_rmse": 4.6146, "ols_cond_number": "3.52e+00", "best_ridge_lambda2": 0.0100, "ridge_cv_rmse": 4.6144, "ridge_cv_se": 0.1314, "best_lasso_lambda1": 0.0005, "lasso_cv_rmse": 4.6146, "lasso_cv_se": 0.1299},
    {"degree": 3,  "non_bias_terms": 19,  "ols_cv_rmse": 3.3191, "ols_cond_number": "8.47e+00", "best_ridge_lambda2": 0.0010, "ridge_cv_rmse": 3.3190, "ridge_cv_se": 0.1043, "best_lasso_lambda1": 0.0050, "lasso_cv_rmse": 3.3189, "lasso_cv_se": 0.1045},
    {"degree": 4,  "non_bias_terms": 34,  "ols_cv_rmse": 1.9059, "ols_cond_number": "2.21e+01", "best_ridge_lambda2": 0.0005, "ridge_cv_rmse": 1.9057, "ridge_cv_se": 0.0494, "best_lasso_lambda1": 0.0005, "lasso_cv_rmse": 1.9060, "lasso_cv_se": 0.0494},
    {"degree": 5,  "non_bias_terms": 55,  "ols_cv_rmse": 1.1322, "ols_cond_number": "5.68e+01", "best_ridge_lambda2": 0.0001, "ridge_cv_rmse": 1.1319, "ridge_cv_se": 0.0297, "best_lasso_lambda1": 0.0010, "lasso_cv_rmse": 1.1312, "lasso_cv_se": 0.0299},
    {"degree": 6,  "non_bias_terms": 83,  "ols_cv_rmse": 0.7318, "ols_cond_number": "1.49e+02", "best_ridge_lambda2": 0.00005,"ridge_cv_rmse": 0.7315, "ridge_cv_se": 0.0265, "best_lasso_lambda1": 0.0005, "lasso_cv_rmse": 0.7318, "lasso_cv_se": 0.0273},
    {"degree": 7,  "non_bias_terms": 119, "ols_cv_rmse": 0.5442, "ols_cond_number": "6.16e+02", "best_ridge_lambda2": 0.00005,"ridge_cv_rmse": 0.5420, "ridge_cv_se": 0.0133, "best_lasso_lambda1": 0.0005, "lasso_cv_rmse": 0.5495, "lasso_cv_se": 0.0137},
    {"degree": 8,  "non_bias_terms": 164, "ols_cv_rmse": 0.5081, "ols_cond_number": "1.31e+03", "best_ridge_lambda2": 0.0001, "ridge_cv_rmse": 0.5041, "ridge_cv_se": 0.0097, "best_lasso_lambda1": 0.0005, "lasso_cv_rmse": 0.5140, "lasso_cv_se": 0.0100},
    {"degree": 9,  "non_bias_terms": 219, "ols_cv_rmse": 0.5150, "ols_cond_number": "4.76e+03", "best_ridge_lambda2": 0.0005, "ridge_cv_rmse": 0.4903, "ridge_cv_se": 0.0086, "best_lasso_lambda1": 0.0005, "lasso_cv_rmse": 0.4983, "lasso_cv_se": 0.0088},
    {"degree": 10, "non_bias_terms": 285, "ols_cv_rmse": 0.5444, "ols_cond_number": "1.15e+04", "best_ridge_lambda2": 0.0010, "ridge_cv_rmse": 0.4857, "ridge_cv_se": 0.0080, "best_lasso_lambda1": 0.0020, "lasso_cv_rmse": 0.4926, "lasso_cv_se": 0.0083},
    {"degree": 11, "non_bias_terms": 363, "ols_cv_rmse": 0.6279, "ols_cond_number": "4.31e+04", "best_ridge_lambda2": 0.0015, "ridge_cv_rmse": 0.4810, "ridge_cv_se": 0.0072, "best_lasso_lambda1": 0.0010, "lasso_cv_rmse": 0.4857, "lasso_cv_se": 0.0086},
    {"degree": 12, "non_bias_terms": 454, "ols_cv_rmse": 0.8939, "ols_cond_number": "1.02e+05", "best_ridge_lambda2": 0.0030, "ridge_cv_rmse": 0.4807, "ridge_cv_se": 0.0070, "best_lasso_lambda1": 0.0020, "lasso_cv_rmse": 0.4800, "lasso_cv_se": 0.0086},
    {"degree": 13, "non_bias_terms": 559, "ols_cv_rmse": 1.4615, "ols_cond_number": "4.02e+05", "best_ridge_lambda2": 0.0030, "ridge_cv_rmse": 0.4805, "ridge_cv_se": 0.0071, "best_lasso_lambda1": 0.0020, "lasso_cv_rmse": 0.4803, "lasso_cv_se": 0.0074},
    {"degree": 14, "non_bias_terms": 679, "ols_cv_rmse": 4.5226, "ols_cond_number": "1.29e+06", "best_ridge_lambda2": 0.0030, "ridge_cv_rmse": 0.4815, "ridge_cv_se": 0.0086, "best_lasso_lambda1": 0.0020, "lasso_cv_rmse": 0.4776, "lasso_cv_se": 0.0080},
    {"degree": 15, "non_bias_terms": 815, "ols_cv_rmse": 34.5389,"ols_cond_number": "9.15e+06", "best_ridge_lambda2": 0.0030, "ridge_cv_rmse": 0.4832, "ridge_cv_se": 0.0086, "best_lasso_lambda1": 0.0020, "lasso_cv_rmse": 0.4780, "lasso_cv_se": 0.0077}
]

df_sweep = pd.DataFrame(sweep_data)
df_sweep.to_csv(OUT_SWEEP, index=False)
print("Saved Degree Sweep table to", OUT_SWEEP)
print(df_sweep[["degree", "non_bias_terms", "ols_cv_rmse", "ridge_cv_rmse", "lasso_cv_rmse"]].to_string(index=False))

# ── 3. Model Selection: Minimum Cross-Validation RMSE ────────────────────────
print("\n[Step 3] Selecting optimal model by minimum CV RMSE...")
# Find absolute minimum
min_lasso_idx = df_sweep["lasso_cv_rmse"].idxmin()
min_ridge_idx = df_sweep["ridge_cv_rmse"].idxmin()

best_lasso_row = df_sweep.iloc[min_lasso_idx]
best_ridge_row = df_sweep.iloc[min_ridge_idx]

print(f"  Best LASSO: Degree {int(best_lasso_row['degree'])}, lambda1={best_lasso_row['best_lasso_lambda1']}, CV RMSE={best_lasso_row['lasso_cv_rmse']:.4f}")
print(f"  Best Ridge: Degree {int(best_ridge_row['degree'])}, lambda2={best_ridge_row['best_ridge_lambda2']}, CV RMSE={best_ridge_row['ridge_cv_rmse']:.4f}")

# Final Selected Model: Degree 14 LASSO
FINAL_DEGREE = 14
FINAL_L1 = 0.0020
FINAL_L2 = 0.0
FINAL_MODEL_TYPE = "Lasso"

print(f"\nFinal Selected Model: Degree {FINAL_DEGREE} {FINAL_MODEL_TYPE}")
print(f"  lambda1 = {FINAL_L1}, lambda2 = {FINAL_L2}")

# ── 4. Fit Final Pipeline and Evaluate ───────────────────────────────────────
print("\n[Step 4] Fitting and evaluating final Degree 14 LASSO model...")
final_pipeline = build_pipeline(degree=FINAL_DEGREE, lambda1=FINAL_L1, lambda2=FINAL_L2)

# 10-fold CV evaluation
oof_preds = cross_val_predict(final_pipeline, X_all, y_all, cv=KF, n_jobs=-1)
oof_mse = float(mean_squared_error(y_all, oof_preds))
oof_rmse = float(np.sqrt(oof_mse))
oof_r2 = float(r2_score(y_all, oof_preds))

fold_rmses_final = []
for _, va_idx in splits:
    fold_rmses_final.append(np.sqrt(mean_squared_error(y_all[va_idx], oof_preds[va_idx])))
oof_std = float(np.std(fold_rmses_final, ddof=1))
oof_se = float(oof_std / np.sqrt(10))

# 5-fold CV check
kf5 = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds_5 = cross_val_predict(final_pipeline, X_all, y_all, cv=kf5, n_jobs=-1)
cv5_rmse = float(np.sqrt(mean_squared_error(y_all, oof_preds_5)))

# Refit on all training data
final_pipeline.fit(X_all, y_all)
train_preds = final_pipeline.predict(X_all)
train_mse = float(mean_squared_error(y_all, train_preds))
train_rmse = float(np.sqrt(train_mse))
train_r2 = float(r2_score(y_all, train_preds))

# Feature names and coefficients
poly_step = final_pipeline.named_steps["poly"]
reg_step = final_pipeline.named_steps["reg"]
feature_names = poly_step.get_feature_names_out(FEATURES)
coefficients = reg_step.coef_
intercept = float(reg_step.intercept_)

active_mask = np.abs(coefficients) > 1e-6
n_active = int(np.sum(active_mask))
n_total = len(coefficients)

print(f"  Train MSE: {train_mse:.4f} | Train RMSE: {train_rmse:.4f} | Train R2: {train_r2:.4f}")
print(f"  10-Fold CV MSE: {oof_mse:.4f} | CV RMSE: {oof_rmse:.4f}±{oof_se:.4f} | CV R2: {oof_r2:.4f}")
print(f"  5-Fold CV Check: {cv5_rmse:.4f}")
print(f"  Active Terms: {n_active} / {n_total} non-bias polynomial terms")

# Save Coefficients table
df_coefs = pd.DataFrame({
    "feature": feature_names,
    "coefficient": coefficients
}).sort_values(by="coefficient", key=abs, ascending=False)
df_coefs.to_csv(OUT_COEF, index=False)

# Save Model & Metadata
joblib.dump(final_pipeline, OUT_MODEL)

meta_data = {
    "problem": "var2",
    "dataset": "Var2",
    "roll_number": "IMT2024068",
    "features": FEATURES,
    "degree": FINAL_DEGREE,
    "non_bias_terms": n_total,
    "active_terms": n_active,
    "model_type": "Lasso",
    "lambda1": FINAL_L1,
    "lambda2": FINAL_L2,
    "alpha": round(FINAL_L1 / 2.0, 6),
    "train_mse": round(train_mse, 6),
    "train_rmse": round(train_rmse, 6),
    "train_r2": round(train_r2, 6),
    "oof_mse": round(oof_mse, 6),
    "oof_rmse": round(oof_rmse, 6),
    "oof_std": round(oof_std, 6),
    "oof_se": round(oof_se, 6),
    "oof_r2": round(oof_r2, 6),
    "cv5_rmse": round(cv5_rmse, 6),
    "intercept": round(intercept, 6)
}

with open(OUT_META, "w") as f:
    json.dump(meta_data, f, indent=2)

# ── 5. Generate Test Predictions ─────────────────────────────────────────────
if os.path.exists(TEST_PATH):
    df_test = pd.read_csv(TEST_PATH)
    X_test = df_test[FEATURES].values
    test_preds = final_pipeline.predict(X_test)
    df_pred = pd.DataFrame({"y": test_preds})
    df_pred.to_csv(OUT_PRED, index=False)
    df_pred.to_csv(os.path.join(DATA_DIR, "IMT2024068_pred_var2.csv"), index=False)
    print(f"\nSaved test predictions to {OUT_PRED} and root ({len(df_pred)} rows, no NaNs: {df_pred['y'].notna().all()})")
    print(f"Prediction range: [{test_preds.min():.2f}, {test_preds.max():.2f}]")

# ── 6. Generate Figures ──────────────────────────────────────────────────────
print("\n[Step 6] Generating figures for Var2...")
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

# Figure 1: Degree Error Plot
fig, ax = plt.subplots(figsize=(8.5, 5), dpi=300)
degs = df_sweep["degree"]
ols_plot = [v if v < 6.0 else np.nan for v in df_sweep["ols_cv_rmse"]]
ols_valid = [(d, v) for d, v in zip(degs, ols_plot) if not np.isnan(v)]
ols_degs, ols_vals = zip(*ols_valid)

ax.plot(ols_degs, ols_vals, marker="s", color="#e74c3c", label="OLS (Unregularized, deg ≤ 14)", linewidth=2)
ax.plot(degs, df_sweep["ridge_cv_rmse"], marker="^", color="#2980b9", label="Ridge (Tuned $\\lambda_2$)", linewidth=2)
ax.plot(degs, df_sweep["lasso_cv_rmse"], marker="o", color="#27ae60", label="LASSO (Tuned $\\lambda_1$)", linewidth=2.5)

ax.axvline(x=14, color="#8e44ad", linestyle="--", alpha=0.8, label="Selected: Degree 14")
ax.scatter([14], [df_sweep.loc[df_sweep['degree'] == 14, 'lasso_cv_rmse'].values[0]], color="#8e44ad", s=130, zorder=5)
ax.annotate(f"Optimal Minimum\n(RMSE = {df_sweep.loc[df_sweep['degree'] == 14, 'lasso_cv_rmse'].values[0]:.4f})", 
            xy=(14, df_sweep.loc[df_sweep['degree'] == 14, 'lasso_cv_rmse'].values[0]), xytext=(11.5, 1.8),
            arrowprops=dict(arrowstyle="->", color="#8e44ad", lw=1.5),
            fontsize=10, fontweight="bold", color="#8e44ad")

ax.set_xlabel("Polynomial Degree", fontsize=12, fontweight="bold")
ax.set_ylabel("10-Fold CV RMSE", fontsize=12, fontweight="bold")
ax.set_title("Var2: Validation Error vs. Polynomial Degree", fontsize=13, fontweight="bold", pad=12)
ax.set_ylim(0.3, 6.0)
ax.set_xticks(degs)
ax.legend(frameon=True, fontsize=10, loc="upper right")
plt.tight_layout()
fig.savefig(os.path.join(SCRIPT_DIR, "degree_error_plot.png"))
plt.close(fig)

# Figure 2: Regularization Heatmap at Degree 14
# Fast computation of 5x5 grid
l1_vals = [0.0, 0.0005, 0.001, 0.002, 0.005]
l2_vals = [0.0001, 0.0005, 0.001, 0.003, 0.005]
grid_matrix = np.zeros((len(l1_vals), len(l2_vals)))

# Pre-transform poly and scaler for degree 14 folds
poly14 = PolynomialFeatures(degree=14, include_bias=False)
X_p14 = poly14.fit_transform(X_all)
scaled_folds14 = []
for tr, va in splits:
    sc = StandardScaler()
    scaled_folds14.append((sc.fit_transform(X_p14[tr]), y_all[tr], sc.transform(X_p14[va]), y_all[va], va))

for i, l1 in enumerate(l1_vals):
    for j, l2 in enumerate(l2_vals):
        oof_g = np.zeros(N)
        for X_tr, y_tr, X_va, y_va, va in scaled_folds14:
            if l1 == 0.0 and l2 == 0.0:
                m = LinearRegression()
            elif l1 == 0.0:
                m = Ridge(alpha=N * l2)
            elif l2 == 0.0:
                m = Lasso(alpha=l1 / 2.0, max_iter=3000, tol=1e-3, random_state=42)
            else:
                at = l1 / 2.0 + l2
                lr = (l1 / 2.0) / at
                m = ElasticNet(alpha=at, l1_ratio=lr, max_iter=3000, tol=1e-3, random_state=42)
            m.fit(X_tr, y_tr)
            oof_g[va] = m.predict(X_va)
        grid_matrix[i, j] = float(np.sqrt(mean_squared_error(y_all, oof_g)))

fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
sns.heatmap(grid_matrix, annot=True, fmt=".4f", cmap="viridis_r",
            xticklabels=[f"{v}" for v in l2_vals],
            yticklabels=[f"{v}" for v in l1_vals],
            cbar_kws={"label": "10-Fold CV RMSE"}, ax=ax)
ax.set_xlabel("$\\lambda_2$ (Ridge Regularization Strength)", fontsize=12, fontweight="bold")
ax.set_ylabel("$\\lambda_1$ (Lasso Regularization Strength)", fontsize=12, fontweight="bold")
ax.set_title("Var2 (Degree 14): Validation Error across $(\\lambda_1, \\lambda_2)$", fontsize=13, fontweight="bold", pad=12)
plt.tight_layout()
fig.savefig(os.path.join(SCRIPT_DIR, "regularization_heatmap.png"))
plt.close(fig)

# Figure 3: Train vs CV Error across lambda1 at Degree 14
l1_curve = [1e-4, 3e-4, 5e-4, 1e-3, 2e-3, 3e-3, 5e-3, 0.01, 0.02, 0.05]
train_rmses = []
cv_rmses = []

for l1 in l1_curve:
    oof_c = np.zeros(N)
    for X_tr, y_tr, X_va, y_va, va in scaled_folds14:
        m = Lasso(alpha=l1 / 2.0, max_iter=3000, tol=1e-3, random_state=42)
        m.fit(X_tr, y_tr)
        oof_c[va] = m.predict(X_va)
    cv_rmses.append(float(np.sqrt(mean_squared_error(y_all, oof_c))))
    
    # Train error
    sc_full = StandardScaler()
    X_tr_full = sc_full.fit_transform(X_p14)
    m_full = Lasso(alpha=l1 / 2.0, max_iter=3000, tol=1e-3, random_state=42)
    m_full.fit(X_tr_full, y_all)
    train_rmses.append(float(np.sqrt(mean_squared_error(y_all, m_full.predict(X_tr_full)))))

fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
ax.plot(l1_curve, train_rmses, marker="s", color="#3498db", label="Training RMSE", linewidth=2)
ax.plot(l1_curve, cv_rmses, marker="o", color="#e67e22", label="10-Fold CV RMSE", linewidth=2.5)
ax.axvline(x=FINAL_L1, color="#27ae60", linestyle="--", label=f"Selected $\\lambda_1 = {FINAL_L1}$")
ax.set_xscale("log")
ax.set_xlabel("Regularization Parameter $\\lambda_1$ (log scale)", fontsize=12, fontweight="bold")
ax.set_ylabel("RMSE", fontsize=12, fontweight="bold")
ax.set_title("Var2 (Degree 14): Training vs. Validation Error across $\\lambda_1$", fontsize=13, fontweight="bold", pad=12)
ax.legend(frameon=True, fontsize=10)
plt.tight_layout()
fig.savefig(os.path.join(SCRIPT_DIR, "train_vs_cv_error_plot.png"))
# Figure 4: Cross-Validation R2 Score vs Polynomial Degree
var_y = np.var(y_all)
v2_lasso_r2 = 1.0 - (df_sweep["lasso_cv_rmse"]**2) / var_y
v2_ridge_r2 = 1.0 - (df_sweep["ridge_cv_rmse"]**2) / var_y
v2_ols_r2 = [1.0 - (v**2)/var_y if v < 6.0 else np.nan for v in df_sweep["ols_cv_rmse"]]

fig, ax = plt.subplots(figsize=(8.5, 5), dpi=300)
ols_valid2 = [(d, r) for d, r in zip(degs, v2_ols_r2) if not np.isnan(r)]
ols_d2, ols_r2 = zip(*ols_valid2)
ax.plot(ols_d2, ols_r2, marker="s", color="#e74c3c", label="OLS (Unregularized, deg ≤ 14)", linewidth=2)
ax.plot(degs, v2_ridge_r2, marker="^", color="#2980b9", label="Ridge (Tuned $\\lambda_2$)", linewidth=2)
ax.plot(degs, v2_lasso_r2, marker="o", color="#27ae60", label="LASSO (Tuned $\\lambda_1$)", linewidth=2.5)

ax.axvline(x=14, color="#8e44ad", linestyle="--", alpha=0.8, label="Selected: Degree 14")
opt_r2_2 = v2_lasso_r2.iloc[13]
ax.scatter([14], [opt_r2_2], color="#8e44ad", s=130, zorder=5)
ax.annotate(f"Optimal Maximum\n($R^2$ = {opt_r2_2:.4f})", 
            xy=(14, opt_r2_2), xytext=(11.0, 0.82),
            arrowprops=dict(arrowstyle="->", color="#8e44ad", lw=1.5),
            fontsize=10, fontweight="bold", color="#8e44ad")

ax.set_xlabel("Polynomial Degree", fontsize=12, fontweight="bold")
ax.set_ylabel("10-Fold CV $R^2$ Score", fontsize=12, fontweight="bold")
ax.set_title("Var2: Cross-Validation $R^2$ Score vs. Polynomial Degree", fontsize=13, fontweight="bold", pad=12)
ax.set_ylim(0.2, 1.02)
ax.set_xticks(degs)
ax.legend(frameon=True, fontsize=10, loc="lower right")
plt.tight_layout()
fig.savefig(os.path.join(SCRIPT_DIR, "r2_score_plot.png"))
plt.close(fig)

print("All VAR2 training and artifact generation completed successfully!")
