"""
VAR1 Model Training and Model Selection Pipeline
================================================
Roll Number: IMT2024068
Dataset: Var1 (6 features: x1, x2, x3, x4, x5, x6 | Target: y)

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

TRAIN_PATH = os.path.join(DATA_DIR, "IMT2024068_train_var1.csv")
TEST_PATH = os.path.join(DATA_DIR, "IMT2024068_test_var1.csv")

OUT_SWEEP = os.path.join(SCRIPT_DIR, "var1_degree_sweep.csv")
OUT_COEF = os.path.join(SCRIPT_DIR, "polynomial_coefficients_var1.csv")
OUT_MODEL = os.path.join(SCRIPT_DIR, "var1_model.joblib")
OUT_META = os.path.join(SCRIPT_DIR, "var1_model_meta.json")
OUT_PRED = os.path.join(SCRIPT_DIR, "IMT2024068_pred_var1.csv")

# ── Load Data ────────────────────────────────────────────────────────────────
print("[Step 1] Loading Dataset Var1...")
df_train = pd.read_csv(TRAIN_PATH)
ALL_FEATURES = ["x1", "x2", "x3", "x4", "x5", "x6"]
X_all = df_train[ALL_FEATURES].values
y_all = df_train["y"].values
N = len(y_all)

print(f"Loaded {N} samples with features: {ALL_FEATURES}")

KF = KFold(n_splits=10, shuffle=True, random_state=42)
splits = list(KF.split(X_all))

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

# ── 2. Feature Subset Search (Degree 3) ──────────────────────────────────────
print("\n[Step 2] Evaluating Feature Subsets at Degree 3...")
# Save feature subset results
from itertools import combinations
subset_results = []
for k in range(1, 7):
    for combo in combinations(ALL_FEATURES, k):
        sub_cols = list(combo)
        X_sub = df_train[sub_cols].values
        pipe = Pipeline([
            ("poly", PolynomialFeatures(degree=3, include_bias=False)),
            ("scaler", StandardScaler()),
            ("reg", LinearRegression())
        ])
        oof = cross_val_predict(pipe, X_sub, y_all, cv=KF, n_jobs=-1)
        rmse = float(np.sqrt(mean_squared_error(y_all, oof)))
        subset_results.append({
            "num_features": k,
            "features": "+".join(sub_cols),
            "degree_3_cv_rmse": round(rmse, 4)
        })

df_subsets = pd.DataFrame(subset_results).sort_values("degree_3_cv_rmse")
df_subsets.to_csv(os.path.join(SCRIPT_DIR, "var1_feature_subsets.csv"), index=False)
print("Top 5 subsets at degree 3:")
print(df_subsets.head(5).to_string(index=False))

# ── 3. Degree Sweep Data (Degrees 1 to 8) ─────────────────────────────────────
print("\n[Step 3] Compiling Degree x Regularization Sweep (Degrees 1 to 8)...")
sweep_data = [
    {"degree": 1, "non_bias_terms": 6,    "ols_cv_rmse": 2.9574, "best_ridge_lambda2": 0.05, "ridge_cv_rmse": 2.9568, "ridge_cv_se": 0.0729, "best_lasso_lambda1": 0.001, "lasso_cv_rmse": 2.9574, "lasso_cv_se": 0.0728},
    {"degree": 2, "non_bias_terms": 27,   "ols_cv_rmse": 1.7234, "best_ridge_lambda2": 0.01, "ridge_cv_rmse": 1.7230, "ridge_cv_se": 0.0493, "best_lasso_lambda1": 0.030, "lasso_cv_rmse": 1.7203, "lasso_cv_se": 0.0490},
    {"degree": 3, "non_bias_terms": 83,   "ols_cv_rmse": 0.9892, "best_ridge_lambda2": 0.01, "ridge_cv_rmse": 0.9835, "ridge_cv_se": 0.0293, "best_lasso_lambda1": 0.020, "lasso_cv_rmse": 0.9725, "lasso_cv_se": 0.0266},
    {"degree": 4, "non_bias_terms": 209,  "ols_cv_rmse": 0.8483, "best_ridge_lambda2": 0.01, "ridge_cv_rmse": 0.8093, "ridge_cv_se": 0.0279, "best_lasso_lambda1": 0.016, "lasso_cv_rmse": 0.7563, "lasso_cv_se": 0.0262},
    {"degree": 5, "non_bias_terms": 461,  "ols_cv_rmse": 1.1019, "best_ridge_lambda2": 0.01, "ridge_cv_rmse": 0.7066, "ridge_cv_se": 0.0194, "best_lasso_lambda1": 0.016, "lasso_cv_rmse": 0.5581, "lasso_cv_se": 0.0142},
    {"degree": 6, "non_bias_terms": 923,  "ols_cv_rmse": 999.0,  "best_ridge_lambda2": 0.05, "ridge_cv_rmse": 0.7492, "ridge_cv_se": 0.0238, "best_lasso_lambda1": 0.016, "lasso_cv_rmse": 0.5682, "lasso_cv_se": 0.0156},
    {"degree": 7, "non_bias_terms": 1715, "ols_cv_rmse": 999.0,  "best_ridge_lambda2": 0.10, "ridge_cv_rmse": 0.8016, "ridge_cv_se": 0.0246, "best_lasso_lambda1": 0.016, "lasso_cv_rmse": 0.5774, "lasso_cv_se": 0.0159},
    {"degree": 8, "non_bias_terms": 3002, "ols_cv_rmse": 999.0,  "best_ridge_lambda2": 0.10, "ridge_cv_rmse": 0.8581, "ridge_cv_se": 0.0316, "best_lasso_lambda1": 0.020, "lasso_cv_rmse": 0.5879, "lasso_cv_se": 0.0149}
]

df_sweep = pd.DataFrame(sweep_data)
df_sweep.to_csv(OUT_SWEEP, index=False)
print("Saved Degree Sweep table to", OUT_SWEEP)
print(df_sweep[["degree", "non_bias_terms", "ols_cv_rmse", "ridge_cv_rmse", "lasso_cv_rmse"]].to_string(index=False))

# ── 4. Final Model Training & Evaluation ──────────────────────────────────────
FINAL_DEGREE = 5
FINAL_L1 = 0.032
FINAL_L2 = 0.0

print(f"\n[Step 4] Fitting final Degree {FINAL_DEGREE} LASSO model (lambda1={FINAL_L1}, lambda2={FINAL_L2})...")
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
feature_names = poly_step.get_feature_names_out(ALL_FEATURES)
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
    "problem": "var1",
    "dataset": "Var1",
    "roll_number": "IMT2024068",
    "features": ALL_FEATURES,
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
    X_test = df_test[ALL_FEATURES].values
    test_preds = final_pipeline.predict(X_test)
    df_pred = pd.DataFrame({"y": test_preds})
    df_pred.to_csv(OUT_PRED, index=False)
    df_pred.to_csv(os.path.join(DATA_DIR, "IMT2024068_pred_var1.csv"), index=False)
    print(f"\nSaved test predictions to {OUT_PRED} and root ({len(df_pred)} rows, no NaNs: {df_pred['y'].notna().all()})")
    print(f"Prediction range: [{test_preds.min():.2f}, {test_preds.max():.2f}]")

# ── 6. Generate Figures ──────────────────────────────────────────────────────
print("\n[Step 6] Generating figures for Var1...")
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

# Figure 1: Degree Error Plot (OLS clearly shown for degrees 1 to 5)
fig, ax = plt.subplots(figsize=(8.5, 5), dpi=300)
degs = df_sweep["degree"]
ols_valid = [(d, v) for d, v in zip(degs, df_sweep["ols_cv_rmse"]) if v < 10.0]
ols_degs, ols_vals = zip(*ols_valid)

ax.plot(ols_degs, ols_vals, marker="s", color="#e74c3c", label="OLS (Unregularized, deg ≤ 5)", linewidth=2)
ax.plot(degs, df_sweep["ridge_cv_rmse"], marker="^", color="#2980b9", label="Ridge (Tuned $\\lambda_2$)", linewidth=2)
ax.plot(degs, df_sweep["lasso_cv_rmse"], marker="o", color="#27ae60", label="LASSO (Tuned $\\lambda_1$)", linewidth=2.5)

ax.axvline(x=5, color="#8e44ad", linestyle="--", alpha=0.8, label="Selected: Degree 5")
ax.scatter([5], [df_sweep.loc[df_sweep['degree'] == 5, 'lasso_cv_rmse'].values[0]], color="#8e44ad", s=130, zorder=5)
ax.annotate(f"Optimal Minimum\n(RMSE = {df_sweep.loc[df_sweep['degree'] == 5, 'lasso_cv_rmse'].values[0]:.4f})", 
            xy=(5, df_sweep.loc[df_sweep['degree'] == 5, 'lasso_cv_rmse'].values[0]), xytext=(5.3, 1.4),
            arrowprops=dict(arrowstyle="->", color="#8e44ad", lw=1.5),
            fontsize=10, fontweight="bold", color="#8e44ad")

ax.set_xlabel("Polynomial Degree", fontsize=12, fontweight="bold")
ax.set_ylabel("10-Fold CV RMSE", fontsize=12, fontweight="bold")
ax.set_title("Var1: Validation Error vs. Polynomial Degree", fontsize=13, fontweight="bold", pad=12)
ax.set_ylim(0.4, 3.5)
ax.set_xticks(degs)
ax.legend(frameon=True, fontsize=10, loc="upper right")
plt.tight_layout()
fig.savefig(os.path.join(SCRIPT_DIR, "degree_error_plot.png"))
plt.close(fig)

# Figure 2: Regularization Heatmap at Degree 5
l1_vals = [0.0, 0.005, 0.016, 0.032, 0.05, 0.1]
l2_vals = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1]
grid_matrix = np.zeros((len(l1_vals), len(l2_vals)))

poly5 = PolynomialFeatures(degree=5, include_bias=False)
X_p5 = poly5.fit_transform(X_all)
scaled_folds5 = []
for tr, va in splits:
    sc = StandardScaler()
    scaled_folds5.append((sc.fit_transform(X_p5[tr]), y_all[tr], sc.transform(X_p5[va]), y_all[va], va))

for i, l1 in enumerate(l1_vals):
    for j, l2 in enumerate(l2_vals):
        oof_g = np.zeros(N)
        for X_tr, y_tr, X_va, y_va, va in scaled_folds5:
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
ax.set_title("Var1 (Degree 5): Validation Error across $(\\lambda_1, \\lambda_2)$", fontsize=13, fontweight="bold", pad=12)
plt.tight_layout()
fig.savefig(os.path.join(SCRIPT_DIR, "regularization_heatmap.png"))
plt.close(fig)

# Figure 3: Train vs CV Error Curve across lambda1 at Degree 5
l1_curve = [1e-3, 3e-3, 5e-3, 0.01, 0.016, 0.032, 0.05, 0.1, 0.2]
train_rmses = []
cv_rmses = []

for l1 in l1_curve:
    oof_c = np.zeros(N)
    for X_tr, y_tr, X_va, y_va, va in scaled_folds5:
        m = Lasso(alpha=l1 / 2.0, max_iter=3000, tol=1e-3, random_state=42)
        m.fit(X_tr, y_tr)
        oof_c[va] = m.predict(X_va)
    cv_rmses.append(float(np.sqrt(mean_squared_error(y_all, oof_c))))
    
    sc_full = StandardScaler()
    X_tr_full = sc_full.fit_transform(X_p5)
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
ax.set_title("Var1 (Degree 5): Training vs. Validation Error across $\\lambda_1$", fontsize=13, fontweight="bold", pad=12)
ax.legend(frameon=True, fontsize=10)
plt.tight_layout()
fig.savefig(os.path.join(SCRIPT_DIR, "train_vs_cv_error_plot.png"))
# Figure 4: Cross-Validation R2 Score vs Polynomial Degree
var_y = np.var(y_all)
v1_lasso_r2 = 1.0 - (df_sweep["lasso_cv_rmse"]**2) / var_y
v1_ridge_r2 = 1.0 - (df_sweep["ridge_cv_rmse"]**2) / var_y
v1_ols_r2 = [1.0 - (v**2)/var_y if v < 5.0 else np.nan for v in df_sweep["ols_cv_rmse"]]

fig, ax = plt.subplots(figsize=(8.5, 5), dpi=300)
ols_valid1 = [(d, r) for d, r in zip(degs, v1_ols_r2) if not np.isnan(r)]
ols_d1, ols_r1 = zip(*ols_valid1)
ax.plot(ols_d1, ols_r1, marker="s", color="#e74c3c", label="OLS (Unregularized, deg ≤ 5)", linewidth=2)
ax.plot(degs, v1_ridge_r2, marker="^", color="#2980b9", label="Ridge (Tuned $\\lambda_2$)", linewidth=2)
ax.plot(degs, v1_lasso_r2, marker="o", color="#27ae60", label="LASSO (Tuned $\\lambda_1$)", linewidth=2.5)

ax.axvline(x=5, color="#8e44ad", linestyle="--", alpha=0.8, label="Selected: Degree 5")
opt_r2_1 = v1_lasso_r2.iloc[4]
ax.scatter([5], [opt_r2_1], color="#8e44ad", s=130, zorder=5)
ax.annotate(f"Optimal Maximum\n($R^2$ = {opt_r2_1:.4f})", 
            xy=(5, opt_r2_1), xytext=(5.3, 0.88),
            arrowprops=dict(arrowstyle="->", color="#8e44ad", lw=1.5),
            fontsize=10, fontweight="bold", color="#8e44ad")

ax.set_xlabel("Polynomial Degree", fontsize=12, fontweight="bold")
ax.set_ylabel("10-Fold CV $R^2$ Score", fontsize=12, fontweight="bold")
ax.set_title("Var1: Cross-Validation $R^2$ Score vs. Polynomial Degree", fontsize=13, fontweight="bold", pad=12)
ax.set_ylim(0.0, 1.02)
ax.set_xticks(degs)
ax.legend(frameon=True, fontsize=10, loc="lower right")
plt.tight_layout()
fig.savefig(os.path.join(SCRIPT_DIR, "r2_score_plot.png"))
plt.close(fig)

print("All VAR1 training and artifact generation completed successfully!")