# Assignment-1: Polynomial Regression and Regularization

**Course Name:** Machine Learning  
**Name:** Divyanshu Ghosh  
**Roll Number:** IMT2024068  
**GitHub Repository:** https://github.com/77-Div-77/ML_Assignment_1  

---

## 1. Project Overview

This assignment implements regularized multivariate polynomial regression across two regression datasets:

- **Dataset Var1:** 1,000 training observations with 6 continuous input features ($x_1, \dots, x_6$).
- **Dataset Var2:** 1,000 training observations with 3 continuous input features ($x_1, \dots, x_3$).

Each problem requires:
1. Feature subset evaluation via 10-fold cross-validation.
2. Degree sweep across polynomial degrees (Var1: Degrees 1–8, Var2: Degrees 1–15) comparing Ordinary Least Squares (OLS), Ridge ($L_2$), and LASSO ($L_1$).
3. Analysis of matrix condition number, multicollinearity, and overfitting in high-degree spaces.
4. Hyperparameter tuning of regularization strengths ($\lambda_1, \lambda_2$).
5. Final model refitting on all 1,000 training observations and generating predictions for 1,000 unlabelled test instances.
6. Complete reproduction via Python scripts, a Jupyter Notebook (`Assignment1_Polynomial_Regression.ipynb`), and a 5-page academic PDF report.

---

## 2. Final Selected Model Configurations

### Var1 Summary (6 Input Features)
- **Selected Polynomial Degree:** Degree 5
- **Input Features Used:** All 6 features ($x_1, x_2, x_3, x_4, x_5, x_6$)
- **Total Non-Bias Polynomial Terms:** 461 terms ($\binom{6+5}{5} - 1$)
- **Active Retained Terms:** 98 terms (78.7% sparsity via $L_1$ penalty)
- **Model Family:** LASSO ($L_1$)
- **Regularization Penalties:** $\lambda_1 = 0.032, \lambda_2 = 0.0$ (scikit-learn `alpha = 0.016`)
- **Training Metrics:** MSE = 0.2667 | RMSE = 0.5164 | $R^2$ = 0.9722
- **10-Fold CV (OOF) Metrics:** MSE = 0.3333 | RMSE = 0.5773 $\pm$ 0.0129 | $R^2$ = 0.9652
- **5-Fold CV Check RMSE:** 0.5857
- **Generated Prediction File:** `Prediction_Data/IMT2024068_pred_var1.csv` (1,000 rows, header `y`)

### Var2 Summary (3 Input Features)
- **Selected Polynomial Degree:** Degree 14
- **Input Features Used:** All 3 features ($x_1, x_2, x_3$)
- **Total Non-Bias Polynomial Terms:** 679 terms ($\binom{3+14}{14} - 1$)
- **Active Retained Terms:** 298 terms (56.1% sparsity via $L_1$ penalty)
- **Model Family:** LASSO ($L_1$)
- **Regularization Penalties:** $\lambda_1 = 0.002, \lambda_2 = 0.0$ (scikit-learn `alpha = 0.001`)
- **Selection Basis:** Global minimum 10-fold cross-validation RMSE (0.4776) among all tested degrees (1 to 15) and regularization models.
- **Overfitting Verification:** Training RMSE is 0.4010, yielding a minimal generalization gap of 0.0766 against 10-fold CV RMSE (0.4776). L1 regularization zeroes out 381 redundant higher-order interaction terms. At Degree 15, CV RMSE increases to 0.4780, confirming Degree 14 as the optimal sweet spot.
- **Training Metrics:** MSE = 0.1608 | RMSE = 0.4010 | $R^2$ = 0.9964
- **10-Fold CV (OOF) Metrics:** MSE = 0.2281 | RMSE = 0.4776 $\pm$ 0.0080 | $R^2$ = 0.9949
- **5-Fold CV Check RMSE:** 0.4834
- **Generated Prediction File:** `Prediction_Data/IMT2024068_pred_var2.csv` (1,000 rows, header `y`)

---

## 3. Directory & File Structure

```
ML_Assignment/
    ├── README.md                           # Complete project documentation
    ├── requirements.txt                    # Python dependencies
    ├── IMT2024068_Report.pdf               # Exactly 5-page academic PDF report
    ├── Assignment1_Polynomial_Regression.ipynb # Reproducible Jupyter Notebook
    │
    ├── Dataset/                            # Provided dataset directory
    │   ├── IMT2024068_train_var1.csv       # Var1 training data (1,000 samples)
    │   ├── IMT2024068_test_var1.csv        # Var1 unlabelled test data (1,000 samples)
    │   ├── IMT2024068_train_var2.csv       # Var2 training data (1,000 samples)
    │   └── IMT2024068_test_var2.csv        # Var2 unlabelled test data (1,000 samples)
    │
    ├── Prediction_Data/                    # Generated test prediction files
    │   ├── IMT2024068_pred_var1.csv        # Final Var1 test predictions (1,000 rows, header 'y')
    │   └── IMT2024068_pred_var2.csv        # Final Var2 test predictions (1,000 rows, header 'y')
    │
    ├── var1/
    │   ├── train_var1.py                   # Var1 full training & artifact generation
    │   ├── predict_var1.py                 # Var1 inference script for test set
    │   ├── var1_model.joblib               # Saved scikit-learn final model pipeline
    │   ├── var1_model_meta.json            # Final hyperparameters and evaluation metrics
    │   ├── polynomial_coefficients_var1.csv# Saved polynomial coefficients
    │   ├── var1_feature_subsets.csv        # 10-fold CV RMSE for all 63 feature subsets
    │   ├── var1_degree_sweep.csv           # 10-fold CV RMSE across degrees 1 to 8
    │   ├── degree_error_plot.png           # Plot: CV RMSE vs degree (OLS deg 1-5, Ridge, LASSO)
    │   ├── r2_score_plot.png               # Plot: 10-fold CV R^2 score vs degree
    │   ├── regularization_heatmap.png      # Plot: 2D regularization grid (lambda1 vs lambda2)
    │   └── train_vs_cv_error_plot.png      # Plot: Training vs CV RMSE across lambda1
    │
    └── var2/
        ├── train_var2.py                   # Var2 full training & artifact generation
        ├── predict_var2.py                 # Var2 inference script for test set
        ├── var2_model.joblib               # Saved scikit-learn final model pipeline
        ├── var2_model_meta.json            # Final hyperparameters and evaluation metrics
        ├── polynomial_coefficients_var2.csv# Saved polynomial coefficients
        ├── var2_feature_subsets.csv        # 10-fold CV RMSE for all feature subsets
        ├── var2_degree_sweep.csv           # 10-fold CV RMSE across degrees 1 to 15 (with OLS cond. No.)
        ├── degree_error_plot.png           # Plot: CV RMSE vs degree (OLS, Ridge, LASSO, optimal deg 14)
        ├── r2_score_plot.png               # Plot: 10-fold CV R^2 score vs degree
        ├── regularization_heatmap.png      # Plot: 2D regularization grid at degree 14
        └── train_vs_cv_error_plot.png      # Plot: Training vs CV RMSE across lambda1
```

---

## 4. Environment & Installation

The project uses standard Python packages and runs on CPU (no GPU, PyTorch, or CUDA required).

```bash
pip install -r requirements.txt
```

Core dependencies:
- `numpy >= 1.24.0`
- `pandas >= 2.0.0`
- `scikit-learn >= 1.3.0`
- `matplotlib >= 3.7.0`
- `seaborn >= 0.12.0`
- `joblib >= 1.3.0`

---

## 5. Execution Instructions

### A. Run Inference Scripts (Predictions)
```bash
python var1/predict_var1.py
python var2/predict_var2.py
```
This immediately generates the predictions into `Prediction_Data/IMT2024068_pred_var1.csv` and `Prediction_Data/IMT2024068_pred_var2.csv`.

### B. Run Training Pipelines
```bash
python var1/train_var1.py
python var2/train_var2.py
```
This runs full feature subset analysis, degree sweeps, fits final models, and saves evaluation metrics and plots.

### C. Run the Jupyter Notebook
```bash
jupyter notebook Assignment1_Polynomial_Regression.ipynb
```
The notebook executes end-to-end and reproduces all metrics and predictions identically.

---

## 6. Verification Summary

| Check Item | Requirement | Actual Status |
| :--- | :--- | :--- |
| **Var1 Prediction Count** | Exactly 1,000 rows | 1,000 rows (`IMT2024068_pred_var1.csv`) |
| **Var2 Prediction Count** | Exactly 1,000 rows | 1,000 rows (`IMT2024068_pred_var2.csv`) |
| **Prediction Columns** | Single column named `y` | Single column `y`, no NaNs |
| **Report Length** | 4 to 5 pages | Exactly 5 pages (`IMT2024068_Report.pdf`) |
| **Report Title** | Exact format | `Assignment-1: Polynomial Regression and Regularization` |
| **Student Metadata** | Exact format | Course Name: Machine Learning \| Name: Divyanshu Ghosh \| Roll Number: IMT2024068 |
| **Header / Footer** | Clean layout | Header: `Divyanshu Ghosh` \| Footer: `Machine Learning Assignment 1: Polynomial Regression & Regularization` |
| **Jupyter Notebook** | Standalone & reproducible | `Assignment1_Polynomial_Regression.ipynb` created & verified |
