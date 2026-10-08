"""
VAR1 Prediction Script
======================
Generates predictions on IMT2024068_test_var1.csv using the trained polynomial model.
Roll Number: IMT2024068
"""

import os, sys, json
import pandas as pd
import numpy as np
import joblib

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "Dataset") if os.path.exists(os.path.join(PROJECT_ROOT, "Dataset")) else os.path.abspath(os.path.join(PROJECT_ROOT, ".."))

TEST_FILE = os.path.join(DATA_DIR, "IMT2024068_test_var1.csv")
MODEL_FILE = os.path.join(SCRIPT_DIR, "var1_model.joblib") if os.path.exists(os.path.join(SCRIPT_DIR, "var1_model.joblib")) else os.path.join(SCRIPT_DIR, "polynomial_model_var1.pkl")
META_FILE = os.path.join(SCRIPT_DIR, "var1_model_meta.json")

PRED_DIR = os.path.join(PROJECT_ROOT, "Prediction_Data")
os.makedirs(PRED_DIR, exist_ok=True)
OUT_PRED_FILE = os.path.join(PRED_DIR, "IMT2024068_pred_var1.csv")
PARENT_PRED_FILE = os.path.join(PROJECT_ROOT, "..", "IMT2024068_pred_var1.csv")

def main():
    print("=" * 60)
    print("VAR1 PREDICTION GENERATOR")
    print("=" * 60)

    if not os.path.exists(TEST_FILE):
        print(f"Error: Test file not found at {TEST_FILE}")
        sys.exit(1)
        
    if not os.path.exists(MODEL_FILE):
        print(f"Error: Model file not found at {MODEL_FILE}")
        sys.exit(1)

    print(f"Loading test data: {TEST_FILE}")
    df_test = pd.read_csv(TEST_FILE)

    print(f"Loading trained pipeline: {MODEL_FILE}")
    model = joblib.load(MODEL_FILE)

    with open(META_FILE) as f:
        meta = json.load(f)

    features = meta["features"]
    print(f"Features: {features}")
    print(f"Selected Degree: {meta['degree']} | Non-bias terms: {meta['non_bias_terms']} | Active: {meta['active_terms']}")
    print(f"Regularization: lambda1={meta['lambda1']}, lambda2={meta['lambda2']}")

    X_test = df_test[features].values
    y_pred = model.predict(X_test)

    # Verification checks
    assert len(y_pred) == len(df_test), f"Mismatch: expected {len(df_test)}, got {len(y_pred)}"
    assert not np.isnan(y_pred).any(), "Error: NaN values detected in predictions!"

    df_out = pd.DataFrame({"y": y_pred})
    df_out.to_csv(OUT_PRED_FILE, index=False)
    if os.path.exists(os.path.dirname(PARENT_PRED_FILE)):
        df_out.to_csv(PARENT_PRED_FILE, index=False)

    print("\nVerification Passed:")
    print(f"  Total test samples: {len(df_out)}")
    print(f"  Target column name: 'y'")
    print(f"  Prediction range:   [{y_pred.min():.4f}, {y_pred.max():.4f}]")
    print(f"  Mean prediction:    {y_pred.mean():.4f}")
    print(f"  Std prediction:     {y_pred.std():.4f}")
    print(f"  Output saved to:    {OUT_PRED_FILE}")
    print("=" * 60)

if __name__ == "__main__":
    main()
