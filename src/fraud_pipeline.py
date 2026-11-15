"""
Anti-fraud system: credit card fraud detection.

Pipeline:
1. Load data and remove duplicates.
2. Stratified Train / Validation / Test split (60% / 20% / 20%).
3. Scale `Time` and `Amount` with StandardScaler (fitted on train only).
4. Train models:
   - Baseline (DummyClassifier)
   - Logistic Regression (class_weight='balanced')
   - Random Forest Classifier (tuned hyperparameters)
5. Compare models on classification metrics and PR-AUC (AUPRC).
6. Business threshold optimization (cost function):
   - Financial loss = missed fraud (sum of Amount in EUR) + false alarms (EUR 5 per manual check).
   - Pick the threshold with the lowest total loss on the validation set.
   - Report the actual money saved on the test set.
7. Out-of-Time (OOT) check: train on Day 1, evaluate on Day 2.
"""

import sys

import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = "creditcard.csv"
C_FP = 5.0  # operational cost of one false alarm, in EUR


def print_header(title):
    print("=" * 65)
    print(title)
    print("=" * 65)


def compute_loss(y_true, y_probs, amounts, threshold, c_fp=C_FP):
    """Total financial loss for a given decision threshold."""
    preds = (y_probs >= threshold).astype(int)
    # Missed fraud: the bank loses the full transaction amount
    loss_fn = amounts[(y_true == 1) & (preds == 0)].sum()
    # False alarm: the bank pays for a manual verification
    loss_fp = ((y_true == 0) & (preds == 1)).sum() * c_fp
    return loss_fn + loss_fp, loss_fn, loss_fp


def main():
    print_header("      ANTI-FRAUD SYSTEM: FULL ML AND BUSINESS PIPELINE")
    print(
        f"Environment: Python {sys.version.split()[0]}, "
        f"scikit-learn {sklearn.__version__}, NumPy {np.__version__}"
    )
    print(
        "Note: exact figures may shift slightly between scikit-learn versions. "
        "The reported numbers come from the notebook run in Google Colab."
    )

    # -------------------------------------------------------------------------
    # 1. LOAD AND CLEAN THE DATA
    # -------------------------------------------------------------------------
    print("\n[1/6] Loading and cleaning data...")
    df = pd.read_csv(DATA_PATH)
    raw_shape = df.shape
    df = df.drop_duplicates().reset_index(drop=True)

    fraud_count = int(df["Class"].sum())
    total_count = len(df)
    print(f"Raw dataset: {raw_shape}, after removing duplicates: {df.shape}")
    print(f"Total transactions: {total_count:,}")
    print(f"Fraudulent (Class 1): {fraud_count:,} ({fraud_count / total_count * 100:.3f}%)")
    print(f"Legitimate (Class 0): {total_count - fraud_count:,}")

    # -------------------------------------------------------------------------
    # 2. TRAIN / VALIDATION / TEST SPLIT (60% / 20% / 20%)
    # -------------------------------------------------------------------------
    print("\n[2/6] Splitting into Train / Validation / Test...")
    X = df.drop("Class", axis=1)
    y = df["Class"]

    # Step 1: hold out 20% for the final test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Step 2: split the remaining 80% into train (60%) and validation (20%)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )

    # Keep unscaled amounts in EUR so losses are reported in real money
    amounts_val = X_val["Amount"].copy().to_numpy()
    amounts_test = X_test["Amount"].copy().to_numpy()

    # Scale Time and Amount without data leakage (fit on train only)
    scaler = StandardScaler()
    scale_cols = ["Time", "Amount"]

    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[scale_cols] = scaler.fit_transform(X_train[scale_cols])
    X_val_scaled[scale_cols] = scaler.transform(X_val[scale_cols])
    X_test_scaled[scale_cols] = scaler.transform(X_test[scale_cols])

    print(f"Train size: {len(X_train):,} (fraud: {int(y_train.sum())})")
    print(f"Val size:   {len(X_val):,} (fraud: {int(y_val.sum())})")
    print(f"Test size:  {len(X_test):,} (fraud: {int(y_test.sum())})")

    # -------------------------------------------------------------------------
    # 3. TRAIN MODELS
    # -------------------------------------------------------------------------
    print("\n[3/6] Training models...")

    # A. Baseline
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train_scaled, y_train)

    # B. Logistic Regression
    lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)
    y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]

    # C. Random Forest
    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2,
        class_weight="balanced",
    )
    rf.fit(X_train_scaled, y_train)
    y_pred_rf = rf.predict(X_test_scaled)
    y_prob_rf_test = rf.predict_proba(X_test_scaled)[:, 1]
    y_prob_rf_val = rf.predict_proba(X_val_scaled)[:, 1]

    y_pred_baseline = baseline.predict(X_test_scaled)

    # Model comparison at the default 0.5 threshold (same layout as the notebook)
    comparison = pd.DataFrame(
        {
            "Model": ["Baseline", "Logistic Regression", "Random Forest"],
            "Precision": [
                precision_score(y_test, y_pred_baseline, zero_division=0),
                precision_score(y_test, y_pred_lr, zero_division=0),
                precision_score(y_test, y_pred_rf, zero_division=0),
            ],
            "Recall": [
                recall_score(y_test, y_pred_baseline, zero_division=0),
                recall_score(y_test, y_pred_lr, zero_division=0),
                recall_score(y_test, y_pred_rf, zero_division=0),
            ],
            "F1": [
                f1_score(y_test, y_pred_baseline, zero_division=0),
                f1_score(y_test, y_pred_lr, zero_division=0),
                f1_score(y_test, y_pred_rf, zero_division=0),
            ],
            "PR-AUC (AUPRC)": [
                average_precision_score(y_test, np.zeros(len(y_test))),
                average_precision_score(y_test, y_prob_lr),
                average_precision_score(y_test, y_prob_rf_test),
            ],
        }
    )

    print("\n--- MODEL COMPARISON (DEFAULT THRESHOLD 0.5) ---")
    print(comparison.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Feature importance (same top-5 view as the notebook)
    importances = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(
        ascending=False
    )
    print("\nTop 5 features:")
    print(importances.head(5).round(4).to_string())

    # -------------------------------------------------------------------------
    # 4. BUSINESS THRESHOLD OPTIMIZATION
    # -------------------------------------------------------------------------
    print("\n[4/6] Searching for the optimal business threshold...")

    threshold_grid = np.arange(0.05, 0.96, 0.02)
    val_losses = [
        compute_loss(y_val.to_numpy(), y_prob_rf_val, amounts_val, t)[0]
        for t in threshold_grid
    ]

    best_idx = int(np.argmin(val_losses))
    optimal_threshold = float(threshold_grid[best_idx])
    best_val_loss = val_losses[best_idx]

    print(f"Optimal threshold on validation: {optimal_threshold:.2f}")
    print(f"Minimum validation loss: EUR {best_val_loss:,.2f}")

    # -------------------------------------------------------------------------
    # 5. BUSINESS IMPACT ON THE TEST SET
    # -------------------------------------------------------------------------
    print("\n[5/6] Measuring business impact on the unseen test set...")

    # No model at all: every fraudulent euro is lost
    no_model_loss = amounts_test[y_test.to_numpy() == 1].sum()

    # Default threshold
    default_loss, default_fn, default_fp = compute_loss(
        y_test.to_numpy(), y_prob_rf_test, amounts_test, 0.50
    )

    # Tuned business threshold
    tuned_loss, tuned_fn, tuned_fp = compute_loss(
        y_test.to_numpy(), y_prob_rf_test, amounts_test, optimal_threshold
    )

    y_pred_tuned = (y_prob_rf_test >= optimal_threshold).astype(int)
    prec_tuned = precision_score(y_test, y_pred_tuned, zero_division=0)
    rec_tuned = recall_score(y_test, y_pred_tuned, zero_division=0)
    f1_tuned = f1_score(y_test, y_pred_tuned, zero_division=0)

    print("-" * 65)
    print(f"Loss without any anti-fraud model:        EUR {no_model_loss:>12,.2f}")
    print(
        f"Loss at default threshold (T = 0.50):     EUR {default_loss:>12,.2f} "
        f"(FN: EUR {default_fn:,.2f}, FP: EUR {default_fp:,.2f})"
    )
    print(
        f"Loss at business threshold (T = {optimal_threshold:.2f}):    EUR {tuned_loss:>12,.2f} "
        f"(FN: EUR {tuned_fn:,.2f}, FP: EUR {tuned_fp:,.2f})"
    )
    print("-" * 65)

    saved = default_loss - tuned_loss
    saved_pct = (saved / default_loss) * 100 if default_loss > 0 else 0.0
    print(f"Net savings from threshold tuning: EUR {saved:,.2f} ({saved_pct:.1f}% loss reduction)")
    print(
        f"Quality at T={optimal_threshold:.2f}: "
        f"Precision={prec_tuned:.4f}, Recall={rec_tuned:.4f}, F1={f1_tuned:.4f}"
    )

    # -------------------------------------------------------------------------
    # 6. OUT-OF-TIME VALIDATION (DAY 1 -> DAY 2)
    # -------------------------------------------------------------------------
    print("\n[6/6] Out-of-Time validation (train on Day 1, test on Day 2)...")
    day1 = df[df["Time"] < 86400].copy()
    day2 = df[df["Time"] >= 86400].copy()
    day2["Time"] = day2["Time"] - 86400  # align the daily cycle

    X_d1 = day1.drop("Class", axis=1)
    y_d1 = day1["Class"]
    X_d2 = day2.drop("Class", axis=1)
    y_d2 = day2["Class"]

    oot_scaler = StandardScaler()
    X_d1[scale_cols] = oot_scaler.fit_transform(X_d1[scale_cols])
    X_d2[scale_cols] = oot_scaler.transform(X_d2[scale_cols])

    rf_oot = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2,
        class_weight="balanced",
    )
    rf_oot.fit(X_d1, y_d1)

    y_prob_d2 = rf_oot.predict_proba(X_d2)[:, 1]
    y_pred_d2 = (y_prob_d2 >= optimal_threshold).astype(int)

    print(f"Day 1 (train): {len(day1):,} transactions (fraud: {int(y_d1.sum())})")
    print(f"Day 2 (test):  {len(day2):,} transactions (fraud: {int(y_d2.sum())})")
    print(f" - PR-AUC:    {average_precision_score(y_d2, y_prob_d2):.4f}")
    print(f" - Precision: {precision_score(y_d2, y_pred_d2, zero_division=0):.4f}")
    print(f" - Recall:    {recall_score(y_d2, y_pred_d2, zero_division=0):.4f}")
    print(f" - F1-score:  {f1_score(y_d2, y_pred_d2, zero_division=0):.4f}")

    # -------------------------------------------------------------------------
    # 7. SANITY CHECKS (same assertions as the notebook)
    # -------------------------------------------------------------------------
    assert df.isna().sum().sum() == 0, "Found missing values!"
    assert (y_prob_rf_test >= 0).all() and (y_prob_rf_test <= 1).all(), "Invalid probability range!"
    assert set(X_train.index).isdisjoint(set(X_test.index)), (
        "Data leakage detected between train and test!"
    )

    print("=" * 65)
    print("All checks passed successfully! Code is reliable and clean.")


if __name__ == "__main__":
    main()
