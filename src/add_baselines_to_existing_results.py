"""
add_baselines_to_existing_results.py
=====================================
Adds baseline comparisons (majority class + logistic regression) and ROC curves
to the EXISTING final test results folder WITHOUT retraining any LSTM model.

This script:
  - Reconstructs the IDENTICAL data split used in run_final_test_trigger50.py
    (same SEED=42, same 70/15/15 chronological split, same feature engineering)
  - Trains only a simple LogisticRegression on the flattened training windows
  - Evaluates majority-class baseline and logistic regression on the test set
  - Saves outputs directly into the existing FINAL_TEST folder subfolders

Run from project root:
    python src/add_baselines_to_existing_results.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# ── Frozen config (must match run_final_test_trigger50.py exactly) ──────────
CRASH_THRESHOLD = 0.03
PREDICTION_TRIGGER = 0.50
WINDOW_SIZE = 24
SEED = 42

# ── Point this at the existing results folder ────────────────────────────────
EXISTING_RESULTS_DIR = Path(
    "results/FINAL_TEST_trigger50_20260326_205807"
)


# ── Data loading (identical to run_final_test_trigger50.py) ─────────────────
def load_and_prepare(file_path: str, feature_columns: list) -> tuple:
    data = pd.read_csv(file_path)
    data["hour"] = pd.to_datetime(data["hour"])
    data = data.sort_values("hour").set_index("hour")

    data["Close_Pct"] = data["Close"].pct_change()
    data["Volume_Pct"] = data["Volume ETH"].pct_change()
    if "massive_whale_volume" in data.columns:
        data["Whale_Vol_Pct"] = data["massive_whale_volume"].pct_change()
    if "High" in data.columns:
        data["High_Pct"] = data["High"].pct_change()

    data = data.dropna().copy()
    data["Target_Crash"] = data["drawdown_6h_label"].astype(int)

    features = data[feature_columns].values
    scaled_features = MinMaxScaler(feature_range=(0, 1)).fit_transform(features)

    X, y = [], []
    for i in range(WINDOW_SIZE, len(scaled_features)):
        X.append(scaled_features[i - WINDOW_SIZE: i])
        y.append(int(data["Target_Crash"].iloc[i]))

    return np.array(X), np.array(y)


def add_baselines(model_name: str, file_path: str, feature_columns: list) -> None:
    print(f"\n{'='*60}")
    print(f"Processing: {model_name}")
    print(f"{'='*60}")

    out_dir = EXISTING_RESULTS_DIR / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Reconstruct identical split ──────────────────────────────────────────
    X, y = load_and_prepare(file_path, feature_columns)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, shuffle=False
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, shuffle=False
    )

    print(f"  Train size : {len(y_train)}  (crashes: {y_train.sum()})")
    print(f"  Val size   : {len(y_val)}   (crashes: {y_val.sum()})")
    print(f"  Test size  : {len(y_test)}  (crashes: {y_test.sum()})")

    # Flatten windows for sklearn models  [n_samples, window*features]
    X_train_flat = X_train.reshape(len(X_train), -1)
    X_test_flat = X_test.reshape(len(X_test), -1)

    # ── Majority-class baseline ──────────────────────────────────────────────
    majority_class = int(np.bincount(y_train).argmax())
    y_pred_majority = np.full(len(y_test), majority_class, dtype=int)

    majority_report = classification_report(
        y_test, y_pred_majority, zero_division=0,
        target_names=["Safe (0)", "Crash (1)"]
    )
    print("\n  [Majority Class Baseline]")
    print(majority_report)

    with open(out_dir / "baseline_majority.txt", "w") as f:
        f.write(f"Baseline: Majority Class (predict always class {majority_class})\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Seed: {SEED}  |  Crash threshold: {CRASH_THRESHOLD}\n\n")
        f.write(majority_report)

    # ── Logistic Regression baseline ─────────────────────────────────────────
    lr = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=SEED,
        solver="lbfgs",
    )
    lr.fit(X_train_flat, y_train)

    y_pred_lr = lr.predict(X_test_flat)
    lr_probs = lr.predict_proba(X_test_flat)[:, 1]

    lr_report = classification_report(
        y_test, y_pred_lr, zero_division=0,
        target_names=["Safe (0)", "Crash (1)"]
    )
    lr_auc = roc_auc_score(y_test, lr_probs)
    print("\n  [Logistic Regression Baseline]")
    print(lr_report)
    print(f"  ROC-AUC: {lr_auc:.4f}")

    with open(out_dir / "baseline_logistic_regression.txt", "w") as f:
        f.write("Baseline: Logistic Regression (class_weight='balanced')\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Seed: {SEED}  |  Crash threshold: {CRASH_THRESHOLD}\n")
        f.write(f"ROC-AUC: {lr_auc:.4f}\n\n")
        f.write(lr_report)

    # ── ROC curve (logistic regression only — LSTM probs not available) ──────
    fpr, tpr, _ = roc_curve(y_test, lr_probs)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color="steelblue", lw=2,
             label=f"Logistic Regression (AUC = {lr_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--",
             label="Random classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve — {model_name}\n(Logistic Regression baseline, test set)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_dir / "roc_curve_baseline.png", dpi=150)
    plt.close()

    print(f"  Saved → {out_dir / 'baseline_majority.txt'}")
    print(f"  Saved → {out_dir / 'baseline_logistic_regression.txt'}")
    print(f"  Saved → {out_dir / 'roc_curve_baseline.png'}")


def main() -> None:
    whale_features = [
        "Close_Pct",
        "Volume_Pct",
        "Whale_Vol_Pct",
        "max_gas_gwei",
        "unique_large_senders",
        "whale_contract_calls",
        "Market_Regime",
    ]
    price_features = ["Close_Pct", "Volume_Pct", "High_Pct"]

    add_baselines(
        model_name="whale_features",
        file_path="data/processed/eth_merged_6h_clustered_2017_to_latest.csv",
        feature_columns=whale_features,
    )

    add_baselines(
        model_name="price_only",
        file_path="data/processed/eth_price_only_6h_2017_to_latest.csv",
        feature_columns=price_features,
    )

    print("\n✓ All baseline files added to:", EXISTING_RESULTS_DIR)


if __name__ == "__main__":
    main()
