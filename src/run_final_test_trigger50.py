import json
import random
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential


# Frozen final-test configuration (chosen before seeing test outcomes).
CRASH_THRESHOLD = 0.03
PREDICTION_TRIGGER = 0.50
WINDOW_SIZE = 24
EPOCHS = 20
BATCH_SIZE = 32
SEED = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.keras.utils.set_random_seed(seed)
    except Exception:
        pass


def build_model(input_shape: tuple[int, int]) -> Sequential:
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(128))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def load_and_prepare(file_path: str, feature_columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
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
        X.append(scaled_features[i - WINDOW_SIZE : i])
        y.append(int(data["Target_Crash"].iloc[i]))

    return np.array(X), np.array(y)


def run_final_test(model_name: str, file_path: str, feature_columns: list[str], output_dir: Path) -> dict:
    X, y = load_and_prepare(file_path, feature_columns)

    # Frozen chronological split used throughout project.
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, shuffle=False)
    X_validate, X_test, y_validate, y_test = train_test_split(X_temp, y_temp, test_size=0.50, shuffle=False)

    neg = np.sum(y_train == 0)
    pos = max(np.sum(y_train == 1), 1)
    class_weights = {
        0: (1 / neg) * (len(y_train) / 2.0),
        1: (1 / pos) * (len(y_train) / 2.0),
    }

    model = build_model((X_train.shape[1], X_train.shape[2]))
    history = model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_validate, y_validate),
        class_weight=class_weights,
        verbose=0,
    )

    # Evaluate ONLY now on held-out test.
    probs_test = model.predict(X_test, verbose=0).reshape(-1)
    y_pred_test = (probs_test > PREDICTION_TRIGGER).astype(int)

    report = classification_report(y_test, y_pred_test, zero_division=0)
    cm = confusion_matrix(y_test, y_pred_test)

    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    with open(model_dir / "classification_report_TEST.txt", "w", encoding="utf-8") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Trigger Threshold used: {PREDICTION_TRIGGER}\n")
        f.write(f"Crash Threshold used: {CRASH_THRESHOLD}\n")
        f.write(f"Seed: {SEED}\n\n")
        f.write(report)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Pred Safe (0)", "Pred Crash (1)"],
        yticklabels=["Actual Safe (0)", "Actual Crash (1)"],
    )
    plt.title(f"TEST Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.savefig(model_dir / "confusion_matrix_TEST.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title(f"Loss Curves - {model_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(model_dir / "loss_graph.png")
    plt.close()

    return {
        "model": model_name,
        "support_safe_test": int(np.sum(y_test == 0)),
        "support_crash_test": int(np.sum(y_test == 1)),
        "accuracy_test": float(accuracy_score(y_test, y_pred_test)),
        "crash_precision_test": float(precision_score(y_test, y_pred_test, zero_division=0)),
        "crash_recall_test": float(recall_score(y_test, y_pred_test, zero_division=0)),
        "crash_f1_test": float(f1_score(y_test, y_pred_test, zero_division=0)),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }


def main() -> None:
    set_seed(SEED)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("results") / f"FINAL_TEST_trigger50_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "protocol": "frozen-final-test",
        "seed": SEED,
        "crash_threshold": CRASH_THRESHOLD,
        "prediction_trigger": PREDICTION_TRIGGER,
        "window_size": WINDOW_SIZE,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "split": "chronological 70/15/15, shuffle=False",
        "note": "No hyperparameter tuning after test results",
    }

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

    results = []
    results.append(
        run_final_test(
            model_name="whale_features",
            file_path="data/processed/eth_merged_6h_clustered_2017_to_latest.csv",
            feature_columns=whale_features,
            output_dir=output_dir,
        )
    )
    results.append(
        run_final_test(
            model_name="price_only",
            file_path="data/processed/eth_price_only_6h_2017_to_latest.csv",
            feature_columns=price_features,
            output_dir=output_dir,
        )
    )

    summary_df = pd.DataFrame(results)
    summary_df.to_csv(output_dir / "summary_test_metrics.csv", index=False)

    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    with open(output_dir / "README.txt", "w", encoding="utf-8") as f:
        f.write("FINAL TEST RUN (frozen config)\n")
        f.write(f"Folder: {output_dir}\n\n")
        f.write("Subfolders:\n")
        f.write("- whale_features/: test report + confusion matrix + loss graph\n")
        f.write("- price_only/: test report + confusion matrix + loss graph\n\n")
        f.write("Top-level files:\n")
        f.write("- config.json: locked experiment settings\n")
        f.write("- summary_test_metrics.csv: side-by-side test metrics\n")

    print("Saved final test artifacts to:", output_dir)
    print("\nFinal test summary:\n", summary_df)


if __name__ == "__main__":
    main()
