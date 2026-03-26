import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential


CRASH_THRESHOLD = 0.03
PREDICTION_TRIGGER = 0.50
WINDOW_SIZE = 24
EPOCHS = 20
BATCH_SIZE = 32
SEEDS = [11, 22, 33]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.keras.utils.set_random_seed(seed)
    except Exception:
        pass


def build_sequences(data: pd.DataFrame, feature_columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
    features = data[feature_columns].values
    scaled_features = MinMaxScaler(feature_range=(0, 1)).fit_transform(features)

    X, y = [], []
    for i in range(WINDOW_SIZE, len(scaled_features)):
        X.append(scaled_features[i - WINDOW_SIZE : i])
        y.append(int(data["Target_Crash"].iloc[i]))

    return np.array(X), np.array(y)


def build_model(input_shape: tuple[int, int]) -> Sequential:
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(128))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def run_single_experiment(model_name: str, file_path: str, feature_columns: list[str], seed: int) -> dict:
    set_seed(seed)

    data = pd.read_csv(file_path)
    data["hour"] = pd.to_datetime(data["hour"])
    data = data.sort_values("hour").set_index("hour")

    # Keep feature engineering consistent with model scripts.
    data["Close_Pct"] = data["Close"].pct_change()
    data["Volume_Pct"] = data["Volume ETH"].pct_change()
    if "massive_whale_volume" in data.columns:
        data["Whale_Vol_Pct"] = data["massive_whale_volume"].pct_change()
    if "High" in data.columns:
        data["High_Pct"] = data["High"].pct_change()

    data = data.dropna().copy()
    data["Target_Crash"] = data["drawdown_6h_label"].astype(int)

    X, y = build_sequences(data, feature_columns)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, shuffle=False)
    X_validate, _, y_validate, _ = train_test_split(X_temp, y_temp, test_size=0.50, shuffle=False)

    neg = np.sum(y_train == 0)
    pos = max(np.sum(y_train == 1), 1)
    class_weights = {
        0: (1 / neg) * (len(y_train) / 2.0),
        1: (1 / pos) * (len(y_train) / 2.0),
    }

    model = build_model((X_train.shape[1], X_train.shape[2]))
    model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_validate, y_validate),
        class_weight=class_weights,
        verbose=0,
    )

    probs = model.predict(X_validate, verbose=0).reshape(-1)
    y_pred = (probs > PREDICTION_TRIGGER).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_validate, y_pred, labels=[0, 1]).ravel()

    return {
        "model": model_name,
        "seed": seed,
        "trigger": PREDICTION_TRIGGER,
        "support_safe": int(np.sum(y_validate == 0)),
        "support_crash": int(np.sum(y_validate == 1)),
        "accuracy": float(accuracy_score(y_validate, y_pred)),
        "crash_precision": float(precision_score(y_validate, y_pred, zero_division=0)),
        "crash_recall": float(recall_score(y_validate, y_pred, zero_division=0)),
        "crash_f1": float(f1_score(y_validate, y_pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("model", as_index=False).agg(
        runs=("seed", "count"),
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        crash_precision_mean=("crash_precision", "mean"),
        crash_precision_std=("crash_precision", "std"),
        crash_recall_mean=("crash_recall", "mean"),
        crash_recall_std=("crash_recall", "std"),
        crash_f1_mean=("crash_f1", "mean"),
        crash_f1_std=("crash_f1", "std"),
        fp_mean=("fp", "mean"),
        tp_mean=("tp", "mean"),
    )
    return grouped


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("results") / f"ROBUST_COMPARISON_trigger50_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "crash_threshold": CRASH_THRESHOLD,
        "prediction_trigger": PREDICTION_TRIGGER,
        "window_size": WINDOW_SIZE,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "seeds": SEEDS,
        "split": "chronological 70/15/15, shuffle=False",
        "target": "drawdown_6h_label",
    }

    runs: list[dict] = []

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

    for seed in SEEDS:
        runs.append(
            run_single_experiment(
                model_name="whale_features",
                file_path="data/processed/eth_merged_6h_clustered_2017_to_latest.csv",
                feature_columns=whale_features,
                seed=seed,
            )
        )
        runs.append(
            run_single_experiment(
                model_name="price_only",
                file_path="data/processed/eth_price_only_6h_2017_to_latest.csv",
                feature_columns=price_features,
                seed=seed,
            )
        )

    runs_df = pd.DataFrame(runs)
    summary_df = summarize(runs_df)

    runs_df.to_csv(output_dir / "per_run_metrics.csv", index=False)
    summary_df.to_csv(output_dir / "summary_metrics.csv", index=False)

    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    with open(output_dir / "README.txt", "w", encoding="utf-8") as f:
        f.write("Robust comparison run (fixed seeds)\n")
        f.write(f"Output folder: {output_dir}\n\n")
        f.write("Files:\n")
        f.write("- config.json: exact run settings\n")
        f.write("- per_run_metrics.csv: one row per model per seed\n")
        f.write("- summary_metrics.csv: mean/std aggregated by model\n")

    print("Saved robust comparison artifacts to:", output_dir)
    print("\nPer-run metrics:\n", runs_df)
    print("\nSummary metrics:\n", summary_df)


if __name__ == "__main__":
    main()
