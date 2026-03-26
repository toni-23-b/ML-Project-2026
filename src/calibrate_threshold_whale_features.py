import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential


# Fast calibration run for selecting decision threshold.
SEED = 42
EPOCHS = 10
BATCH_SIZE = 32
WINDOW_SIZE = 24

np.random.seed(SEED)
tf.random.set_seed(SEED)


def main() -> None:
    file_path = "data/processed/eth_merged_6h_clustered_2017_to_latest.csv"
    data = pd.read_csv(file_path)
    data["hour"] = pd.to_datetime(data["hour"])
    data = data.sort_values("hour").set_index("hour")

    data["Close_Pct"] = data["Close"].pct_change()
    data["Volume_Pct"] = data["Volume ETH"].pct_change()
    data["Whale_Vol_Pct"] = data["massive_whale_volume"].pct_change()
    data = data.dropna().copy()
    data["Target_Crash"] = data["drawdown_6h_label"].astype(int)

    feature_columns = [
        "Close_Pct",
        "Volume_Pct",
        "Whale_Vol_Pct",
        "max_gas_gwei",
        "unique_large_senders",
        "whale_contract_calls",
        "Market_Regime",
    ]

    features = data[feature_columns].values
    scaled_features = MinMaxScaler(feature_range=(0, 1)).fit_transform(features)

    X, y = [], []
    for i in range(WINDOW_SIZE, len(scaled_features)):
        X.append(scaled_features[i - WINDOW_SIZE : i])
        y.append(int(data["Target_Crash"].iloc[i]))

    X = np.array(X)
    y = np.array(y)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, shuffle=False
    )
    X_validate, _, y_validate, _ = train_test_split(
        X_temp, y_temp, test_size=0.50, shuffle=False
    )

    neg = np.sum(y_train == 0)
    pos = max(np.sum(y_train == 1), 1)
    class_weights = {
        0: (1 / neg) * (len(y_train) / 2.0),
        1: (1 / pos) * (len(y_train) / 2.0),
    }

    model = Sequential()
    model.add(
        LSTM(
            128,
            return_sequences=True,
            input_shape=(X_train.shape[1], X_train.shape[2]),
        )
    )
    model.add(Dropout(0.2))
    model.add(LSTM(128))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

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

    thresholds = np.arange(0.30, 0.96, 0.05)
    rows = []
    for threshold in thresholds:
        y_pred = (probs > threshold).astype(int)
        precision = precision_score(y_validate, y_pred, zero_division=0)
        recall = recall_score(y_validate, y_pred, zero_division=0)
        f1 = f1_score(y_validate, y_pred, zero_division=0)
        acc = accuracy_score(y_validate, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_validate, y_pred, labels=[0, 1]).ravel()
        rows.append(
            {
                "thr": round(float(threshold), 2),
                "acc": float(acc),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "fp": int(fp),
                "tp": int(tp),
                "fn": int(fn),
                "tn": int(tn),
            }
        )

    baseline = min(rows, key=lambda row: abs(row["thr"] - 0.50))
    best_f1 = max(rows, key=lambda row: row["f1"])
    recall_floor = [row for row in rows if row["recall"] >= 0.30]
    if recall_floor:
        best_low_noise = max(recall_floor, key=lambda row: (row["precision"], row["f1"]))
    else:
        best_low_noise = max(rows, key=lambda row: row["precision"])

    print("support_safe_crash", int(np.sum(y_validate == 0)), int(np.sum(y_validate == 1)))
    print("baseline_0.50", baseline)
    print("best_f1", best_f1)
    print("best_low_noise_recall_ge_0.30", best_low_noise)
    print("table_start")
    for row in rows:
        print(row)
    print("table_end")


if __name__ == "__main__":
    main()
