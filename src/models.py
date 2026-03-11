"""Model helpers for ETH drawdown classification."""

from __future__ import annotations

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


def build_lstm_model(input_shape: tuple[int, int]) -> keras.Model:
    """Build and compile a simple LSTM binary classifier.

    Args:
        input_shape: Shape of one sequence sample (timesteps, features).
    """
    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.LSTM(64, return_sequences=False),
            layers.Dropout(0.2),
            layers.Dense(32, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def make_lstm_sequences(
    features: np.ndarray,
    labels: np.ndarray,
    lookback: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert tabular rows into rolling sequences for LSTM training."""
    if len(features) != len(labels):
        raise ValueError("features and labels must have the same number of rows")
    if lookback < 1:
        raise ValueError("lookback must be >= 1")

    x_list, y_list = [], []
    for end_idx in range(lookback - 1, len(features)):
        start_idx = end_idx - lookback + 1
        x_list.append(features[start_idx : end_idx + 1])
        y_list.append(labels[end_idx])

    return np.asarray(x_list), np.asarray(y_list)
