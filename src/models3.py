import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc

# ==========================================
#  Fine-Tuning Hyperparameters
# ==========================================
CRASH_THRESHOLD = 0.03       
PREDICTION_TRIGGER = 0.50    # Shared threshold for fair whale vs price-only comparison.
RUN_TEST_EVAL = False

# Generates a unique name 
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
EXPERIMENT_NAME = f"LSTM_Run_{timestamp}"

OUTPUT_DIR = f"results/{EXPERIMENT_NAME}"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"--- Experiment tracking active. Saving to: {OUTPUT_DIR} ---")

# ==========================================
# 1. LOAD DATASET
# ==========================================

print("\n--- 1. Loading & Engineering Features ---")

file_path = 'data/processed/eth_merged_6h_clustered_2017_to_latest.csv'
data = pd.read_csv(file_path)
data['hour'] = pd.to_datetime(data['hour'])
data.set_index('hour', inplace=True)
data = data.sort_index()

# THE BIG FIX: Convert raw numbers into Momentum (% Change)
data['Close_Pct'] = data['Close'].pct_change()
data['Volume_Pct'] = data['Volume ETH'].pct_change()
data['Whale_Vol_Pct'] = data['massive_whale_volume'].pct_change()

# Drop the first row because pct_change leaves a blank (NaN) value
data = data.dropna()

# ==========================================
# 2. TARGET (Canonical 6-Hour Dataset Label)
# ==========================================
print(f"--- 2. Tagging Crashes (Drops >= {CRASH_THRESHOLD*100}% in EXACTLY the next 6 hours) ---")
data['Target_Crash'] = data['drawdown_6h_label']
print(f"Total Crashes found in entire dataset: {int(data['Target_Crash'].sum())}")

# ==========================================
# 3. FEATURE SELECTION & SCALING
# ==========================================
print("--- 3. Selecting and Scaling Features ---")
# We swap out the raw numbers for our new Momentum columns!
feature_columns = [
    'Close_Pct', 'Volume_Pct', 'Whale_Vol_Pct', 
    'max_gas_gwei', 'unique_large_senders', 'whale_contract_calls', 'Market_Regime'
]

features = data[feature_columns].values
scaler = MinMaxScaler(feature_range=(0,1))
scaled_features = scaler.fit_transform(features)

# ==========================================
# 4. CREATE WINDOWS (6 Days)
# ==========================================
print("--- 4. Creating 6-Day Windows ---")

window_size = 24 
X, y = [], []
target_dates = data.index[window_size:]

for i in range(window_size, len(scaled_features)):
    X.append(scaled_features[i - window_size:i]) 
    y.append(data['Target_Crash'].iloc[i])       

X = np.array(X)
y = np.array(y)
target_dates = pd.to_datetime(target_dates)

# ==========================================
# 5. CHRONOLOGICAL SPLIT (By Date)
# ==========================================
print("--- 5. Splitting Data Chronologically ---")

# First split: train vs temp (30%)
X_train, X_temp, y_train, y_temp, dates_train, dates_temp = train_test_split(
    X, y, target_dates, test_size=0.30, shuffle=False
)

# Second split: validation vs test (15% each)
X_validate, X_test, y_validate, y_test, dates_validate, dates_test = train_test_split(
    X_temp, y_temp, dates_temp, test_size=0.50, shuffle=False
)

# ==========================================
# 6. CALCULATE WEIGHTS FOR RARE CRASHES
# ==========================================
neg = np.sum(y_train == 0)
pos = np.sum(y_train == 1)
pos = max(pos, 1)

# Because crashes are rare again, we MUST penalize the AI for missing them
weight_for_0 = (1 / neg) * (len(y_train) / 2.0)
weight_for_1 = (1 / pos) * (len(y_train) / 2.0)
class_weights = {0: weight_for_0, 1: weight_for_1}

# ==========================================
# 7. BUILD AND TRAIN THE AI
# ==========================================
print("\n--- 7. Building & Training the LSTM ---")
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(
    X_train, y_train,
    epochs=20,          
    batch_size=32,      
    validation_data=(X_validate, y_validate),
    class_weight=class_weights,  # <--- WEIGHTS ARE BACK ON!
    verbose=1
)

# ==========================================
# 8. EVALUATE & SAVE LOSS GRAPH
# ==========================================
print("\n--- 8. Plotting and Saving Loss Graph ---")
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title(f'LSTM Model Loss ({EXPERIMENT_NAME})')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/loss_graph.png") # Save image
plt.close()

# ==========================================
# 9. THE LIE DETECTOR (CONFUSION MATRIX)
# ==========================================
print("\n--- 9. Evaluating on Validation Set ---")
raw_probabilities = model.predict(X_validate)

print(f"Highest confidence AI ever had for a crash: {max(raw_probabilities)[0]*100:.2f}%")
print(f"Average confidence AI had overall: {np.mean(raw_probabilities)*100:.2f}%")

# Force prediction to 1 if confidence is above our lowered trigger
y_pred = (raw_probabilities > PREDICTION_TRIGGER).astype(int)

# Print & Save the Report
report_text = classification_report(y_validate, y_pred, zero_division=0)
print(f"\nClassification Report (Trigger > {PREDICTION_TRIGGER*100}%):")
print(report_text)

with open(f"{OUTPUT_DIR}/classification_report.txt", "w") as f:
    f.write(f"Experiment: {EXPERIMENT_NAME}\n")
    f.write(f"Trigger Threshold used: {PREDICTION_TRIGGER}\n\n")
    f.write(report_text)

# Plot & Save the Confusion Matrix
cm = confusion_matrix(y_validate, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted Safe (0)', 'Predicted Crash (1)'],
            yticklabels=['Actually Safe (0)', 'Actually Crash (1)'])
plt.title(f'Confusion Matrix: Did it find crashes?\n(Trigger: {PREDICTION_TRIGGER})')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png") # Save image
plt.close()

print(f"\n✅ All done! Check the '{OUTPUT_DIR}' folder for your saved graphs and text report.")

# ==========================================
# 10. FINAL EVALUATION ON HELD-OUT TEST SET
# ==========================================
if RUN_TEST_EVAL:
    print("\n--- 10. Final Evaluation on Test Set ---")
    raw_probs_test = model.predict(X_test)
    y_pred_test = (raw_probs_test > PREDICTION_TRIGGER).astype(int)

    report_test = classification_report(y_test, y_pred_test, zero_division=0)
    print(report_test)
    with open(f"{OUTPUT_DIR}/classification_report_TEST.txt", "w") as f:
        f.write(f"Experiment: {EXPERIMENT_NAME}\n")
        f.write(f"Trigger Threshold: {PREDICTION_TRIGGER}\n\n")
        f.write(report_test)

    cm_test = confusion_matrix(y_test, y_pred_test)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted Safe', 'Predicted Crash'],
                yticklabels=['Actually Safe', 'Actually Crash'])
    plt.title('Confusion Matrix — TEST SET')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/confusion_matrix_TEST.png")
    plt.close()

    # ==========================================
    # 11. BASELINE COMPARISONS ON TEST SET
    # ==========================================
    print("\n--- 11. Baseline Comparisons ---")

    # Majority class baseline
    y_pred_majority = np.zeros(len(y_test), dtype=int)
    report_majority = classification_report(y_test, y_pred_majority, zero_division=0)
    print("Majority Class Baseline:\n", report_majority)
    with open(f"{OUTPUT_DIR}/baseline_majority.txt", "w") as f:
        f.write(report_majority)

    # Logistic regression baseline
    X_train_flat = X_train.reshape(len(X_train), -1)
    X_test_flat  = X_test.reshape(len(X_test), -1)
    lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    lr.fit(X_train_flat, y_train)
    y_pred_lr = lr.predict(X_test_flat)
    report_lr = classification_report(y_test, y_pred_lr, zero_division=0)
    print("Logistic Regression Baseline:\n", report_lr)
    with open(f"{OUTPUT_DIR}/baseline_logistic_regression.txt", "w") as f:
        f.write(report_lr)

    # ==========================================
    # 12. ROC CURVE — TEST SET
    # ==========================================
    print("\n--- 12. ROC Curve ---")
    fpr_lstm, tpr_lstm, _ = roc_curve(y_test, raw_probs_test)
    auc_lstm = auc(fpr_lstm, tpr_lstm)

    lr_probs = lr.predict_proba(X_test_flat)[:, 1]
    fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_probs)
    auc_lr = auc(fpr_lr, tpr_lr)

    print(f"LSTM AUC: {auc_lstm:.4f}")
    print(f"Logistic Regression AUC: {auc_lr:.4f}")

    plt.figure(figsize=(7, 5))
    plt.plot(fpr_lstm, tpr_lstm, color='blue',
             label=f'LSTM + Whale Features (AUC = {auc_lstm:.3f})')
    plt.plot(fpr_lr, tpr_lr, color='orange', linestyle='--',
             label=f'Logistic Regression (AUC = {auc_lr:.3f})')
    plt.plot([0, 1], [0, 1], linestyle=':', color='grey', label='Random Baseline')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (Recall)')
    plt.title('ROC Curve — Test Set')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/roc_curve.png")
    plt.close()
    print(f"All results saved to {OUTPUT_DIR}")
else:
    print("\n--- Test evaluation skipped (RUN_TEST_EVAL=False). ---")