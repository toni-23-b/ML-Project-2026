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

# ==========================================
#  Fine-Tuning Hyperparameters
# ==========================================
EXPERIMENT_NAME = "LSTM_CanonicalLabel_3Percent"
CRASH_THRESHOLD = 0.03       # Kept for logging only; target comes from dataset label.
PREDICTION_TRIGGER = 0.15    # If AI is >15% sure, sound the alarm! (Lowered from 50%)

# Setup automatic saving folder
OUTPUT_DIR = f"results/{EXPERIMENT_NAME}"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"--- Experiment tracking active. Saving results to: {OUTPUT_DIR} ---")

# ==========================================
# 1. LOAD DATASET
# ==========================================
print("\n--- 1. Loading Clustered Dataset ---")

file_path = 'data/processed/eth_merged_6h_2021_to_latest.csv'
data = pd.read_csv(file_path)
data['hour'] = pd.to_datetime(data['hour'])
data.set_index('hour', inplace=True)
data = data.sort_index()

# ==========================================
# 2. TARGET
# ==========================================
print(f"--- 2. Tagging Crashes (Drops >= {CRASH_THRESHOLD*100}% in EXACTLY the next 6 hours) ---")

if 'drawdown_6h_label' not in data.columns:
    raise ValueError("Missing drawdown_6h_label in dataset; rebuild dataset before training.")

data['Target_Crash'] = data['drawdown_6h_label'].astype(int)
print(f"Total Crashes found in entire dataset: {int(data['Target_Crash'].sum())}")

# ==========================================
# 3. FEATURE SELECTION & SCALING
# ==========================================
print("--- 3. Selecting and Scaling Features ---")

feature_columns = [
    'Close', 'Volume ETH', 'massive_whale_volume', 'max_gas_gwei',
    'unique_large_senders', 'whale_contract_calls', 'total_network_volume'
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
# 6. CALCULATE WEIGHTS & BIAS FOR RARE CRASHES
# ==========================================
neg = np.sum(y_train == 0)
pos = np.sum(y_train == 1)
pos = max(pos, 1) # Prevent dividing by zero just in case

# Penalties for the AI
weight_for_0 = (1 / neg) * (len(y_train) / 2.0)
weight_for_1 = (1 / pos) * (len(y_train) / 2.0)
class_weights = {0: weight_for_0, 1: weight_for_1}

# Tell AI upfront that crashes are rare
initial_bias = np.log([pos/neg])
output_bias = tf.keras.initializers.Constant(initial_bias)

# ==========================================
# 7. BUILD AND TRAIN THE AI
# ==========================================
print("\n--- 7. Building & Training the LSTM ---")
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid', bias_initializer=output_bias))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(
    X_train, y_train,
    epochs=20,          
    batch_size=32,      
    validation_data=(X_validate, y_validate),
    class_weight=class_weights, 
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
plt.show()

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
plt.show()

print(f"\n✅ All done! Check the '{OUTPUT_DIR}' folder for your saved graphs and text report.")