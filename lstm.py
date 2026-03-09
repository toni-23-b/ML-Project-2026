import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

data = pd.read_csv('dataset.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
prices = data['Price'].values

drawdown_labels = []

threshold = 0.03  # 3% drop

# label[t] = 1 if price drops ≥3% in next 6h
for i in range(len(prices)):
    if i == len(prices) - 1:
        drawdown_labels.append(0)
    else:
        drop = (prices[i] - prices[i+1]) / prices[i]
        drawdown_labels.append(1 if drop >= threshold else 0)

drawdown_labels = np.array(drawdown_labels)

# !!!!add needed features!!!!
features = data[['Price']].values

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(features)

# 6h intervals × 24 timesteps = 6 days of context
window_size = 24
X = []
y = []
target_dates = data.index[window_size:]

for i in range(window_size, len(scaled_data)):
    X.append(scaled_data[i - window_size:i])
    y.append(drawdown_labels[i])

X = np.array(X)
y = np.array(y)

# First split: train vs temp (30%)
X_train, X_temp, y_train, y_temp, dates_train, dates_temp = train_test_split(
    X, y, target_dates, test_size=0.30, shuffle=False
)

# Second split: validation vs test (15% each)
X_validate, X_test, y_validate, y_test, dates_validate, dates_test = train_test_split(
    X_temp, y_temp, dates_temp, test_size=0.50, shuffle=False
)

# Reshape for LSTM
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_validate = X_validate.reshape((X_validate.shape[0], X_validate.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

#For an LSTM use sigmoid + binary cross entropy
model = Sequential()

model.add(LSTM(128, return_sequences=True,
               input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))

model.add(LSTM(128))
model.add(Dropout(0.2))

model.add(Dense(1, activation='sigmoid'))

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

#---------------------------------------- TRAIN ------------------------------------------------------
history = model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=32,
    validation_data=(X_validate, y_validate)
)

#---------------------------------------- EVALUATE ------------------------------------------------------

test_loss, test_accuracy = model.evaluate(X_test, y_test)

print("Test accuracy:", test_accuracy)

#---------------------------------------- PLOTS ------------------------------------------------------

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['train','validation'])
plt.title('Model loss')
plt.show()
