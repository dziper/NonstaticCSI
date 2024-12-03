import pickle
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.stattools import adfuller
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
# Load and preprocess data
with open(r"C:\Users\nrazavi\Downloads\batch1_1Lane_450U_1Rx_32Tx_1T_80K_2620000000.0fc.pickle", 'rb') as f:
    data = pickle.load(f)

CSI = np.squeeze(data["freq_channel"])
test_CSI = CSI[:,:,-1]
print(np.array(test_CSI).shape)
CSI = CSI[:,:,:-1]
receivers, subcarriers, frames = CSI.shape
train_data = CSI[:, :, :-1]  # Use the first 449 frames for training
target_data = CSI[:, :, 1:]  # Predict the next frame

# Reshape for LSTM input: (samples, timesteps, features)
X_train = train_data.reshape(-1, frames-1, 1)  # Each subcarrier-receiver pair is a sample
y_train = target_data.reshape(-1, frames-1, 1)

# Build the model
model = Sequential([
    LSTM(64, input_shape=(frames-1, 1), return_sequences=True),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=5, batch_size=32)

# Predict the next frame
print(np.array(CSI[:,:,-1]).shape)
predictions = model.predict(CSI[:,:,-1])
