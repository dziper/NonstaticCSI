import numpy as np
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load and preprocess data
with open(r"C:\Users\nrazavi\Downloads\batch1_1Lane_450U_1Rx_32Tx_1T_80K_2620000000.0fc.pickle", 'rb') as f:
    data = pickle.load(f)

freq_channel = np.squeeze(data["freq_channel"])
print(freq_channel.shape)
CSI =  np.transpose(freq_channel, axes=(1, 2, 0))
print(CSI.shape)
test_CSI = CSI[:,:,-1]
CSI = CSI[:,:,:-1]
print(CSI.shape)
rows,cols,num_samples = CSI.shape

reshaped_CSI = CSI.reshape((rows*cols,num_samples))
reshaped_CSI = np.transpose(reshaped_CSI)

# Prepare sequences for LSTM
X_r = np.real(reshaped_CSI)
X_c = np.imag(reshaped_CSI)
print(X_c.shape)
X_c = X_c.reshape((1, num_samples, rows*cols))
X_r = X_r.reshape((1, num_samples, rows*cols))

# Create the model
model_c = Sequential([
    LSTM(128, input_shape=(num_samples, rows*cols), return_sequences=True),
    LSTM(64),
    Dense(2560)
])

model_c.compile(optimizer='adam', loss='mse')

# Train the model
model_c.fit(X_c, X_c[:, -1, :], epochs=100, batch_size=1)

# Predict the next vector
next_vector_c = model_c.predict(X_c)
print(next_vector_c.shape)

model_r = Sequential([
    LSTM(128, input_shape=(449, 2560), return_sequences=True),
    LSTM(64),
    Dense(2560)
])
model_r.compile(optimizer='adam',loss='mse')
model_r.fit(X_r,X_r[:,-1,:], epochs=100, batch_size=1)
next_vector_r = model_r.predict(X_r)
print(next_vector_c.shape)

predicted_CSI = next_vector_r + 1j*next_vector_c
print(predicted_CSI.shape)
print(predicted_CSI)
print("##########")
print(test_CSI)

