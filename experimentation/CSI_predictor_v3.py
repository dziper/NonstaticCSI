import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Just for this case, I load and preprocess Sionna data directly to compare with later
with open(r"C:\Users\nrazavi\Downloads\batch1_1Lane_450U_1Rx_32Tx_1T_80K_2620000000.0fc.pickle", 'rb') as f:
    data = pickle.load(f)
# Just isolating the CSI info
freq_channel = np.squeeze(data["freq_channel"])
CSI = np.transpose(freq_channel, axes=(1, 2, 0))
test_CSI = CSI[:, :, -1]  # Getting the 450th frame to act as test of model
CSI = CSI[:, :, :-1]      # Removing test from the training dataset
rows, cols, num_samples = CSI.shape

# Reshape CSI data for real and imaginary parts
reshaped_CSI = CSI.reshape((rows * cols, num_samples)).T
X_r = np.real(reshaped_CSI)
X_c = np.imag(reshaped_CSI)

# Normalize the data
scaler_r = StandardScaler()
scaler_c = StandardScaler()
X_r = scaler_r.fit_transform(X_r)
X_c = scaler_c.fit_transform(X_c)

# Reshape for sequence processing
X_r = X_r.reshape((1, num_samples, rows * cols))
X_c = X_c.reshape((1, num_samples, rows * cols))

# Create sequences
def create_sequences(data, sequence_length):
    sequences = []
    targets = []
    for i in range(data.shape[1] - sequence_length):
        sequences.append(data[0, i:i + sequence_length, :])  # Remove the batch dimension here
        targets.append(data[0, i + sequence_length, :])
    return np.array(sequences), np.array(targets)


sequence_length = 1  # Define sequence length
X_r_seq, y_r = create_sequences(X_r, sequence_length)
X_c_seq, y_c = create_sequences(X_c, sequence_length)

print(f"Real Part Sequence Shape: {X_r_seq.shape}, Target Shape: {y_r.shape}")
print(f"Imaginary Part Sequence Shape: {X_c_seq.shape}, Target Shape: {y_c.shape}")
# Define and compile the real part model
model_r = Sequential([
    LSTM(128, input_shape=(sequence_length, rows * cols), return_sequences=True),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(rows * cols)
])
model_r.compile(optimizer='adam', loss='mse')

# Define and compile the imaginary part model
model_c = Sequential([
    LSTM(128, input_shape=(sequence_length, rows * cols), return_sequences=True),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(rows * cols)
])
model_c.compile(optimizer='adam', loss='mse')

# Train the models
batch_size = 16
epochs = 50

print("Training Real Part Model...")
model_r.fit(X_r_seq, y_r, epochs=epochs, batch_size=batch_size)

print("Training Imaginary Part Model...")
model_c.fit(X_c_seq, y_c, epochs=epochs, batch_size=batch_size)

# Predict the next vector
print("Predicting Next Real Part...")
next_vector_r = model_r.predict(X_r_seq)
print("Predicting Next Imaginary Part...")
next_vector_c = model_c.predict(X_c_seq)

# Reconstruct the complex CSI prediction
predicted_CSI = scaler_r.inverse_transform(next_vector_r) + 1j * scaler_c.inverse_transform(next_vector_c)

# Reshape the predicted CSI to match the original dimensions
predicted_CSI = predicted_CSI.reshape(rows, cols, -1)

# Evaluate the model
mse = np.mean(np.abs(predicted_CSI[:, :, -1] - test_CSI) ** 2)
print(f"Mean Squared Error (MSE): {mse}")
print(predicted_CSI[:, :, -1].shape)
# Plot the predictions (optional)
plt.figure(figsize=(10, 6))
plt.plot(np.real(predicted_CSI[:, :, -1].flatten()), label="Predicted Magnitude")
plt.plot(np.real(test_CSI.flatten()), label="True Magnitude", linestyle="dashed")
plt.title("Predicted vs. True CSI")
plt.legend()
plt.show()
