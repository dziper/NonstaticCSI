import pickle
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR

# Load and preprocess data
with open(r"C:\Users\nrazavi\Downloads\batch1_1Lane_450U_1Rx_32Tx_1T_80K_2620000000.0fc.pickle", 'rb') as f:
    data = pickle.load(f)

freq_channel = np.squeeze(data["freq_channel"])

# Combine all subcarriers into a single input vector for time-series analysis
ant1_all_subcarriers_csi = [np.squeeze(freq_channel[x, :, :]) for x in range(450)]
print(np.array(ant1_all_subcarriers_csi).shape)

def var_analysis(data):
    # Transpose the data for VAR: time steps as rows, variables as columns
    data = data.T  # Now shape is (450, 2560)
    
    # Step 1: Check stationarity (optional, not shown)
    print("\nStep 1: Checking stationarity (not implemented)")

    # Step 2: Applying VAR model
    print("\nStep 2: Applying VAR model")
    model = VAR(data)
    results = model.fit(maxlags=15)  # Fit with up to 15 lags (adjustable)

    # Step 3: Forecasting
    print("\nStep 3: Forecasting")
    lag_order = results.k_ar
    print("Lag order = ", lag_order)
    last_obs = data[-lag_order:]  # Get the last 'lag_order' rows
    forecast = results.forecast(last_obs, steps=1)
    return forecast

# Convert to real part and perform VAR
samples,rows,cols = np.array(ant1_all_subcarriers_csi).shape
print(samples, rows, cols)
samples_considered = samples - 1

ant1_all_subcarriers_csi = np.array(ant1_all_subcarriers_csi)
vectorized_csi = np.empty((rows * cols, samples_considered), dtype=complex)

# Fill vectorized_csi
for i in range(samples_considered):
    reshaped_csi = ant1_all_subcarriers_csi[i, :, :].reshape(rows * cols)
    vectorized_csi[:, i] = reshaped_csi
test_csi = ant1_all_subcarriers_csi[samples_considered, :, :].reshape(rows * cols)
# Perform VAR analysis
predicted = var_analysis(np.real(vectorized_csi))
print(np.array(predicted).shape)
print("predicted = ", predicted[[0]])
print("actual =", np.real(test_csi[0]))

