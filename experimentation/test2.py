import pickle
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.stattools import adfuller

# Load and preprocess data
with open(r"C:\Users\nrazavi\Downloads\batch1_1Lane_450U_1Rx_32Tx_1T_80K_2620000000.0fc.pickle", 'rb') as f:
    data = pickle.load(f)

freq_channel = np.squeeze(data["freq_channel"])

ant1_subcarrier1_csi = [np.squeeze(freq_channel[x, :, :])[0][0] for x in range(75)]

# Separate magnitude and phase
# magnitudes = [np.abs(num) for num in ant1_subcarrier1_csi]
# phases = [np.angle(num) for num in ant1_subcarrier1_csi]

magnitudes = [np.real(num) for num in ant1_subcarrier1_csi] # uncomment to compare to real and imag parts
phases = [np.imag(num) for num in ant1_subcarrier1_csi]

# Function to check stationarity
def check_stationarity(timeseries):
    result = adfuller(timeseries)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:', result[4])
    if result[1] <= 0.05:
        print("Stationary")
    else:
        print("Non-stationary")

# Function to find optimal lag order
def find_optimal_lag(data, max_lag=30):
    best_aic = np.inf
    best_lag = 0
    for lag in range(1, max_lag + 1):
        model = AutoReg(data, lags=lag)
        results = model.fit()
        if results.aic < best_aic:
            best_aic = results.aic
            best_lag = lag
    return best_lag

# Function to train and evaluate model
def train_evaluate_model(train_data, test_data, lag_order, component_name):
    model = AutoReg(train_data, lags=lag_order)
    results = model.fit()
    
    predictions = results.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1, dynamic=False)
    
    mae = mean_absolute_error(test_data, predictions)
    rmse = np.sqrt(mean_squared_error(test_data, predictions))
    
    print(f'Mean Absolute Error for {component_name}: {mae:.6f}')
    print(f'Root Mean Squared Error for {component_name}: {rmse:.6f}')
    
    plt.figure(figsize=(12, 6))
    plt.plot(test_data, label=f'Actual {component_name}')
    plt.plot(predictions, label=f'Predicted {component_name}', linestyle='--')
    plt.xlabel('Time')
    plt.ylabel(component_name)
    plt.legend()
    plt.title(f'{component_name} Prediction with Autoregressive Model')
    plt.show()
    
    return predictions

# Split data
train_size = int(0.99 * len(magnitudes))
train_mag, test_mag = magnitudes[:train_size], magnitudes[train_size:]
train_phase, test_phase = phases[:train_size], phases[train_size:]

# Check stationarity
print("Checking stationarity for magnitude:")
check_stationarity(magnitudes)
print("\nChecking stationarity for phase:")
check_stationarity(phases)

plot_acf(magnitudes)
plt.title('ACF for Magnitude')
plt.show()

plot_acf(phases)
plt.title('ACF for Phase')
plt.show()

# Find optimal lag order
lag_order_mag = find_optimal_lag(train_mag)
lag_order_phase = find_optimal_lag(train_phase)

print(f"Optimal lag order for magnitude: {lag_order_mag}")
print(f"Optimal lag order for phase: {lag_order_phase}")

# Train and evaluate models
pred_mag = train_evaluate_model(train_mag, test_mag, lag_order_mag, "Magnitude")
pred_phase = train_evaluate_model(train_phase, test_phase, lag_order_phase, "Phase")

# Combine predictions
# predicted_csi = pred_mag * np.exp(1j * pred_phase)
predicted_csi = pred_mag  + 1j*pred_phase #if using mag=real part and phase = imaginary

# print("Predicted CSI (first 5 values):", predicted_csi[:5])
# print("Actual CSI (first 5 values):", [abs(x) * np.exp(1j * np.angle(x)) for x in ant1_subcarrier1_csi[train_size:]][:5])

print("Predicted CSI (first 5 values):", predicted_csi[:5])
print("Actual CSI (first 5 values):", [np.real(x) + 1j*np.imag(x) for x in ant1_subcarrier1_csi[train_size:]][:5])

