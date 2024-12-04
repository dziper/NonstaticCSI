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

# Combine all subcarriers into a single input vector for time-series analysis
ant1_all_subcarriers_csi = [np.squeeze(freq_channel[x, :, :]) for x in range(450)]
print(np.array(ant1_all_subcarriers_csi).shape)
ant1_subcarrier1_csi = [subcarrier[0][0] for subcarrier in ant1_all_subcarriers_csi]  
print(np.array(ant1_subcarrier1_csi).shape)
ant1_subcarrier1_csi = ant1_subcarrier1_csi[1:400]  # Adjusted the data range to minimize the effect of interference for better prediction accuracy.

magnitudes = [np.real(num) for num in ant1_subcarrier1_csi]
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
def find_optimal_lag(data, max_lag):
    best_aic = np.inf
    best_lag = 0
    for lag in range(1, max_lag + 1):
        model = AutoReg(data, lags=lag)
        results = model.fit()
        if results.aic < best_aic:
            best_aic = results.aic
            best_lag = lag
    return best_lag

# Function to train and evaluate model for next time step prediction
def train_evaluate_model(train_data, test_data, lag_order, component_name):
    model = AutoReg(train_data, lags=lag_order)
    results = model.fit()
    
    # Predict the next time step
    prediction = results.predict(start=len(train_data), end=len(train_data), dynamic=False)[0]
    
    mae = abs(test_data[0] - prediction)
    rmse = np.sqrt((test_data[0] - prediction) ** 2)
    
    print(f'Mean Absolute Error for {component_name}: {mae:.6f}')
    print(f'Root Mean Squared Error for {component_name}: {rmse:.6f}')
    
    plt.figure(figsize=(12, 6))
    plt.plot(train_data, label=f'Training {component_name}')
    plt.plot([len(train_data), len(train_data) + 1], [train_data[-1], prediction], label=f'Predicted {component_name}', marker='o', linestyle='--')
    plt.plot([len(train_data) + 1], [test_data[0]], label=f'Actual {component_name}', marker='o')
    plt.xlabel('Time')
    plt.ylabel(component_name)
    plt.legend()
    plt.title(f'{component_name} Prediction for Next Time Step')
    plt.show()
    
    return prediction

# Split data
train_size = int(0.99 * len(magnitudes))
train_mag, test_mag = magnitudes[:train_size], magnitudes[train_size:]
train_phase, test_phase = phases[:train_size], phases[train_size:]
print(np.array(magnitudes).shape)
# Check stationarity
# print("Checking stationarity for magnitude:")
# check_stationarity(magnitudes)
# print("\nChecking stationarity for phase:")
# check_stationarity(phases)

# Plot ACF
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
predicted_csi = pred_mag * np.exp(1j * pred_phase)

print("Predicted CSI for next time step:", predicted_csi)
print("Actual CSI for next time step:", abs(ant1_subcarrier1_csi[train_size]) * np.exp(1j * np.angle(ant1_subcarrier1_csi[train_size])))