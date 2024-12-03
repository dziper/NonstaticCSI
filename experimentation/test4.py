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
ant1_subcarrier1_csi = [subcarrier[1][1] for subcarrier in ant1_all_subcarriers_csi]  
print(np.array(ant1_subcarrier1_csi).shape)
ant1_subcarrier1_csi = ant1_subcarrier1_csi[1:75]  # Adjusted the data range to minimize the effect of interference for better prediction accuracy.

real_part = [np.real(num) for num in ant1_subcarrier1_csi]
imag_part = [np.imag(num) for num in ant1_subcarrier1_csi]

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
def find_optimal_lag(data, max_lag=2):
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
    print(prediction)
    print(test_data)
    # mae = abs(test_data[0] - prediction)
    # rmse = np.sqrt((test_data[0] - prediction) ** 2)
    
    # print(f'Mean Absolute Error for {component_name}: {mae}')
    # print(f'Root Mean Squared Error for {component_name}: {rmse}')
    
    # plt.figure(figsize=(12, 6))
    # plt.plot(train_data, label=f'Training {component_name}')
    # plt.plot([len(train_data), len(train_data) + 1], [train_data[-1], prediction], label=f'Predicted {component_name}', marker='o', linestyle='--')
    # plt.plot([len(train_data) + 1], [test_data[0]], label=f'Actual {component_name}', marker='o')
    # plt.xlabel('Time')
    # plt.ylabel(component_name)
    # plt.legend()
    # plt.title(f'{component_name} Prediction for Next Time Step')
    # plt.show()
    
    return prediction

# Split data
train_size = len(real_part)-1
train_mag, test_mag = real_part[:train_size], real_part[train_size:]
train_phase, test_phase = imag_part[:train_size], imag_part[train_size:]

# Check stationarity
print("Checking stationarity for real_part:")
check_stationarity(real_part)
print("\nChecking stationarity for imag_part:")
check_stationarity(imag_part)

# Plot ACF
# plot_acf(real_part)
# plt.title('ACF for real_part')
# plt.show()

# plot_acf(imag_part)
# plt.title('ACF for imag_part')
# plt.show()

# Find optimal lag order
lag_order_mag = find_optimal_lag(train_mag)
lag_order_phase = find_optimal_lag(train_phase)

print(f"Optimal lag order for real_part: {lag_order_mag}")
print(f"Optimal lag order for imag_part: {lag_order_phase}")

# Train and evaluate models
pred_mag = train_evaluate_model(train_mag, test_mag, lag_order_mag, "real_part")
pred_phase = train_evaluate_model(train_phase, test_phase, lag_order_phase, "imag_part")

# # Combine predictions
# predicted_csi = pred_mag + 1j * pred_phase

# print("Predicted CSI for next time step:", predicted_csi)
# print("Actual CSI for next time step:", )