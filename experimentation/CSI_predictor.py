import pickle
import numpy as np

with open(r"C:\Users\nrazavi\Downloads\batch1_1Lane_450U_1Rx_32Tx_1T_80K_2620000000.0fc.pickle", 'rb') as f: # change file link for your machine
    data = pickle.load(f)

freq_channel = np.array(data["freq_channel"])
ue_loc = np.array(data["UE_loc"])
ue_speed = np.array(data["UE_speed"])
antenna_orient = np.array(data["antenna_orient"])

freq_channel = np.squeeze(freq_channel)


ant1_subcarrier1_csi = []
for x in range(450):
    csi = np.squeeze(freq_channel[x, :, :])
    ant1_subcarrier1_csi.append(csi[0][0])
print(np.array(ant1_subcarrier1_csi).shape)

real_parts = [num.real for num in ant1_subcarrier1_csi]
complex_parts = [num.imag for num in ant1_subcarrier1_csi]

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
train_size = int(0.8 * len(real_parts)) # can change if too big
train_data_r = real_parts[:train_size]
test_data_r = real_parts[train_size:]

train_data_c = complex_parts[:train_size]
test_data_c = complex_parts[train_size:]
from statsmodels.graphics.tsaplots import plot_acf
trend = np.arange(450)
seasonality = np.array(real_parts)
series = seasonality+trend
plot_acf(series)
plt.show() 

from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.api import AutoReg
from sklearn.metrics import mean_absolute_error, mean_squared_error


# Create and train the autoregressive model
lag_order = 160 # Adjust this based on the ACF plot
ar_model = AutoReg(train_data_r, lags=lag_order)
ar_results = ar_model.fit()

# Make predictions on the test set
y_pred = ar_results.predict(start=len(train_data_r), end=len(train_data_r) + len(test_data_r) - 1, dynamic=False)
print(y_pred)
print(test_data_r)
# Calculate MAE and RMSE
mae_r = mean_absolute_error(test_data_r, y_pred)
rmse_r = np.sqrt(mean_squared_error(test_data_r, y_pred))
print(f'Mean Absolute Error for Real comp: {mae_r:.16f}')
print(f'Root Mean Squared Error for Real comp: {rmse_r:.16f}')

plt.figure(figsize=(12, 6))
plt.plot(test_data_r, label='Actual CSI')
plt.plot(y_pred, label='Predicted CSI', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Re(CSI)')
plt.legend()
plt.title('Real part of CSI Prediction with Autoregressive Model')
plt.show()



ar_model_c = AutoReg(train_data_c, lags=lag_order)
ar_results_c = ar_model_c.fit()

# Make predictions on the test set
y_pred_c = ar_results_c.predict(start=len(train_data_c), end=len(train_data_c) + len(test_data_c) - 1, dynamic=False)
print(f'Predicted Complex: {y_pred_c}')
print(f'Actual Complex : {test_data_c}')

# Calculate MAE and RMSE
mae_c = mean_absolute_error(test_data_c, y_pred_c)
rmse_c = np.sqrt(mean_squared_error(test_data_c, y_pred_c))
print(f'Mean Absolute Error for Complex comp: {mae_c:.2f}')
print(f'Root Mean Squared Error for Complex comp: {rmse_c:.2f}')

plt.figure(figsize=(12, 6))
plt.plot(test_data_c, label='Actual CSI')
plt.plot(y_pred_c, label='Predicted CSI', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Complex(CSI)')
plt.legend()
plt.title('Complex part of CSI Prediction with Autoregressive Model')
plt.show()


predicted_csi = y_pred+1j*y_pred_c
