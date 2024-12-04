import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

class ComplexTimeSeriesPredictor:
    def __init__(self, train_input, train_output, test_input, test_output, batch_size=32, epochs=20, mode="amp_phase"):
        """
        Initialize the predictor for complex time series.

        Args:
            train_input (np.ndarray): Complex training input data of shape (samples, time_steps).
            train_output (np.ndarray): Complex training output data of shape (samples,).
            test_input (np.ndarray): Complex testing input data of shape (samples, time_steps).
            test_output (np.ndarray): Complex testing output data of shape (samples,).
            batch_size (int): Batch size for training.
            epochs (int): Number of epochs for training.
            mode (str): Prediction mode - "amp_phase" or "real_imag".
        """
        assert mode in ["amp_phase", "real_imag"], "Mode must be 'amp_phase' or 'real_imag'."
        self.mode = mode
        self.batch_size = batch_size
        self.epochs = epochs

        # Convert complex data to selected representation
        self.train_input = self.convert_complex_to_representation(train_input)
        self.train_output = self.convert_complex_to_representation(train_output)
        self.test_input = self.convert_complex_to_representation(test_input)
        self.test_output = self.convert_complex_to_representation(test_output)

        # Normalize data
        self.scaler_input = None
        self.scaler_output = None
        self.prepare_data()

        self.model = None

    def convert_complex_to_representation(self, data):
        """
        Convert complex data to amplitude/phase or real/imaginary.

        Args:
            data (np.ndarray): Complex data to convert.

        Returns:
            np.ndarray: Converted data.
        """
        if self.mode == "amp_phase":
            amplitude = np.abs(data)
            phase = np.angle(data)
            return np.stack([amplitude, phase], axis=-1)
        elif self.mode == "real_imag":
            real = np.real(data)
            imag = np.imag(data)
            return np.stack([real, imag], axis=-1)

    def prepare_data(self):
        """
        Prepare data by normalizing inputs and outputs.
        """
        self.train_input, self.scaler_input = self.normalize_data(self.train_input)
        self.train_output, self.scaler_output = self.normalize_data(self.train_output)
        self.test_input, _ = self.normalize_data(self.test_input, self.scaler_input)
        self.test_output, _ = self.normalize_data(self.test_output, self.scaler_output)

    def normalize_data(self, data, scaler=None):
        """
        Normalize data using MinMaxScaler.

        Args:
            data (np.ndarray): Data to normalize.
            scaler (MinMaxScaler, optional): Pre-fitted scaler. If None, a new scaler is created.

        Returns:
            np.ndarray: Normalized data.
            MinMaxScaler: Scaler object for inverse transformation.
        """
        if scaler is None:
            scaler = MinMaxScaler(feature_range=(0, 1))
            normalized_data = scaler.fit_transform(data.reshape(-1, data.shape[-1])).reshape(data.shape)
        else:
            normalized_data = scaler.transform(data.reshape(-1, data.shape[-1])).reshape(data.shape)
        return normalized_data, scaler

    def build_lstm_model(self):
        """
        Build the LSTM model for prediction.

        Returns:
            Sequential: Compiled LSTM model.
        """
        model = Sequential()
        model.add(LSTM(50, activation='relu', input_shape=(self.train_input.shape[1], self.train_input.shape[2])))
        model.add(Dense(self.train_output.shape[1]))  # Outputs match the number of features
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        return model

    def train(self):
        """
        Train the LSTM model.
        """
        self.model = self.build_lstm_model()
        self.model.fit(self.train_input, self.train_output, epochs=self.epochs, batch_size=self.batch_size,
                       validation_data=(self.test_input, self.test_output))

    def evaluate(self):
        """
        Evaluate the model and compute metrics.

        Returns:
            dict: Evaluation metrics (e.g., NMSE).
        """
        y_pred = self.model.predict(self.test_input)

        # Inverse transform predictions and true values
        y_pred_original = self.scaler_output.inverse_transform(y_pred)
        y_test_original = self.scaler_output.inverse_transform(self.test_output)

        # Compute NMSE
        nmse = self.calculate_nmse(y_test_original, y_pred_original)
        return {'NMSE': nmse}

    @staticmethod
    def calculate_nmse(y_true, y_pred):
        """
        Compute Normalized Mean Squared Error (NMSE).

        Args:
            y_true (np.ndarray): True values.
            y_pred (np.ndarray): Predicted values.

        Returns:
            float: NMSE.
        """
        mse = np.mean((y_true - y_pred) ** 2)
        norm = np.mean(y_true ** 2)
        return mse / norm

    def plot_predictions(self):
        """
        Plot true vs predicted values.
        """
        y_pred = self.model.predict(self.test_input)
        y_pred_original = self.scaler_output.inverse_transform(y_pred)
        y_test_original = self.scaler_output.inverse_transform(self.test_output)

        if self.mode == "amp_phase":
            labels = ['Amplitude', 'Phase']
        else:  # "real_imag"
            labels = ['Real', 'Imaginary']

        plt.figure(figsize=(12, 6))
        for i in range(self.train_output.shape[1]):  # Iterate over output features
            plt.subplot(self.train_output.shape[1], 1, i + 1)
            plt.plot(y_test_original[:, i], label='True')
            plt.plot(y_pred_original[:, i], label='Predicted', linestyle='--')
            plt.title(f'{labels[i]} Prediction')
            plt.legend()
        plt.tight_layout()
        plt.show()

import utils
import dataset
# Use this cfg variable whenever we need to access some constant
cfg = utils.Config(
    num_rx_antennas=1,
    num_tx_antennas=64,
    num_subcarriers=160,
    train_test_split=0.8,
    data_root="../data/dataset1",
    # duplicate_data=1,
    # data_snr=-1
)


train_set, test_set = dataset.load_data(cfg)

train_input = utils.reshape_tensor(train_set.csi_windows,K=2)
train_output = utils.reshape_tensor(train_set.csi_samples,K=2)

test_input = utils.reshape_tensor(test_set.csi_windows,K=2)
test_output = utils.reshape_tensor(test_set.csi_samples,K=2)

# Instantiate and train the predictor
predictor = ComplexTimeSeriesPredictor(train_input, train_output, test_input, test_output, mode="real_imag")
predictor.train()

# Evaluate and plot predictions
metrics = predictor.evaluate()
print(f"Performance Metrics: {metrics}")
predictor.plot_predictions()