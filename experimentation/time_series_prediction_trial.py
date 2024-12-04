# import pickle
#
# import numpy as np
# import matplotlib.pyplot as plt
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
# from tensorflow.keras.optimizers import Adam
# from sklearn.preprocessing import MinMaxScaler
#
# class ComplexVectorTimeSeriesPredictor:
#     def __init__(self, X_train, X_test, y_train, y_test, mode='real_imag', epochs=20, batch_size=32):
#         """
#         Initialize the ComplexVectorTimeSeriesPredictor.
#
#         Args:3
#
#             X_train (np.ndarray): Training input data (N_samples, M_windows, vector_length)
#             X_test (np.ndarray): Test input data (N_samples, M_windows, vector_length)
#             y_train (np.ndarray): Training output data (N_samples, vector_length)
#             y_test (np.ndarray): Test output data (N_samples, vector_length)
#             mode (str): Representation mode - 'real_imag' or 'angle_amplitude'
#             epochs (int): Number of training epochs
#             batch_size (int): Batch size for training
#         """
#         self.X_train = X_train
#         self.X_test = X_test
#         self.y_train = y_train
#         self.y_test = y_test
#         self.mode = mode
#         self.epochs = epochs
#         self.batch_size = batch_size
#         self.vector_size = y_train.shape[1]  # The size of each complex vector (real + imaginary)
#
#         self.model = None
#
#     def convert_complex_vector_to_features(self, complex_vector):
#         """
#         Convert complex vector to selected representation mode.
#
#         Args:
#             complex_vector (np.ndarray): Complex input vector of shape (N)
#
#         Returns:
#             np.ndarray: Converted feature representation of shape (2N) for real_imag or (2) for angle_amplitude
#         """
#         if self.mode == 'real_imag':
#             # Split into real and imaginary parts
#             return np.hstack([np.real(complex_vector), np.imag(complex_vector)])
#         elif self.mode == 'angle_amplitude':
#             # Convert to angle and amplitude
#             angle = np.angle(complex_vector)
#             amplitude = np.abs(complex_vector)
#
#             # Normalize angle and amplitude separately
#             scaler_angle = MinMaxScaler(feature_range=(0, 1))
#             scaler_amplitude = MinMaxScaler(feature_range=(0, 1))
#
#             angle_normalized = scaler_angle.fit_transform(angle.reshape(-1, 1)).flatten()
#             amplitude_normalized = scaler_amplitude.fit_transform(amplitude.reshape(-1, 1)).flatten()
#
#             return np.hstack([angle_normalized, amplitude_normalized])
#         else:
#             raise ValueError(f"Unsupported mode: {self.mode}")
#
#     def build_lstm_model(self):
#         """
#         Build LSTM model for time series prediction.
#
#         Returns:
#             Compiled Keras Sequential model.
#         """
#         model = Sequential([
#             LSTM(64, activation='relu', input_shape=(self.X_train.shape[1], self.X_train.shape[2])),
#             Dense(self.vector_size * 2)  # Output features for entire vector (real and imaginary parts)
#         ])
#         model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
#         return model
#
#     def train(self):
#         """
#         Train the LSTM model.
#         """
#         self.model = self.build_lstm_model()
#         self.model.fit(
#             self.X_train, self.y_train,
#             epochs=self.epochs,
#             batch_size=self.batch_size,
#             validation_data=(self.X_test, self.y_test)
#         )
#
#     def evaluate(self):
#         """
#         Evaluate model performance.
#
#         Returns:
#             Dict of performance metrics.
#         """
#         # Predict on test set
#         y_pred = self.model.predict(self.X_test)
#
#         z_dl_hat = self.convert_output_to_complex(y_pred)
#         z_dl = self.convert_output_to_complex(self.y_test)
#
#
#         # Calculate performance metrics
#         mse = np.mean(np.sum(np.abs(z_dl_hat - z_dl) ** 2,axis=1))
#         nmse = np.mean(mse /np.linalg.norm(z_dl,axis=1))
#
#         return {
#             'MSE': mse,
#             'NMSE': nmse
#         }
#
#     def plot_predictions(self, title="Prediction Comparison"):
#         """
#         Plot predicted vs actual values for first few vector elements.
#
#         Args:
#             title (str): Plot title.
#         """
#         y_pred = self.model.predict(self.X_test)
#
#         z_dl_hat = self.convert_output_to_complex(y_pred)
#         z_dl = self.convert_output_to_complex(self.y_test)
#
#         # Plot first few elements of the vector (adjust as needed)
#         plot_elements = min(5, self.vector_size)
#
#         plt.figure(figsize=(15, 10))
#
#         for i in range(plot_elements):
#             plt.subplot(2, plot_elements, i + 1)
#             plt.plot(np.angle(z_dl[:,i]), label='Actual', linestyle='-')
#             plt.plot(np.angle(z_dl_hat[:,i]), label='Predicted', linestyle='--')
#             plt.title(f'{self.mode.replace("_", " ").title()} - Feature {i + 1}')
#             plt.legend()
#
#             plt.subplot(2, plot_elements, i + plot_elements + 1)
#             plt.plot(np.abs(z_dl[:, i]), label='Actual', linestyle='-')
#             plt.plot(np.abs(z_dl_hat[:, i]), label='Predicted', linestyle='--')
#             plt.title(f'{self.mode.replace("_", " ").title()} - Feature {i + 1} (2nd Part)')
#             plt.legend()
#
#         plt.tight_layout()
#         plt.suptitle(title)
#         plt.show()
#
#     def convert_output_to_complex(self, y_pred):
#         """
#         Convert predicted real and imaginary parts back to complex vector.
#
#         Args:
#             y_pred (np.ndarray): Predicted output from the LSTM of shape (N_samples, vector_length * 2)
#
#         Returns:
#             np.ndarray: Reconstructed complex vector of shape (N_samples, vector_length)
#         """
#         real_part = y_pred[:, :self.vector_size]
#         imag_part = y_pred[:, self.vector_size:]
#
#         # Reconstruct complex vector
#         complex_output = real_part + 1j * imag_part
#         return complex_output
#
#     def preprocess_data(self, X_data, y_data):
#         """
#         Preprocess data by converting complex vectors to features.
#
#         Args:
#             X_data (np.ndarray): Complex input data of shape (N_samples, M_windows, vector_length)
#             y_data (np.ndarray): Complex output data of shape (N_samples, vector_length)
#
#         Returns:
#             np.ndarray: Preprocessed input data for LSTM (N_samples, M_windows, vector_length * 2)
#             np.ndarray: Preprocessed output data for LSTM (N_samples, vector_length * 2)
#         """
#         # Flatten the complex data to real and imaginary parts or angle-amplitude based on mode
#         X_processed = np.array([
#             np.array([self.convert_complex_vector_to_features(x) for x in sample])  # Convert each vector in the sample
#             for sample in X_data
#         ])
#
#         # Process output data as well
#         y_processed = np.array([self.convert_complex_vector_to_features(y) for y in y_data])
#
#         # The resulting data should now be of shape (N_samples, M_windows, vector_length * 2) for X
#         # and (N_samples, vector_length * 2) for y
#         return X_processed, y_processed
#
#
# def main(train_set_input, train_set_output, test_set_input, test_set_output):
#     """
#     Main function to demonstrate predictor usage.
#
#     Args:
#         train_set_input (np.ndarray): Training input data of shape (N_samples, M_windows, vector_length)
#         train_set_output (np.ndarray): Training output data of shape (N_samples, vector_length)
#         test_set_input (np.ndarray): Testing input data of shape (N_samples, M_windows, vector_length)
#         test_set_output (np.ndarray): Testing output data of shape (N_samples, vector_length)
#     """
#     # Initialize the predictor
#     predictor = ComplexVectorTimeSeriesPredictor(
#         train_set_input, test_set_input, train_set_output, test_set_output,
#         mode='real_imag',
#         epochs=20, batch_size=32
#     )
#
#     # Preprocess the data to convert complex vectors to features
#     X_train_processed, y_train_processed = predictor.preprocess_data(train_set_input, train_set_output)
#     X_test_processed, y_test_processed = predictor.preprocess_data(test_set_input, test_set_output)
#
#     # Train and evaluate the model
#     predictor.X_train = X_train_processed
#     predictor.X_test = X_test_processed
#     predictor.y_train = y_train_processed
#     predictor.y_test = y_test_processed
#
#     predictor.train()
#
#     # Print performance metrics
#     print("Real-Imaginary Mode Performance:")
#     print(predictor.evaluate())
#     predictor.plot_predictions(title="Real-Imaginary Parts Prediction")


import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

import reference_impl as ref
import model
import utils
import dataset


class ComplexVectorPreprocessor:
    def __init__(self, conversion_method='real_imag'):
        """
        Initialize the preprocessor with conversion method and normalization tracking

        Args:
            conversion_method (str): Method to convert complex numbers
                                     ('real_imag' or 'amplitude_angle')
        """
        self.conversion_method = conversion_method
        self.normalization_factors = {}

    def convert_complex_to_features(self, complex_data):
        """
        Convert complex data to real features based on selected method

        Args:
            complex_data (np.ndarray): Input complex data of shape (Nsamples, Nfeatures)

        Returns:
            np.ndarray: Converted real-valued features
        """
        if self.conversion_method == 'real_imag':
            return np.column_stack([complex_data.real, complex_data.imag])
        elif self.conversion_method == 'amplitude_angle':
            return np.column_stack([np.abs(complex_data), np.angle(complex_data)])
        else:
            raise ValueError("Invalid conversion method. Choose 'real_imag' or 'amplitude_angle'")

    def fit_normalization(self, features):
        """
        Compute and store normalization factors from training data

        Args:
            features (np.ndarray): Input features to compute normalization factors

        Returns:
            self: Allows method chaining
        """
        for i in range(features.shape[1]):
            max_val = np.max(np.abs(features[:, i]))
            self.normalization_factors[i] = max_val

        return self

    def normalize_features(self, features, apply_existing=False):
        """
        Normalize features using stored or computed normalization factors

        Args:
            features (np.ndarray): Input features to normalize
            apply_existing (bool): If True, use existing normalization factors

        Returns:
            np.ndarray: Normalized features
        """
        normalized_features = np.zeros_like(features)

        for i in range(features.shape[1]):
            if apply_existing:
                # Use pre-computed normalization factor from training data
                if i not in self.normalization_factors:
                    raise ValueError(f"Normalization factor for feature {i} not found. Call fit_normalization() first.")
                max_val = self.normalization_factors[i]
            else:
                # Compute new normalization factor
                max_val = np.max(np.abs(features[:, i]))
                self.normalization_factors[i] = max_val

            # Normalize using the selected max value
            normalized_features[:, i] = features[:, i] / (max_val + 1e-8)

        return normalized_features

    def denormalize_features(self, normalized_features):
        """
        Convert normalized features back to original scale using stored normalization factors

        Args:
            normalized_features (np.ndarray): Normalized input features

        Returns:
            np.ndarray: Denormalized features
        """
        denormalized_features = np.zeros_like(normalized_features)

        for i in range(normalized_features.shape[1]):
            max_val = self.normalization_factors[i]
            denormalized_features[:, i] = normalized_features[:, i] * (max_val + 1e-8)

        return denormalized_features

    def reconstruct_complex_data(self, denormalized_features):
        """
        Reconstruct complex data from denormalized features

        Args:
            denormalized_features (np.ndarray): Denormalized features

        Returns:
            np.ndarray: Reconstructed complex data
        """
        if self.conversion_method == 'real_imag':
            # Split features into real and imaginary parts
            half_features = denormalized_features.shape[1] // 2
            return denormalized_features[:, :half_features] + \
                1j * denormalized_features[:, half_features:]

        elif self.conversion_method == 'amplitude_angle':
            # Reconstruct from magnitude and phase
            half_features = denormalized_features.shape[1] // 2
            magnitudes = denormalized_features[:, :half_features]
            phases = denormalized_features[:, half_features:]
            return magnitudes * np.exp(1j * phases)

        else:
            raise ValueError("Invalid conversion method")

    def create_windowed_samples(self, normalized_data, window_size):
        """
        Create windowed samples with M-1 steps as input and Mth step as label

        Args:
            normalized_data (np.ndarray): Normalized input data
            window_size (int): Size of the sliding window

        Returns:
            tuple: X (input samples), y (labels)
        """
        X, y = [], []
        for i in range(len(normalized_data) - window_size):
            X.append(normalized_data[i:i + window_size])
            y.append(normalized_data[i + window_size])

        return np.array(X), np.array(y)


class LSTMComplexPredictor:
    def __init__(self, input_shape, output_shape, units=100):
        """
        Initialize LSTM model for complex vector prediction

        Args:
            input_shape (tuple): Shape of input data
            output_shape (int): Number of output features
            units (int): Number of LSTM units
        """
        self.model = self._build_model(input_shape, output_shape, units)

    def _build_model(self, input_shape, output_shape, units):
        """
        Build LSTM model architecture

        Args:
            input_shape (tuple): Shape of input data
            output_shape (int): Number of output features
            units (int): Number of LSTM units

        Returns:
            tf.keras.Model: Compiled LSTM model
        """
        model = Sequential([
            LSTM(units, input_shape=input_shape, return_sequences=False),
            Dense(output_shape)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def train(self, X_train, y_train, epochs=30, batch_size=16):
        """
        Train the LSTM model

        Args:
            X_train (np.ndarray): Training input samples
            y_train (np.ndarray): Training labels
            epochs (int): Number of training epochs
            batch_size (int): Training batch size
        """
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    def predict(self, X_test):
        """
        Make predictions using the trained model

        Args:
            X_test (np.ndarray): Test input samples

        Returns:
            np.ndarray: Predicted values
        """
        return self.model.predict(X_test)


def complex_normalized_mean_squared_error(y_true, y_pred):
    """
    Calculate Normalized Mean Squared Error (NMSE) for complex vectors

    Args:
        y_true (np.ndarray): Ground truth complex values
        y_pred (np.ndarray): Predicted complex values

    Returns:
        float: Complex NMSE value
    """
    # Compute squared error with complex subtraction
    squared_error = np.abs(y_true - y_pred) ** 2

    # Compute mean squared error
    mse = np.mean(squared_error)

    # Compute normalized error (power of ground truth)
    normalization_factor = np.mean(np.abs(y_true) ** 2)

    # Compute NMSE
    nmse = mse / (normalization_factor + 1e-8)

    return nmse


def plot_prediction_comparison(true_data, pred_data):
    """
    Plot abs and angle values of predicted vs ground truth complex vectors

    Args:
        true_data (np.ndarray): Ground truth complex data
        pred_data (np.ndarray): Predicted complex data
    """
    plt.figure(figsize=(15, 6))

    # Abs comparison
    plt.subplot(1, 2, 1)
    plt.title('Magnitude Comparison')
    plt.plot(np.abs(true_data), label='True Magnitude')
    plt.plot(np.abs(pred_data), label='Predicted Magnitude')
    plt.legend()

    # Angle comparison
    plt.subplot(1, 2, 2)
    plt.title('Phase Comparison')
    plt.plot(np.angle(true_data), label='True Phase')
    plt.plot(np.angle(pred_data), label='Predicted Phase')
    plt.legend()

    plt.tight_layout()
    plt.show()


def main():

    import error_compression
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

    import matlab.engine
    matlab = matlab.engine.start_matlab()

    pca = ref.ReferencePCA(cfg, matlab)
    model.train_or_load(pca, cfg.pca_path, cfg.retrain_all, train_set.csi_samples)  # pca.fit() includes reduce overhead

    # with open("C:\\Users\ibrahimkilinc\Documents\ECE257_Project\csis.pickle","rb") as f:
    #     [zdl_train,zdl_train_windows] = pickle.load(f)
    #

    # test_set_input = error_compression.reshape_tensor(test_set.csi_windows,K=2)
    # test_set_output = error_compression.reshape_tensor(test_set.csi_samples,K=1)

    zdl_train = pca.process(train_set.csi_samples)
    zdl_test = pca.process(test_set.csi_samples)

    train_complex_data = zdl_train
    test_complex_data = zdl_test

    # Preprocessing
    preprocessor = ComplexVectorPreprocessor(conversion_method='real_imag')

    # Convert train data
    train_features = preprocessor.convert_complex_to_features(train_complex_data)

    # Fit normalization factors on training data
    preprocessor.fit_normalization(train_features)

    # Normalize train features using fitted factors
    train_normalized = preprocessor.normalize_features(train_features)

    # Convert and normalize test data using training data's normalization factors
    test_features = preprocessor.convert_complex_to_features(test_complex_data)
    test_normalized = preprocessor.normalize_features(test_features, apply_existing=True)

    # Create windowed samples
    window_size = 10
    X_train, y_train = preprocessor.create_windowed_samples(train_normalized, window_size)
    X_test, y_test = preprocessor.create_windowed_samples(test_normalized, window_size)

    # Initialize and train LSTM predictor
    predictor = LSTMComplexPredictor(
        input_shape=(window_size, X_train.shape[2]),
        output_shape=X_train.shape[2]
    )
    predictor.train(X_train, y_train)

    # Predict on test data
    y_pred_normalized = predictor.predict(X_test)

    # Denormalize predictions
    y_pred_denormalized = preprocessor.denormalize_features(y_pred_normalized)

    # Reconstruct complex predictions
    y_pred_complex = preprocessor.reconstruct_complex_data(y_pred_denormalized)

    # Reconstruct true complex values for comparison
    true_complex = preprocessor.reconstruct_complex_data(
        preprocessor.denormalize_features(y_test)
    )

    # Calculate Complex NMSE
    nmse = complex_normalized_mean_squared_error(true_complex, y_pred_complex)
    print(f"Complex Normalized Mean Squared Error: {nmse}")

    # H_hat = pca.decode(y_pred_complex[0:1])
    # H_true = pca.decode(zdl_test[0:1])
    #

    # Reconstruct true complex values for comparison
    true_complex = preprocessor.reconstruct_complex_data(
        preprocessor.denormalize_features(y_train)
    )

    # Predict on test data
    y_pred_normalized = predictor.predict(X_train)

    # Denormalize predictions
    y_pred_denormalized = preprocessor.denormalize_features(y_pred_normalized)

    # Reconstruct complex predictions
    y_pred_complex = preprocessor.reconstruct_complex_data(y_pred_denormalized)

    H_true= pca.decode(true_complex[0:1])
    H_hat = pca.decode(y_pred_complex[0:1])


    plt.figure()
    plt.imshow(np.squeeze(np.abs(H_hat)))
    plt.show()
    plt.figure()
    plt.imshow(np.squeeze(np.abs(H_true)))
    plt.show()

    # # Plot prediction comparison
    # plot_prediction_comparison(true_complex[0], y_pred_complex[0])




if __name__ == "__main__":
    main()

