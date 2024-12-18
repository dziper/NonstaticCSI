from model import DecodableModel,Model
import numpy as np
from utils import Config
from dataset import Dataset
from reference_impl import ReferencePCA, ReferenceKmeans
from preprocessor import ComplexVectorPreprocessor
from typing import Tuple
from DCT_compression import DCTCompression
from DFT_compression import DFTCompression
from tensorflow.keras.models import Sequential
import tensorflow.keras.models as models
from tensorflow.keras.layers import LSTM, Dense
import copy
from tqdm.notebook import tqdm
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten,Dropout
from tensorflow.keras.layers import Input, Dense, MultiHeadAttention, LayerNormalization
import  tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs {gpus}")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Memory growth enabled for GPUs.")
    except RuntimeError as e:
        print(f"RuntimeError: {e}")




class FullLSTMModel(DecodableModel):
    def __init__(self, cfg: Config, matlab):
        self.cfg = cfg
        self.matlab = matlab

        print("This is the LSTM")
        self.pca = ReferencePCA(cfg, matlab)
        self.preprocessor = ComplexVectorPreprocessor(normalization=cfg.normalization_type,conversion_method=cfg.preprocessor_type)
        self.predictor = None
        # With NullPredictor, prediction_error is just zDL! This lets us test the ref impl

        if cfg.compressor_type == "kmeans":
            self.error_compressor = ReferenceKmeans(cfg, matlab)
        elif cfg.compressor_type == "dct":
            self.error_compressor = DCTCompression(cfg, matlab)
        elif cfg.compressor_type == "dft":
            self.error_compressor = DFTCompression(cfg, matlab)
        else:
            assert False, f"Unrecognized Compressor Type for LSTM Model {cfg.compressor_type}"

    def fit(self, dataset: Dataset):
        print("Fitting the PCA")

        self.pca.fit(dataset.csi_samples)
        zdl_train = self.pca.process(dataset.csi_samples)                # N * zdl_len

        X_train, y_train = self._preprocess(zdl_train, apply_existing=False)
        print("Fitting the LSTM")
        self._fit_LSTM(X_train, y_train)

        predicted_zdl_normalized = self.predictor.predict(X_train)

        y_pred_denormalized = self.preprocessor.denormalize_features(predicted_zdl_normalized)
        predicted_zdl = self.preprocessor.reconstruct_complex_data(y_pred_denormalized)

        prediction_error = zdl_train[self.cfg.predictor_window_size:] - predicted_zdl
        self.error_compressor.fit(prediction_error)

    def _preprocess(self, zdl, apply_existing=True):
        # Convert train data
        train_features = self.preprocessor.convert_complex_to_features(zdl)
        # Fit normalization factors on training data
        if not apply_existing:
            self.preprocessor.fit_normalization(train_features)
        # Normalize train features using fitted factors
        train_normalized = self.preprocessor.normalize_features(train_features, apply_existing=apply_existing)
        # Create windowed samples
        window_size = self.cfg.predictor_window_size
        X_train, y_train = self.preprocessor.create_windowed_samples(train_normalized, window_size)
        return X_train, y_train

    def _fit_LSTM(self, X_train, y_train):
        self.predictor = LSTMComplexPredictor(
            input_shape=(self.cfg.predictor_window_size, X_train.shape[2]),
            output_shape=X_train.shape[2]
        )
        self.predictor.train(X_train, y_train,epochs=self.cfg.epochs)

    def process(self, dataset: Dataset) -> Tuple[np.ndarray, np.ndarray]:
        zdl_test = self.pca.process(dataset.csi_samples)                # N * zdl_len

        X_test, y_test = self._preprocess(zdl_test, apply_existing=True)

        predicted_zdl_normalized = self.predictor.predict(X_test)

        y_pred_denormalized = self.preprocessor.denormalize_features(predicted_zdl_normalized)
        predicted_zdl = self.preprocessor.reconstruct_complex_data(y_pred_denormalized)

        print(f"Predicted zdl: {predicted_zdl.shape}")
        prediction_error = zdl_test[self.cfg.predictor_window_size:] - predicted_zdl
        compressed_error = self.error_compressor.process(prediction_error)
        return compressed_error, X_test

    def decode(self, compressed_error: np.ndarray, X_test: np.ndarray, return_zdl=False):
        ul_pred_error = self.error_compressor.decode(compressed_error)

        ul_pred_zdl = self.predictor.predict(X_test)
        ul_pred_zdl = self.preprocessor.denormalize_features(ul_pred_zdl)
        ul_pred_zdl = self.preprocessor.reconstruct_complex_data(ul_pred_zdl)

        print(f"Predicted zdl: {ul_pred_zdl.shape}")
        print(f"ul_pred_error: {ul_pred_error.shape}")
        ul_reconst_zdl = ul_pred_error + ul_pred_zdl
        ul_pred_csi = self.pca.decode(ul_reconst_zdl)
        if return_zdl:
            return ul_pred_csi, ul_pred_zdl
        return ul_pred_csi

    def get_initial_history(self, dataset: Dataset):
        initial_history = dataset.csi_samples[0:self.cfg.predictor_window_size]
        initial_history = self.pca.process(initial_history)
        initial_history = self.preprocessor.convert_complex_to_features(initial_history)
        initial_history = self.preprocessor.normalize_features(initial_history, apply_existing=True)
        return [initial_history[i] for i in range(len(initial_history))]

    def simulate_ue(self, dataset: Dataset, initial_history) -> np.ndarray:
        norm_zdl_history = copy.copy(initial_history)
        compressed_errors = []
        # History starts with initial "truth" or

        for i in tqdm(range(self.cfg.predictor_window_size, len(dataset))):
            # PCA -> Preproc -> Predict(Hist) -> KMeans
            new_csi = dataset.csi_samples[i:i+1]
            new_zdl = self.pca.process(new_csi)

            history_now = np.expand_dims(np.array(norm_zdl_history[-self.cfg.predictor_window_size:]), 0)
            predicted_zdl_normalized = self.predictor.predict(history_now)

            y_pred_denormalized = self.preprocessor.denormalize_features(predicted_zdl_normalized)
            predicted_zdl = self.preprocessor.reconstruct_complex_data(y_pred_denormalized)

            prediction_error = new_zdl - predicted_zdl
            compressed_error = self.error_compressor.process(prediction_error)

            compressed_errors.append(compressed_error.squeeze())

            # Simulate the reconstruction on BS
            decompressed_error = self.error_compressor.decode(compressed_error)
            reconstructed_zdl = decompressed_error + predicted_zdl

            # Normalize the ZDL before saving it to history
            train_features = self.preprocessor.convert_complex_to_features(reconstructed_zdl)
            normalized_zdl = self.preprocessor.normalize_features(train_features, apply_existing=True)

            norm_zdl_history.append(normalized_zdl.squeeze())

        return np.array(compressed_errors)

    def simulate_bs(self, compressed_errors, initial_csi_history):
        norm_zdl_history = copy.copy(initial_csi_history)
        pred_csis = []

        for i in tqdm(range(len(compressed_errors))):
            new_err = compressed_errors[i:i+1]
            ul_pred_error = self.error_compressor.decode(new_err)

            history_now = np.expand_dims(np.array(norm_zdl_history[-self.cfg.predictor_window_size:]), 0)
            ul_pred_zdl = self.predictor.predict(history_now)
            ul_pred_zdl = self.preprocessor.denormalize_features(ul_pred_zdl)
            ul_pred_zdl = self.preprocessor.reconstruct_complex_data(ul_pred_zdl)

            ul_reconst_zdl = ul_pred_error + ul_pred_zdl
            ul_pred_csi = self.pca.decode(ul_reconst_zdl)

            pred_csis.append(ul_pred_csi.squeeze())

            # Normalize the ZDL before saving it to history
            train_features = self.preprocessor.convert_complex_to_features(ul_reconst_zdl)
            normalized_zdl = self.preprocessor.normalize_features(train_features, apply_existing=True)
            norm_zdl_history.append(normalized_zdl.squeeze())

        return np.array(pred_csis)

    def load(self, path):
        pass

    def save(self, path):
        pass

    def _compute_pca_for_windows(self, dataset: Dataset):
        windows_shape = dataset.csi_windows.shape                   # N * window_size * na * nc
        zdl_train_windows = dataset.csi_windows.reshape(            # (N * window_size) * na * nc
            -1, windows_shape[2], windows_shape[3]
        )
        zdl_train_windows = self.pca.process(zdl_train_windows)
        zdl_train_windows = zdl_train_windows.reshape(
            windows_shape[0], windows_shape[1], -1
        )  # N * window_size * zdl_len
        return zdl_train_windows



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

        """model 0"""
        model = Sequential([
            LSTM(units, input_shape=input_shape, return_sequences=False),
            Dense(output_shape)
        ])
        model.compile(optimizer='adam', loss='mse')

        """model 1"""
        # model = Sequential([
        #     Conv1D(filters=units, kernel_size=3, activation='relu', input_shape=input_shape),
        #     MaxPooling1D(pool_size=2),
        #     LSTM(128, return_sequences=True),
        #     Dropout(0.2),
        #     LSTM(64),
        #     Dense(output_shape, activation='relu')
        # ])


        """model 2"""
        # model = Sequential([
        #     LSTM(units, input_shape=input_shape, return_sequences=False),  # Encodes the temporal dependencies
        #     Dropout(0.2),
        #     Dense(256, activation='relu'),  # Captures high-dimensional feature interactions
        #     Dropout(0.2),
        #     Dense(1000)  # Predicts 1,000 features for the next time step
        # ])
        # model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        # model.summary()
        #




        # # Input shape: 5 time steps, 1000 features
        # input_layer = Input(shape=input_shape)
        #
        # # Multi-head attention
        # attention_output = MultiHeadAttention(num_heads=8, key_dim=64)(input_layer, input_layer)
        #
        # # Layer normalization
        # attention_output = LayerNormalization()(attention_output)
        #
        # # Flatten to prepare for dense layers
        # flatten_layer = Flatten()(attention_output)
        #
        # # Dense layers for feature transformation
        # dense_layer = Dense(512, activation='relu')(flatten_layer)
        # dense_layer = Dropout(0.2)(dense_layer)
        #
        # # Output layer
        # output_layer = Dense(1000)(dense_layer)
        #
        # # Define model
        # model = models.Model(inputs=input_layer, outputs=output_layer)
        # model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        # model.summary()


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
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    def predict(self, X_test):
        """
        Make predictions using the trained model

        Args:
            X_test (np.ndarray): Test input samples

        Returns:
            np.ndarray: Predicted values
        """
        return self.model.predict(X_test, verbose=0)


class HistoryPredictor(Model):

    def __init__(self):
        pass

    def fit(self, *args):
        pass

    def process(self, data) -> np.ndarray:
        return np.average(data, axis=1)

    def load(self, path):
        pass

    def save(self, path):
        pass


class SimpleHistoryModel(DecodableModel):
    def __init__(self, cfg: Config, matlab):
        self.cfg = cfg
        self.matlab = matlab

        print("This is the History Model")
        self.pca = ReferencePCA(cfg, matlab)
        self.preprocessor = ComplexVectorPreprocessor(conversion_method=cfg.preprocessor_type)
        self.predictor = HistoryPredictor()
        # With NullPredictor, prediction_error is just zDL! This lets us test the ref impl

        if cfg.compressor_type == "kmeans":
            self.error_compressor = ReferenceKmeans(cfg, matlab)
        elif cfg.compressor_type == "dct":
            self.error_compressor = DCTCompression(cfg, matlab)
        elif cfg.compressor_type == "dft":
            self.error_compressor = DFTCompression(cfg, matlab)
        else:
            assert False, f"Unrecognized Compressor Type for LSTM Model {cfg.compressor_type}"

    def fit(self, dataset: Dataset):
        print("Fitting the PCA")

        self.pca.fit(dataset.csi_samples)
        zdl_train = self.pca.process(dataset.csi_samples)                # N * zdl_len

        X_train, y_train = self._preprocess(zdl_train, apply_existing=False)
        print("Fitting the LSTM")
        self._fit_LSTM(X_train, y_train)

        predicted_zdl_normalized = self.predictor.process(X_train)

        y_pred_denormalized = self.preprocessor.denormalize_features(predicted_zdl_normalized)
        predicted_zdl = self.preprocessor.reconstruct_complex_data(y_pred_denormalized)

        prediction_error = zdl_train[self.cfg.predictor_window_size:] - predicted_zdl
        self.error_compressor.fit(prediction_error)

    def _preprocess(self, zdl, apply_existing=True):
        # Convert train data
        train_features = self.preprocessor.convert_complex_to_features(zdl)
        # Fit normalization factors on training data
        self.preprocessor.fit_normalization(train_features)
        # Normalize train features using fitted factors
        train_normalized = self.preprocessor.normalize_features(train_features, apply_existing=apply_existing)
        # Create windowed samples
        window_size = self.cfg.predictor_window_size
        X_train, y_train = self.preprocessor.create_windowed_samples(train_normalized, window_size)
        return X_train, y_train


    def _fit_LSTM(self, X_train, y_train):
        pass

    def process(self, dataset: Dataset) -> Tuple[np.ndarray, np.ndarray]:
        zdl_test = self.pca.process(dataset.csi_samples)                # N * zdl_len

        X_test, y_test = self._preprocess(zdl_test, apply_existing=True)

        predicted_zdl_normalized = self.predictor.process(X_test)

        y_pred_denormalized = self.preprocessor.denormalize_features(predicted_zdl_normalized)
        predicted_zdl = self.preprocessor.reconstruct_complex_data(y_pred_denormalized)

        print(f"Predicted zdl: {predicted_zdl.shape}")
        prediction_error = zdl_test[self.cfg.predictor_window_size:] - predicted_zdl
        compressed_error = self.error_compressor.process(prediction_error)
        return compressed_error, X_test

    def decode(self, compressed_error: np.ndarray, X_test: np.ndarray):
        ul_pred_error = self.error_compressor.decode(compressed_error)

        ul_pred_zdl = self.predictor.process(X_test)

        ul_pred_zdl = self.preprocessor.denormalize_features(ul_pred_zdl)
        ul_pred_zdl = self.preprocessor.reconstruct_complex_data(ul_pred_zdl)

        print(f"Predicted zdl: {ul_pred_zdl.shape}")
        print(f"ul_pred_error: {ul_pred_error.shape}")
        ul_reconst_zdl = ul_pred_error + ul_pred_zdl
        ul_pred_csi = self.pca.decode(ul_reconst_zdl)
        return ul_pred_csi

    def load(self, path):
        pass

    def save(self, path):
        pass

    def _compute_pca_for_windows(self, dataset: Dataset):
        windows_shape = dataset.csi_windows.shape                   # N * window_size * na * nc
        zdl_train_windows = dataset.csi_windows.reshape(            # (N * window_size) * na * nc
            -1, windows_shape[2], windows_shape[3]
        )
        zdl_train_windows = self.pca.process(zdl_train_windows)
        zdl_train_windows = zdl_train_windows.reshape(
            windows_shape[0], windows_shape[1], -1
        )  # N * window_size * zdl_len
        return zdl_train_windows


class TruncatedLSTMModel(DecodableModel):
    def __init__(self, cfg: Config, matlab):
        self.cfg = cfg
        self.matlab = matlab

        print("This is the LSTM")
        self.pca = ReferencePCA(cfg, matlab)
        self.preprocessor = ComplexVectorPreprocessor(conversion_method=cfg.preprocessor_type)
        self.predictor = None
        # With NullPredictor, prediction_error is just zDL! This lets us test the ref impl

        if cfg.compressor_type == "kmeans":
            self.error_compressor = ReferenceKmeans(cfg, matlab)
        elif cfg.compressor_type == "dct":
            self.error_compressor = DCTCompression(cfg, matlab)
        elif cfg.compressor_type == "dft":
            self.error_compressor = DFTCompression(cfg, matlab)
        else:
            assert False, f"Unrecognized Compressor Type for LSTM Model {cfg.compressor_type}"

    def fit(self, dataset: Dataset):
        print("Fitting the PCA")

        self.pca.fit(dataset.csi_samples)
        zdl_train = self.pca.process(dataset.csi_samples)                # N * zdl_len

        X_train, y_train = self._preprocess(zdl_train, apply_existing=False)
        print("Fitting the LSTM")

        original_y_train_zeros = np.zeros_like(y_train)
        self.correct_shape_thing = original_y_train_zeros.shape
        self.correct_dtype = original_y_train_zeros.dtype

        original_x_train_zeros = np.zeros_like(X_train)
        trunc_y_train = y_train[:, :self.cfg.trunc_lstm_pred]
        trunc_x_train = X_train[:, :, :self.cfg.trunc_lstm_pred]

        self._fit_LSTM(X_train, trunc_y_train)

        trunc_y_pred = self.predictor.predict(X_train)

        original_y_train_zeros[:, :self.cfg.trunc_lstm_pred] = trunc_y_pred
        predicted_zdl_normalized = original_y_train_zeros

        y_pred_denormalized = self.preprocessor.denormalize_features(predicted_zdl_normalized)
        predicted_zdl = self.preprocessor.reconstruct_complex_data(y_pred_denormalized)

        prediction_error = zdl_train[self.cfg.predictor_window_size:] - predicted_zdl
        self.error_compressor.fit(prediction_error)

    def _preprocess(self, zdl, apply_existing=True):
        # Convert train data
        train_features = self.preprocessor.convert_complex_to_features(zdl)
        # Fit normalization factors on training data
        self.preprocessor.fit_normalization(train_features)
        # Normalize train features using fitted factors
        train_normalized = self.preprocessor.normalize_features(train_features, apply_existing=apply_existing)
        # Create windowed samples
        window_size = self.cfg.predictor_window_size
        X_train, y_train = self.preprocessor.create_windowed_samples(train_normalized, window_size)
        return X_train, y_train


    def _fit_LSTM(self, X_train, y_train):
        self.predictor = LSTMComplexPredictor(
            input_shape=(self.cfg.predictor_window_size, X_train.shape[2]),
            output_shape=y_train.shape[-1]
        )
        self.predictor.train(X_train, y_train,epochs=self.cfg.epochs)

    def process(self, dataset: Dataset) -> Tuple[np.ndarray, np.ndarray]:
        zdl_test = self.pca.process(dataset.csi_samples)                # N * zdl_len

        X_test, y_test = self._preprocess(zdl_test, apply_existing=True)
        original_y_test_zeros = np.zeros_like(y_test)
        original_x_test_zeros = np.zeros_like(X_test)
        trunc_y_test = y_test[:, :self.cfg.trunc_lstm_pred]
        trunc_x_test = X_test[:, :, :self.cfg.trunc_lstm_pred]

        trunc_y_pred = self.predictor.predict(X_test)

        original_y_test_zeros[:, :self.cfg.trunc_lstm_pred] = trunc_y_pred
        predicted_zdl_normalized = original_y_test_zeros

        y_pred_denormalized = self.preprocessor.denormalize_features(predicted_zdl_normalized)
        predicted_zdl = self.preprocessor.reconstruct_complex_data(y_pred_denormalized)

        print(f"Predicted zdl: {predicted_zdl.shape}")
        prediction_error = zdl_test[self.cfg.predictor_window_size:] - predicted_zdl
        compressed_error = self.error_compressor.process(prediction_error)
        return compressed_error, X_test

    def decode(self, compressed_error: np.ndarray, X_test: np.ndarray):
        ul_pred_error = self.error_compressor.decode(compressed_error)

        trunc_x_test = X_test[:, :, :self.cfg.trunc_lstm_pred]
        ul_pred_zdl_trunc = self.predictor.predict(X_test)

        ul_pred_zdl = np.zeros((X_test.shape[0], X_test.shape[2]), self.correct_dtype)
        ul_pred_zdl[:, :self.cfg.trunc_lstm_pred] = ul_pred_zdl_trunc

        ul_pred_zdl = self.preprocessor.denormalize_features(ul_pred_zdl)
        ul_pred_zdl = self.preprocessor.reconstruct_complex_data(ul_pred_zdl)

        print(f"Predicted zdl: {ul_pred_zdl.shape}")
        print(f"ul_pred_error: {ul_pred_error.shape}")
        ul_reconst_zdl = ul_pred_error + ul_pred_zdl
        ul_pred_csi = self.pca.decode(ul_reconst_zdl)
        return ul_pred_csi

    def load(self, path):
        pass

    def save(self, path):
        pass

    def _compute_pca_for_windows(self, dataset: Dataset):
        windows_shape = dataset.csi_windows.shape                   # N * window_size * na * nc
        zdl_train_windows = dataset.csi_windows.reshape(            # (N * window_size) * na * nc
            -1, windows_shape[2], windows_shape[3]
        )
        zdl_train_windows = self.pca.process(zdl_train_windows)
        zdl_train_windows = zdl_train_windows.reshape(
            windows_shape[0], windows_shape[1], -1
        )  # N * window_size * zdl_len
        return zdl_train_windows

