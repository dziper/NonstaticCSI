from model import DecodableModel
import numpy as np
from utils import Config
from dataset import Dataset
from reference_impl import ReferencePCA, ReferenceKmeans
from preprocessor import ComplexVectorPreprocessor
from typing import Tuple
from DCT_compression import DCTCompression
from DFT_compression import DFTCompression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

class FullLSTMModelSensor(DecodableModel):
    def __init__(self, cfg: Config, matlab):
        self.cfg = cfg
        self.matlab = matlab

        print("This is the LSTM")
        self.pca = ReferencePCA(cfg, matlab)
        self.preprocessor = ComplexVectorPreprocessor(conversion_method="real_imag")
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

        locations = dataset.ue_locations
        _, y_train = self._preprocess(zdl_train, apply_existing=False)
        X_train, _ = self._create_training_data(locations,zdl_train)

        print("Fitting the LSTM")
        self._fit_LSTM(X_train, y_train)

        X_train = X_train.reshape(X_train.shape[0], -1)

        predicted_zdl_normalized = self.predictor.predict(X_train)

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
        # self.predictor = LSTMComplexPredictor(
        #     input_shape=(self.cfg.predictor_window_size,3),
        #     output_shape=y_train.shape[1]
        #
        # )
        #
        # self.predictor.train(X_train, y_train, epochs=self.cfg.epochs)


        # Initialize and train the model
        self.predictor = XGBoostPredictor(n_estimators=15, max_depth=7, learning_rate=0.3)


        # Flatten X to shape (14280, 60)
        X_train = X_train.reshape(X_train.shape[0], -1)
        print(X_train.shape)
        print(y_train.shape)
        self.predictor.train(X_train, y_train)

    def process(self, dataset: Dataset) -> Tuple[np.ndarray, np.ndarray]:
        zdl_test = self.pca.process(dataset.csi_samples)                # N * zdl_len

        _, y_test = self._preprocess(zdl_test, apply_existing=True)
        X_test,_ = self._create_training_data(dataset.ue_locations,zdl_test)

        X_test = X_test.reshape(X_test.shape[0], -1)
        predicted_zdl_normalized = self.predictor.predict(X_test)

        y_pred_denormalized = self.preprocessor.denormalize_features(predicted_zdl_normalized)
        predicted_zdl = self.preprocessor.reconstruct_complex_data(y_pred_denormalized)

        print(f"Predicted zdl: {predicted_zdl.shape}")
        prediction_error = zdl_test[self.cfg.predictor_window_size:] - predicted_zdl
        compressed_error = self.error_compressor.process(prediction_error)
        return compressed_error, X_test

    def decode(self, compressed_error: np.ndarray, X_test: np.ndarray):
        ul_pred_error = self.error_compressor.decode(compressed_error)


        ul_pred_zdl = self.predictor.predict(X_test)
        ul_pred_zdl = self.preprocessor.denormalize_features(ul_pred_zdl)
        ul_pred_zdl = self.preprocessor.reconstruct_complex_data(ul_pred_zdl)

        print(f"Predicted zdl: {ul_pred_zdl.shape}")
        print(f"ul_pred_error: {ul_pred_error.shape}")
        ul_reconst_zdl = ul_pred_error + ul_pred_zdl
        ul_pred_csi = self.pca.decode(ul_reconst_zdl)
        return ul_pred_csi, ul_pred_zdl

    def _create_training_data(self,location,zdl):

        X, y = [], []
        for i in range(len(location) - self.cfg.predictor_window_size):
            X.append(location[i:i + self.cfg.predictor_window_size])
            y.append(zdl[i + self.cfg.predictor_window_size])

        return np.array(X),np.array(y)

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


import xgboost as xgb


class XGBoostPredictor:
    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.1):
        """
        Initialize the XGBoost model for prediction

        Args:
            n_estimators (int): Number of boosting rounds
            max_depth (int): Maximum depth of trees
            learning_rate (float): Step size shrinkage used in updates
        """
        self.model = self._build_model(n_estimators, max_depth, learning_rate)

    def _build_model(self, n_estimators, max_depth, learning_rate):
        """
        Build the XGBoost model architecture

        Args:
            n_estimators (int): Number of boosting rounds
            max_depth (int): Maximum depth of trees
            learning_rate (float): Step size shrinkage used in updates

        Returns:
            xgb.XGBRegressor: Initialized XGBoost model
        """
        return  xgb.XGBRegressor(
                tree_method="hist",
                # objective="reg:squaredlogerror",
                objective="reg:squarederror",
                # objective="reg:absoluteerror",
                # n_estimators=4,
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                n_jobs=16,
                multi_strategy="one_output_per_tree",  # {'multi_output_tree', 'one_output_per_tree'}
                subsample=0.8,
                verbosity = 2
            )


    def train(self, X_train, y_train):
        """
        Train the XGBoost model

        Args:
            X_train (np.ndarray): Training input samples
            y_train (np.ndarray): Training labels (1D or 2D for multitask learning)
        """
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """
        Make predictions using the trained model

        Args:
            X_test (np.ndarray): Test input samples

        Returns:
            np.ndarray: Predicted values
        """
        return self.model.predict(X_test)

