from model import Model, DecodableModel
import numpy as np
from utils import Config
from scipy.fftpack import fftn, ifftn
from tqdm.notebook import tqdm
from dataset import Dataset
from reference_impl import ReferencePCA, ReferenceKmeans
from time_series_prediction_trial import LSTMComplexPredictor, ComplexVectorPreprocessor
from typing import Tuple


class FullLSTMModel(DecodableModel):
    def __init__(self, cfg: Config, matlab):
        self.cfg = cfg
        self.matlab = matlab

        print("This is the LSTM")
        self.pca = ReferencePCA(cfg, matlab)
        self.preprocessor = ComplexVectorPreprocessor(conversion_method="real_imag")
        self.predictor = None
        # With NullPredictor, prediction_error is just zDL! This lets us test the ref impl
        self.error_compressor = ReferenceKmeans(cfg, matlab)

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
        self.predictor.train(X_train, y_train)


    def process(self, dataset: Dataset) -> Tuple[np.ndarray, np.ndarray]:
        zdl_test = self.pca.process(dataset.csi_samples)                # N * zdl_len

        X_test, y_test = self._preprocess(zdl_test, apply_existing=True)

        predicted_zdl_normalized = self.predictor.predict(X_test)

        y_pred_denormalized = self.preprocessor.denormalize_features(predicted_zdl_normalized)
        predicted_zdl = self.preprocessor.reconstruct_complex_data(y_pred_denormalized)

        prediction_error = zdl_test[self.cfg.predictor_window_size:] - predicted_zdl
        compressed_error = self.error_compressor.process(prediction_error)
        return compressed_error, X_test

    def decode(self, compressed_error: np.ndarray, X_test: np.ndarray):
        ul_pred_error = self.error_compressor.decode(compressed_error)

        ul_pred_zdl = self.predictor.predict(X_test)
        ul_pred_zdl = self.preprocessor.denormalize_features(ul_pred_zdl)
        ul_pred_zdl = self.preprocessor.reconstruct_complex_data(ul_pred_zdl)

        ul_reconst_zdl = ul_pred_error + ul_pred_zdl
        ul_pred_csi = self.pca.decode(ul_reconst_zdl)
        return ul_pred_csi, ul_pred_zdl

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

