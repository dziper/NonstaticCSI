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


np.random.seed(543)
class NoisePredictor(DecodableModel):

    def __init__(self, cfg: Config, matlab):
        self.cfg = cfg
        self.matlab = matlab

        self.noise = None
        self.noise_std = None
        self.predicted_zdl = None
        print("Loading noise predictor")
        self.predictor = None
        # With NullPredictor, prediction_error is just zDL! This lets us test the ref impl

        self.preprocessor = ComplexVectorPreprocessor(conversion_method="real_imag")


        print("This is the LSTM")
        self.pca = ReferencePCA(cfg, matlab)

        if cfg.compressor_type == "kmeans":
            self.error_compressor = ReferenceKmeans(cfg, matlab)
        elif cfg.compressor_type == "dct":
            self.error_compressor = DCTCompression(cfg, matlab)
        elif cfg.compressor_type == "dft":
            self.error_compressor = DFTCompression(cfg, matlab)
        else:
            assert False, f"Unrecognized Compressor Type for LSTM Model {cfg.compressor_type}"

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

    #
    # def _fit_LSTM(self, X_train, y_train):
    #     self.predictor = LSTMComplexPredictor(
    #         input_shape=(self.cfg.predictor_window_size, X_train.shape[2]),
    #         output_shape=X_train.shape[2]
    #     )
    #     self.predictor.train(X_train, y_train,epochs=self.cfg.epochs)



    def fit(self, dataset: Dataset):
        print("Fitting the PCA")

        self.pca.fit(dataset.csi_samples)
        zdl_train = self.pca.process(dataset.csi_samples)                # N * zdl_len


        X_train, y_train = self._preprocess(zdl_train, apply_existing=False)
        print("Fitting the LSTM")

        # self.noise_std = np.sqrt(np.max(np.abs(zdl_train)))/10000
        self.noise_std = 1/1e9
        noise =  self.noise_std * np.random.randn(*zdl_train.shape)+self.noise_std*1j*np.random.randn(*zdl_train.shape)
        # noise = np.sqrt(np.max(np.abs(zdl_train)))/10000 * np.zeros_like(zdl_train)+np.sqrt(np.max(np.abs(zdl_train)))/10000*1j*np.zeros_like(zdl_train)
        predicted_zdl = zdl_train + noise


        prediction_error = zdl_train - predicted_zdl
        self.error_compressor.fit(prediction_error)



    def process(self, dataset: Dataset) -> Tuple[np.ndarray, np.ndarray]:

        self.pca.fit(dataset.csi_samples)
        zdl_test = self.pca.process(dataset.csi_samples)


        self.noise = self.noise_std* np.random.randn(*zdl_test.shape)+self.noise_std*np.random.randn(*zdl_test.shape)
        # self.noise = np.sqrt(np.max(np.abs(zdl_test)))/10000 * np.zeros(zdl_test.shape)+np.sqrt(np.max(np.abs(zdl_test)))/10000*1j*np.zeros(zdl_test.shape)
        predicted_zdl = zdl_test + self.noise

        self.predicted_zdl = predicted_zdl

        print(f"Predicted zdl: {predicted_zdl.shape}")
        prediction_error = zdl_test - self.predicted_zdl
        self.error_compressor.fit(prediction_error)

        compressed_error = self.error_compressor.process(prediction_error)
        return compressed_error


    def decode(self, compressed_error: np.ndarray):
        ul_pred_error = self.error_compressor.decode(compressed_error)

        ul_pred_zdl = self.predicted_zdl


        print(f"Predicted zdl: {ul_pred_zdl.shape}")
        print(f"ul_pred_error: {ul_pred_error.shape}")
        ul_reconst_zdl = ul_pred_error + ul_pred_zdl
        ul_pred_csi = self.pca.decode(ul_reconst_zdl)
        return ul_pred_csi, ul_pred_zdl

    def load(self, path):
        pass

    def save(self, path):
        pass