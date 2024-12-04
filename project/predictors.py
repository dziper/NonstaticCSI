import numpy as np
from statsmodels.tsa.api import VAR
from model import Model
from utils import Config


class VARPredictor(Model):
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def fit(self, data):
        real_data = np.real(data)  # Now shape is (450, 2560)
        real_model = VAR(real_data)
        self.real_results = real_model.fit(maxlags=self.cfg.predictor_window_size)
        self.real_lag_order = self.real_results.k_ar

        imag_data = np.imag(data)  # Now shape is (450, 2560)
        imag_model = VAR(imag_data)
        self.imag_results = imag_model.fit(maxlags=self.cfg.predictor_window_size)
        self.imag_lag_order = self.imag_results.k_ar


    def process(self, data) -> np.ndarray:
        real_data = np.real(data)
        real_last_obs = real_data[-self.real_lag_order:]  # Get the last 'lag_order' rows
        real_forecast = self.real_results.forecast(real_last_obs, steps=1)

        imag_data = np.imag(data)
        imag_last_obs = imag_data[-self.imag_lag_order:]  # Get the last 'lag_order' rows
        imag_forecast = self.imag_results.forecast(imag_last_obs, steps=1)
        return real_forecast + 1j * imag_forecast

    def load(self, path):
        pass

    def save(self, path):
        pass
