from abc import ABC, abstractmethod
from utils import Config
from model import Model, DecodableModel
from copy import deepcopy
from typing import List
from dataset import DataSample, DataWindow, Dataset
import numpy as np


class SimpleSimulator(ABC):
    def __init__(self, cfg: Config, pca: DecodableModel, predictor: Model, error_compressor: DecodableModel):
        # Given pretrained components
        self.pca = pca
        self.predictor = predictor
        self.error_compressor = error_compressor
        self.csi_history = []
        self.cfg = cfg

    def add_history(self, zdl):
        self.csi_history.append(zdl)
        if self.csi_history > self.cfg.predictor_window_size:
            self.csi_history.pop(0)

    def set_history(self, csi_history: List[np.ndarray]):
        self.csi_history = deepcopy(csi_history)

    def reset(self):
        self.csi_history = []

    def get_current_csi_window(self):
        return self.csi_history[-self.cfg.predictor_window_size:]


class DLSimulator(SimpleSimulator):
    @abstractmethod
    def simulate(self, sample: DataSample):
        pass


class ULSimulator(SimpleSimulator):
    @abstractmethod
    def simulate(self, channel_input: np.ndarray) -> np.ndarray:
        # Returns a CSI Estimation for this timestamp
        pass


class DLSimple(DLSimulator):
    def simulate(self, sample: DataSample) -> np.ndarray:
        # Returns Quantized Error for this timestep
        zdl = self.pca.process(sample.csi)
        predicted = self.predictor.process(self.get_current_csi_window())
        error = zdl - predicted
        compressed_error = self.error_compressor.process(error)
        decompressed_error = self.error_compressor.decode(compressed_error)
        reconstructed_zdl = predicted + decompressed_error
        self.add_history(reconstructed_zdl)
        return compressed_error  # Send on channel


class ULSimple(ULSimulator):
    def simulate(self, compressed_error: np.ndarray) -> np.ndarray:
        decompressed_error = self.error_compressor.decode(compressed_error)
        predicted = self.predictor.process(self.get_current_csi_window())
        reconstructed_zdl = predicted + decompressed_error
        self.add_history(reconstructed_zdl)
        return self.pca.decode(reconstructed_zdl)  # Estimated CSI on UL Side


class Evaluator:
    def evaluate(self, sample: DataSample, predicted_csi: np.ndarray):
        # Evaluate the sample from one time step
        pass

    def report(self):
        # Finalize evaluation and provide report
        pass

    def visualize(self):
        pass

    def reset(self):
        pass
