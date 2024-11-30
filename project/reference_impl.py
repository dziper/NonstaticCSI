from project.model import Model, DecodableModel
import numpy as np


# TODO: Implement Reference
class ReferencePCA(DecodableModel):
    def __init__(self):
        pass

    def fit(self, data: np.ndarray):
        pass

    def process(self, csis: np.ndarray) -> np.ndarray:
        # return zdls
        pass

    def decode(self, zdl: np.ndarray) -> np.ndarray:
        # Go from zdl to csi space
        pass

    def load(self, path):
        pass

    def save(self, path):
        pass


# TODO: Implement Reference
class ReferenceKmeans(DecodableModel):
    def __init__(self):
        pass

    def fit(self, data: np.ndarray):
        pass

    def process(self, data: np.ndarray) -> np.ndarray:
        pass

    def decode(self, error: np.ndarray) -> np.ndarray:
        pass

    def load(self, path):
        pass

    def save(self, path):
        pass


class NullPredictor(Model):
    def __init__(self):
        pass

    def fit(self, data: np.ndarray):
        pass

    def process(self, data: np.ndarray) -> np.ndarray:
        return np.zeros_like(data)

    def load(self, path):
        pass

    def save(self, path):
        pass


