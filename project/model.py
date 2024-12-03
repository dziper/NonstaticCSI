from abc import ABC, abstractmethod
import numpy as np
import os


class Model(ABC):
    # Model interface for different predictors - basically anything that needs to be trained and then can predict stuff
    @abstractmethod
    def __init__(self):  # Instantiate Model hyperparameters
        pass

    @abstractmethod
    def fit(self, *args):
        pass

    @abstractmethod
    def process(self, *args) -> np.ndarray:
        pass

    @abstractmethod
    def load(self, path):
        pass

    @abstractmethod
    def save(self, path):
        pass


class DecodableModel(Model):
    @abstractmethod
    def decode(self, *args):
        pass


# Train PCA
def train_or_load(model: Model, path: str, retrain_all: bool, *args, **kwargs):
    if not retrain_all and os.path.exists(path):
        model.load(path)
    else:
        model.fit(*args, **kwargs)
        model.save(path)
