import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    num_rx_antennas: int
    num_tx_antennas: int
    num_subcarriers: int
    # Additional options/configurations...
    train_test_split: float
    data_root: str

    # For saving and loading models
    model_root: str = "../models"
    pca_model_name: str = "pca"
    predictor_model_name: str = "predictor"
    kmeans_model_name: str = "kmeans"
    retrain_all: bool = True

    reduce_pca_overhead = True
    null_predictor = False      # If True, disable the CSI predictor, essentially falling back to reference model
    predictor_window_size = 5

    @property
    def data_path(self):
        return os.path.join(
            self.data_root,
            f"batch1_1Lane_450U_{self.num_rx_antennas}Rx_{self.num_tx_antennas}Tx_1T_{self.num_subcarriers}K_2620000000.0fc.pickle"
        )

    @property
    def pca_path(self):
        return os.path.join(self.model_root, self.pca_model_name)

    @property
    def predictor_path(self):
        return os.path.join(self.model_root, self.predictor_model_name)

    @property
    def kmeans_path(self):
        return os.path.join(self.model_root, self.kmeans_model_name)

