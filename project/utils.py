import os
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import pickle
from typing import Optional, Literal


@dataclass(frozen=True)
class Config:
    num_rx_antennas: int
    num_tx_antennas: int  # na
    num_subcarriers: int  # nc
    # Additional options/configurations...
    train_test_split: float
    data_root: str
    results_save_path: str
    duplicate_data: int = 5
    train_snr: float = 10 #10dB
    test_snr: float = 10 #10dB

    # For saving and loading models
    results_save_path: Optional[str] = None
    model_root: str = "../models"
    pca_model_name: str = "pca"
    predictor_model_name: str = "predictor"
    kmeans_model_name: str = "kmeans"
    retrain_all: bool = True


    # PCA Config
    reduce_pca_overhead: bool = True
    max_pca_coeffs: int = 500        # C
    compression_ratio: int = 16      # CR

    # Predictor Config
    null_predictor: bool = False      # If True, disable the CSI predictor, essentially falling back to reference model
    predictor_window_size: int = 5
    epochs: int = 10

    # KMeans/Compressor Config
    total_bits: int = 512        # BTot
    compressor_type: Literal["kmeans", "dct", "dft"] = "kmeans"
    preprocessor_type: Literal["real_imag", "amplitude_angle"] = "real_imag"
    normalization_type: Literal["mean_std", "max"] = "mean_std"

    #DCT compression
    float_bits: int = 6
    compression_rate_dct: float = 1

    trunc_lstm_pred: int = 20

    def __post_init__(self):
        if(not os.path.isdir(self.results_save_path)):
            os.makedirs(self.results_save_path)

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


def func_rho(h_hat, h):
    """
    Calculate the correlation coefficient between the estimated channel (h_hat)
    and the actual channel (h).

    Parameters:
    h_hat (numpy.ndarray): Estimated channel matrix of shape (n, k).
    h (numpy.ndarray): Actual channel matrix of shape (n, k).

    Returns:
    float: Average correlation coefficient rho_h.
    """
    rho_i = 0
    n, k = h.shape

    for i in range(k):
        # Compute the correlation for each column
        numerator = abs(np.dot(h_hat[:, i].conj().T, h[:, i]))
        denominator = np.linalg.norm(h_hat[:, i]) * np.linalg.norm(h[:, i])
        rho_i += numerator / denominator

    # Average the correlation coefficients
    rho_h = rho_i / k
    return rho_h


def func_gram_schmidt(V):
    """
    Perform Gram-Schmidt orthogonalization on the columns of V.
    The input matrix V (n x k) is replaced by an orthonormal matrix U (n x k)
    whose columns span the same subspace as V.

    Parameters:
    V (numpy.ndarray): Input matrix of shape (n, k).

    Returns:
    U (numpy.ndarray): Orthonormal matrix of shape (n, k).
    """
    n, k = V.shape
    U = np.zeros((n, k), dtype=V.dtype)

    # Normalize the first column
    U[:, 0] = V[:, 0] / np.linalg.norm(V[:, 0])

    # Iterate through remaining columns
    for i in range(1, k):
        U[:, i] = V[:, i]
        for j in range(i):
            projection = np.dot(U[:, j].conj().T, U[:, i]) / (np.linalg.norm(U[:, j])**2)
            U[:, i] -= projection * U[:, j]
        U[:, i] /= np.linalg.norm(U[:, i])

    return U


def func_nmse(h_hat, h):
    """
    Calculate the Normalized Mean Squared Error (NMSE) between the estimated
    channel (h_hat) and the actual channel (h).

    Parameters:
    h_hat (numpy.ndarray): Estimated channel matrix.
    h (numpy.ndarray): Actual channel matrix.

    Returns:
    float: NMSE value.
    """
    nmse_h = (np.linalg.norm(h_hat - h, 'fro') / np.linalg.norm(h, 'fro'))**2
    return nmse_h


def reference_nmse_rho_test(name, HDL_test, HDL_ori_reconst, save_path: Optional[str]=None, btot=512, show_plot=True):
    # Assessing performance
    print("Assessing performance...")

    nTest = len(HDL_test)
    nmse = np.zeros(nTest)
    rho = np.zeros(nTest)

    for i in range(nTest):
        ch = HDL_test[i]
        ch_h = HDL_ori_reconst[i]
        nmse[i] = func_nmse(ch_h, ch)  # Assumes func_nmse is defined
        rho[i] = func_rho(ch_h, ch)    # Assumes func_rho is defined

    # Plotting results
    print("Plotting results...")
    LineW = 1.5

    cdf_nmse = np.sort(10 * np.log10(nmse))
    cdf_rho = np.sort(10 * np.log10(1 - rho))

    probabilities = np.arange(1, len(cdf_nmse) + 1) / len(cdf_nmse)

    if show_plot:
        plt.figure()
        plt.plot(cdf_nmse, probabilities, label='CDF 10log(NMSE)', linewidth=LineW)
        plt.plot(cdf_rho, probabilities, label='CDF 10log(1-RHO)', linewidth=LineW)

        # plt.xlim([-22, 0])
        # plt.xticks(range(-22, 1, 2))
        plt.xlabel('Metric (dB)')
        plt.ylabel('CDF')
        plt.legend(loc='lower right')
        plt.title(f"{name}_Btot_{btot}")
        plt.grid(True)
        plt.show()

    if save_path is not None:
        np.save(os.path.join(save_path, f'nmse-{name}_Btot_{btot}.npy'), nmse)
        np.save(os.path.join(save_path, f'rho-{name}_Btot_{btot}.npy'), rho)

    return nmse


def plot_single_zdl(zdl, pca):
    plt.figure()
    zdl = np.expand_dims(zdl, axis=0)
    recovered = pca.decode(zdl)
    plt.imshow(np.squeeze(np.abs(recovered)))


def add_noise(H,snr):
    """

    :param H:
    :param snr: in linear
    :return:
    """
    if(snr==-1):
        return H

    H_power = np.mean(np.abs(H)**2,axis=(1,2)).reshape(H.shape[0],-1)
    N_power = H_power/snr

    N = (np.random.randn(*H.shape) + 1j*np.random.randn(*H.shape))
    for i in range(N.shape[0]):
        N[i] = N[i]*np.sqrt(N_power[i]/2)

    return H+N
