from model import Model, DecodableModel
import numpy as np
from utils import Config
from scipy.fftpack import fftn, ifftn
from tqdm.notebook import tqdm
from dataset import Dataset
from DCT_compression import DCTCompression
class FullReferenceModel(DecodableModel):
    def __init__(self, cfg: Config, matlab):
        self.cfg = cfg
        self.matlab = matlab

        self.pca = ReferencePCA(cfg, matlab)
        self.predictor = NullPredictor()
        # With NullPredictor, prediction_error is just zDL! This lets us test the ref impl
        self.error_compressor = ReferenceKmeans(cfg, matlab)

    def fit(self, dataset: Dataset):
        self.pca.fit(dataset.csi_samples)
        zdl_train = self.pca.process(dataset.csi_samples)                # N * zdl_len
        zdl_train_windows = self._compute_pca_for_windows(dataset)
        self.predictor.fit(zdl_train, zdl_train_windows)    # Doesn't do anything, just null predictor
        predicted_zdl = self.predictor.process(zdl_train_windows)

        prediction_error = zdl_train - predicted_zdl
        self.error_compressor.fit(prediction_error)

    def process(self, dataset: Dataset) -> np.ndarray:
        zdl_train = self.pca.process(dataset.csi_samples)                # N * zdl_len
        zdl_train_windows = self._compute_pca_for_windows(dataset)
        predicted_zdl = self.predictor.process(zdl_train_windows)
        prediction_error = zdl_train - predicted_zdl
        compressed_error = self.error_compressor.process(prediction_error)
        return compressed_error, zdl_train_windows

    def decode(self, compressed_error: np.ndarray, zdl_test_windows: np.ndarray):
        ul_pred_error = self.error_compressor.decode(compressed_error)
        ul_pred_zdl = self.predictor.process(zdl_test_windows)
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


# TODO: Implement Reference
class ReferencePCA(DecodableModel):
    def __init__(self, cfg: Config, matlab):
        self.matlab = matlab
        self.cfg = cfg

    def fit(self, csis: np.ndarray):
        H_train = np.transpose(csis, (1, 2, 0))
        HUL_train_compl = self._get_compl(H_train, train_mode=True)
        self.coeff_ori = np.array(self.matlab.pca(HUL_train_compl))
        if self.cfg.reduce_pca_overhead:
            self.reduce_pca_overhead()
        else:
            self.coeff = self.coeff_ori
        self.coeff_trunc = self.coeff[:, :self.cfg.max_pca_coeffs]

    def process(self, csis: np.ndarray, train_mode=False) -> np.ndarray:
        H_train = np.transpose(csis, (1, 2, 0))
        HUL_test_compl = self._get_compl(H_train, train_mode=train_mode)
        return np.dot(HUL_test_compl, self.coeff_trunc)  # zDL

    def decode(self, zDL: np.ndarray) -> np.ndarray:
        # Go from zdl to csi space
        HDL_reconst_tmp = np.matmul(zDL, self.coeff_trunc.conj().T)
        HDL_reconst = HDL_reconst_tmp + self.HUL_train_compl_tmp_mean
        HDL_ori_reconst = HDL_reconst.T.reshape(self.cfg.num_tx_antennas, self.cfg.num_subcarriers, len(zDL), order='F')
        HDL_ori_reconst  = np.transpose(HDL_ori_reconst, (2, 0, 1))
        return HDL_ori_reconst

    def load(self, path):
        pass

    def save(self, path):
        pass

    def _get_compl(self, HUL_train_n, train_mode=False):
        # TODO naming
        Lambda = 1.0 / np.mean(np.abs(HUL_train_n)**2, axis=(0, 1))
        HUL_train_n *= np.sqrt(Lambda[np.newaxis, np.newaxis, :])
        HUL_train_compl_tmp = HUL_train_n.reshape(
            self.cfg.num_tx_antennas * self.cfg.num_subcarriers, HUL_train_n.shape[-1], order='F'
        ).T
        if train_mode:
            self.HUL_train_compl_tmp_mean = np.mean(HUL_train_compl_tmp, axis=0)
        HUL_train_compl = HUL_train_compl_tmp - self.HUL_train_compl_tmp_mean
        return HUL_train_compl


    def reduce_pca_overhead(self):
        print("Reducing offloading overhead...")
        self.coeff = np.zeros_like(self.coeff_ori)
        for i in range(self.coeff_ori.shape[1]):
            # Reshape the i-th principal component
            pc_shape = (
                int(np.sqrt(self.cfg.num_tx_antennas)),
                int(np.sqrt(self.cfg.num_tx_antennas)),
                self.cfg.num_subcarriers
            )
            pc = self.coeff_ori[:, i].reshape(pc_shape)

            # Perform FFT
            pcDFT = fftn(pc).flatten()

            # Create mask for top coefficients
            mask = np.zeros_like(pcDFT)
            locs = np.argsort(-np.abs(pcDFT))[
                   :int(self.cfg.num_tx_antennas * self.cfg.num_subcarriers / self.cfg.compression_ratio)
            ]  # Find top |na*nc/CR| values
            mask[locs] = 1

            # Apply the mask
            pcDFT = pcDFT * mask

            # Perform inverse FFT
            pcIFFT = ifftn(pcDFT.reshape(pc_shape))
            self.coeff[:, i] = pcIFFT.flatten()

        # Orthogonalize using Gram-Schmidt
        self.coeff = np.array(
            self.matlab.func_gram_schmidt(self.coeff[:, :500])
        )  # TODO Py gram_schmidt not working, matlab.func_gram_schmidt is working


# TODO: Implement Reference
class ReferenceKmeans(DecodableModel):
    def __init__(self, cfg: Config, matlab):
        self.cfg = cfg
        self.matlab = matlab

    def fit(self, zUL_train: np.ndarray):
        self.matlab.rng(69)
        self.matlab.warning('off', 'stats:kmeans:FailedToConverge')

        num_train = len(zUL_train)
        self.num_original_coeffs = zUL_train.shape[1]
        print("Training k-means clustering...")

        # Project data using PCA
        zUL_train_entries = np.stack((np.real(zUL_train), np.imag(zUL_train)), axis=-1)

        # Calculate importances (variance of each PCA component)
        importances = np.var(zUL_train, axis=0)

        # Allocate bits for quantization
        Bs = np.squeeze(np.array(
            self.matlab.func_allocate_bits(self.cfg.total_bits, importances, zUL_train_entries),
            dtype=np.int16
        ))
        self.num_coeffs = len(Bs)

        self.allocated_bits = Bs
        # Scale data for k-means clustering
        zUL_train_entries_scaled = np.zeros((num_train, self.num_coeffs, 2))
        for i in range(self.num_coeffs):
            zUL_train_entries_scaled[:, i, :] = zUL_train_entries[:, i, :] / np.sqrt(importances[i])

        # Determine the number of samples for k-means
        nTrainKMeans = min(num_train, round(1e5 / self.num_coeffs))
        zUL_train_entriesCSCG = zUL_train_entries_scaled[:nTrainKMeans, :self.num_coeffs, :].reshape(-1, 2, order='F')

        # Train k-means for different bit levels
        quantLevelsCSCG = [None] * Bs[0]
        for i in tqdm(range(1, Bs[0] + 1)):
            quantLevelsCSCG[i-1] = np.array(
                self.matlab.kmeans(zUL_train_entriesCSCG,2**i, nargout=2)[1]   # Get cluster centers
            )
            # kmeans = KMeans(n_clusters=2**i, n_init=10)
            # kmeans.fit(zUL_train_entriesCSCG)
            # quantLevelsCSCG[i - 1] = kmeans.cluster_centers_

        # Scale quantization levels back
        self.quantLevels = [None] * self.num_coeffs
        for i in range(self.num_coeffs):
            self.quantLevels[i] = quantLevelsCSCG[Bs[i] - 1] * np.sqrt(importances[i])

    def process(self, zDL: np.ndarray) -> np.ndarray:
        zDL = zDL[:, :self.num_coeffs]
        quantized_zdl = np.zeros_like(zDL)
        quantLevels = self.quantLevels
        for i in range(zDL.shape[0]):
            for j in range(zDL.shape[1]):
                # Find the closest quantization level
                distances = np.abs(zDL[i, j] - (quantLevels[j][:, 0] + 1j * quantLevels[j][:, 1]))
                vecIdx = np.argmin(distances)
                quantized_zdl[i, j] = quantLevels[j][vecIdx, 0] + 1j * quantLevels[j][vecIdx, 1]
        return quantized_zdl

    def decode(self, quantized_zDL: np.ndarray) -> np.ndarray:
        padded_zDL = np.zeros((len(quantized_zDL), self.num_original_coeffs), dtype=quantized_zDL.dtype)
        padded_zDL[:, :self.num_coeffs] = quantized_zDL
        return padded_zDL

    # def decode(self, error: np.ndarray) -> np.ndarray:
    #     for i in range(zDL.shape[0]):
    #         for j in range(zDL.shape[1]):
    #             # Find the closest quantization level
    #             distances = np.abs(zDL[i, j] - (quantLevels[j][:, 0] + 1j * quantLevels[j][:, 1]))
    #             vecIdx = np.argmin(distances)
    #             zDL[i, j] = quantLevels[j][vecIdx, 0] + 1j * quantLevels[j][vecIdx, 1]

    def load(self, path):
        pass

    def save(self, path):
        pass


class NullPredictor(Model):
    def __init__(self):
        pass

    def fit(self, csis: np.ndarray, windows: np.ndarray):
        pass

    def process(self, windows: np.ndarray) -> np.ndarray:
        """
        :param windows: N x window_size x na x nc
        :return:        N x na x nc
        """
        new_shape = (windows.shape[0], *windows.shape[2:])
        return np.zeros(new_shape, dtype=windows.dtype)

    def load(self, path):
        pass

    def save(self, path):
        pass

class NewPredictor(Model):
    def __init__(self):
        pass

    def fit(self, csis: np.ndarray, windows: np.ndarray):
        pass

    def process(self, windows: np.ndarray) -> np.ndarray:
        """
        :param windows: N x window_size x na x nc
        :return:        N x na x nc
        """
        new_shape = (windows.shape[0], *windows.shape[2:])
        return np.zeros(new_shape, dtype=windows.dtype)

    def load(self, path):
        pass

    def save(self, path):
        pass


