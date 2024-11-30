from project.model import Model, DecodableModel
import numpy as np
from utils import Config
from scipy.fftpack import fftn, ifftn
from tqdm.notebook import tqdm


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
        return np.dot(HUL_test_compl, self.coeff)  # zDL

    def decode(self, zDL: np.ndarray) -> np.ndarray:
        # Go from zdl to csi space
        HDL_reconst_tmp = np.matmul(zDL, self.coeff_trunc.conj().T)
        HDL_reconst = HDL_reconst_tmp + self.HUL_train_compl_tmp_mean
        HDL_ori_reconst = HDL_reconst.T.reshape(self.cfg.num_tx_antennas, self.cfg.num_subcarriers, len(zDL), order='F')
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

            pass

    # TODO: Finish implementation
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


