import numpy as np
from abc import ABC, abstractmethod
from utils import Config
from model import DecodableModel
#info:
class DFTCompression(DecodableModel):
    """
    DFT-based compression technique with true compression rate.
    """

    def __init__(self, cfg: Config, matlab=None):
        """
        Initialize DFT Compression model.

        Parameters:
        -----------
        cfg : Config
            Configuration object containing compression parameters
        matlab : optional
            Matlab engine (kept for consistency with existing interface)
        """
        self.cfg = cfg
        self.matlab = matlab
        self.num_original_coeffs = None
        self.num_coeffs = None
        self.compression_rate = getattr(cfg, 'compression_rate', 0.5)

    def fit(self, zUL_train: np.ndarray):
        """
        Prepare the compression model based on training data.

        Parameters:
        -----------
        zUL_train : np.ndarray
            Training data for compression model preparation
        """
        # Store original number of coefficients
        self.num_original_coeffs = zUL_train.shape[1]

        # Calculate the number of coefficients to retain
        retention_rate = 1 - self.compression_rate
        self.num_coeffs = int(self.num_original_coeffs * retention_rate)

        # Optional: Any additional preprocessing or learning from training data
        print(f"DFT Compression: Preparing to compress to {self.num_coeffs} coefficients")

    def process(self, zDL: np.ndarray) -> np.ndarray:
        """
        Compress the input vector using DFT.

        Parameters:
        -----------
        zDL : np.ndarray
            Input data to be compressed

        Returns:
        --------
        np.ndarray
            Compressed data
        """
        # Limit to original number of coefficients if needed
        zDL = zDL[:, :self.num_original_coeffs]

        compressed_data = []
        for vector in zDL:
            # Compute DFT
            dft_coeffs = np.fft.fft(vector)

            # Retain top coefficients by magnitude
            indices = np.argsort(-np.abs(dft_coeffs))[:self.num_coeffs]

            # Create a compressed representation
            compressed_vector = np.zeros_like(dft_coeffs)
            compressed_vector[indices] = dft_coeffs[indices]

            compressed_data.append(compressed_vector)

        return np.array(compressed_data)

    def decode(self, quantized_zDL: np.ndarray) -> np.ndarray:
        """
        Reconstruct the original vector from compressed representation.

        Parameters:
        -----------
        quantized_zDL : np.ndarray
            Compressed data to be reconstructed

        Returns:
        --------
        np.ndarray
            Reconstructed data padded to original dimensions
        """
        # Reconstruct each vector using inverse DFT
        reconstructed_data = []
        for compressed_vector in quantized_zDL:
            # Reconstruct using inverse DFT
            reconstructed_vector = np.fft.ifft(compressed_vector).real
            reconstructed_data.append(reconstructed_vector)

        padded_zDL= np.array(reconstructed_data)

        return padded_zDL

    def load(self, path):
        """
        Load a pre-trained compression model.

        Parameters:
        -----------
        path : str
            Path to the saved model
        """
        # Implement model loading if needed
        pass

    def save(self, path):
        """
        Save the current compression model.

        Parameters:
        -----------
        path : str
            Path to save the model
        """
        # Implement model saving if needed
        pass