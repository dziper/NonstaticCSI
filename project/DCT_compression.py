import numpy as np
from scipy.fftpack import dct, idct
from utils import Config
from model import DecodableModel

class DCTCompression(DecodableModel):
    """
    DCT-based compression technique with true compression rate.
    """

    def __init__(self, cfg: Config, matlab=None):
        """
        Initialize DCT Compression model.

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
        self.compression_rate = cfg.compression_rate_dct

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

        print(f"DCT Compression: Preparing to compress to {self.num_coeffs} coefficients")

    def process(self, zDL: np.ndarray) -> np.ndarray:
        """
        Compress the input vector using DCT.

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
            # Separate real and imaginary parts
            real_part = vector.real
            imag_part = vector.imag

            # Apply DCT to real and imaginary parts
            real_dct = dct(real_part, norm='ortho')
            imag_dct = dct(imag_part, norm='ortho')

            # Retain top coefficients by magnitude for real and imaginary parts
            real_indices = np.argsort(-np.abs(real_dct))[:self.num_coeffs]
            imag_indices = np.argsort(-np.abs(imag_dct))[:self.num_coeffs]

            # Create a compressed representation
            compressed_real = np.zeros_like(real_dct)
            compressed_imag = np.zeros_like(imag_dct)
            compressed_real[real_indices] = real_dct[real_indices]
            compressed_imag[imag_indices] = imag_dct[imag_indices]

            # Combine compressed real and imaginary parts
            compressed_vector = compressed_real + 1j * compressed_imag
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
        # Reconstruct each vector
        reconstructed_data = []
        for compressed_vector in quantized_zDL:
            # Separate real and imaginary parts
            real_part = compressed_vector.real
            imag_part = compressed_vector.imag

            # Reconstruct using inverse DCT
            real_reconstructed = idct(real_part, norm='ortho')
            imag_reconstructed = idct(imag_part, norm='ortho')

            # Combine reconstructed real and imaginary parts
            reconstructed_vector = real_reconstructed + 1j * imag_reconstructed
            reconstructed_data.append(reconstructed_vector)

        # Pad to original number of coefficients
        padded_zDL = np.zeros((len(reconstructed_data), self.num_original_coeffs),
                              dtype=reconstructed_data[0].dtype)
        padded_zDL = np.array(reconstructed_data)

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