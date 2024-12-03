import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from scipy.fftpack import dct, idct, fft, ifft
from sklearn.decomposition import PCA
import reference_impl as ref
import utils
import dataset

def reshape_tensor(matrix, K):
    """
    Reshapes the input tensor by keeping the first K-1 dimensions intact
    and combining the last dimensions (from K to N) into one dimension.

    Args:
    - matrix (numpy.ndarray): The input tensor to reshape.
    - K (int): The dimension where the reshaping starts. Dimensions from K to N will be combined.

    Returns:
    - numpy.ndarray: The reshaped tensor.
    """
    # Get the shape of the input tensor
    shape = matrix.shape

    # Check if K is a valid dimension
    if K > len(shape) or K < 1:
        raise ValueError("K must be between 1 and the number of dimensions of the matrix.")

    # Split the shape into two parts: before K and from K onwards
    shape_before_K = shape[:K]
    shape_after_K = shape[K:]

    # Calculate the product of all dimensions after K
    new_shape_after_K = np.prod(shape_after_K)

    # Combine the first K-1 dimensions with the new flattened dimension
    new_shape = shape_before_K + (new_shape_after_K,)

    # Reshape the matrix
    reshaped_matrix = matrix.reshape(new_shape)

    return reshaped_matrix
class CompressionTechnique(ABC):
    """
    Abstract base class for compression techniques
    """

    @abstractmethod
    def compress(self, vector, compression_rate):
        """
        Compress the input vector

        Parameters:
        -----------
        vector : np.ndarray
            Input complex vector
        compression_rate : float
            Fraction of coefficients to keep

        Returns:
        --------
        compressed_data : tuple
            Compressed representation and additional metadata
        """
        pass

    @abstractmethod
    def reconstruct(self, compressed_data, original_length):
        """
        Reconstruct the vector from compressed representation

        Parameters:
        -----------
        compressed_data : tuple
            Compressed representation and metadata
        original_length : int
            Length of the original vector

        Returns:
        --------
        reconstructed_vector : np.ndarray
            Reconstructed complex vector
        """
        pass
class DCTCompression(CompressionTechnique):
    """
    DCT-based compression technique with true compression rate.
    """

    def compress(self, vector, compression_rate):
        """
        Compress a vector using DCT by retaining a fraction of its coefficients.

        Parameters:
        ---------
        vector : np.ndarray
            Input complex vector
        compression_rate : float
            Fraction of data compressed (e.g., 0.2 means 80% of coefficients retained)

        Returns:
        --------
        compressed_data : tuple
            Compressed DCT coefficients and indices of retained coefficients
        """
        # Compute retention rate
        retention_rate = 1 - compression_rate

        # Apply DCT
        real_dct = dct(vector.real, norm='ortho')
        imag_dct = dct(vector.imag, norm='ortho')

        # Retain coefficients based on retention rate
        num_coeffs = int(len(real_dct) * retention_rate)
        real_indices = np.argsort(-np.abs(real_dct))[:num_coeffs]
        imag_indices = np.argsort(-np.abs(imag_dct))[:num_coeffs]

        # Store the retained coefficients and their indices
        compressed_data = (
            (real_indices, real_dct[real_indices]),
            (imag_indices, imag_dct[imag_indices])
        )
        return compressed_data

    def reconstruct(self, compressed_data, original_length):
        """
        Reconstruct a vector from DCT coefficients.

        Parameters:
        ---------
        compressed_data : tuple
            Compressed DCT coefficients and indices of retained coefficients
        original_length : int
            Length of the original vector

        Returns:
        --------
        reconstructed_vector : np.ndarray
            Reconstructed vector
        """
        (real_indices, real_values), (imag_indices, imag_values) = compressed_data

        # Recreate the full DCT coefficient arrays
        real_dct = np.zeros(original_length)
        imag_dct = np.zeros(original_length)
        real_dct[real_indices] = real_values
        imag_dct[imag_indices] = imag_values

        # Apply the inverse DCT to reconstruct the vector
        real_reconstructed = idct(real_dct, norm='ortho')
        imag_reconstructed = idct(imag_dct, norm='ortho')

        return real_reconstructed + 1j * imag_reconstructed
class DFTCompression(CompressionTechnique):
    """
    DFT-based compression technique with true compression rate.
    """

    def compress(self, vector, compression_rate):
        """
        Compress a vector using DFT by retaining a fraction of its coefficients.

        Parameters:
        ---------
        vector : np.ndarray
            Input complex vector
        compression_rate : float
            Fraction of data compressed (e.g., 0.2 means 80% of coefficients retained)

        Returns:
        --------
        compressed_data : tuple
            Compressed DFT coefficients and indices of retained coefficients
        """
        # Compute retention rate
        retention_rate = 1 - compression_rate

        # Apply DFT
        dft_coeffs = np.fft.fft(vector)

        # Retain coefficients based on retention rate
        num_coeffs = int(len(dft_coeffs) * retention_rate)
        indices = np.argsort(-np.abs(dft_coeffs))[:num_coeffs]

        # Store the retained coefficients and their indices
        compressed_data = (indices, dft_coeffs[indices])
        return compressed_data

    def reconstruct(self, compressed_data, size):
        """
        Reconstruct the vector from compressed DFT coefficients.

        Parameters:
        ---------
        compressed_data : tuple
            Compressed DFT coefficients and their indices
        size : int
            Original vector size

        Returns:
        --------
        np.ndarray
            Reconstructed vector
        """
        indices, coeffs = compressed_data
        dft_reconstructed = np.zeros(size, dtype=complex)
        dft_reconstructed[indices] = coeffs

        # Apply inverse DFT
        return np.fft.ifft(dft_reconstructed)
class VariableLengthEncodingCompression(CompressionTechnique):
    """
    Variable-length encoding for complex numbers with true compression rate.
    """

    def compress(self, vector, compression_rate):
        """
        Compress a vector using variable-length encoding of significant values.

        Parameters:
        ---------
        vector : np.ndarray
            Input complex vector
        compression_rate : float
            Fraction of data compressed (e.g., 0.2 means 80% of significant values retained)

        Returns:
        --------
        compressed_data : tuple
            Encoded significant values and their indices
        """
        # Compute retention rate
        retention_rate = 1 - compression_rate

        # Determine magnitude of coefficients
        magnitudes = np.abs(vector)

        # Retain based on retention rate
        num_coeffs = int(len(vector) * retention_rate)
        indices = np.argsort(-magnitudes)[:num_coeffs]
        retained_values = vector[indices]

        # Encode retained values
        encoded_values = [(val.real, val.imag) for val in retained_values]

        return indices, encoded_values

    def reconstruct(self, compressed_data, size):
        """
        Reconstruct the vector from compressed data.

        Parameters:
        ---------
        compressed_data : tuple
            Encoded significant values and their indices
        size : int
            Original vector size

        Returns:
        --------
        np.ndarray
            Reconstructed vector
        """
        indices, encoded_values = compressed_data
        reconstructed = np.zeros(size, dtype=complex)
        reconstructed[indices] = [complex(real, imag) for real, imag in encoded_values]
        return reconstructed
class CompressionAnalyzer:
    """
    Performs compression analysis and comparison
    """

    def __init__(self, techniques):
        """
        Initialize with compression techniques

        Parameters:
        -----------
        techniques : dict
            Dictionary of compression techniques
            {technique_name: compression_technique_instance}
        """
        self.techniques = techniques

    def normalized_mse(self, original, reconstructed):
        """
        Calculate Normalized Mean Squared Error (NMSE)
        """
        # Calculate normalized squared error for each sample
        nse_samples = np.abs(original - reconstructed) ** 2 / np.abs(original) ** 2

        # Take mean across all samples
        return np.mean(nse_samples)

    def monte_carlo_compression(self, M, N, N0_values, compression_rates):
        """
        Perform Monte Carlo simulation of compression techniques
        """
        results = {}

        for N0 in N0_values:
            results[N0] = {}

            for technique_name, technique in self.techniques.items():
                nmse_values = {cr: [] for cr in compression_rates}

                for _ in range(N):
                    # Generate complex Gaussian vector
                    original_vector = np.sqrt(1 / 2) * (
                            np.random.normal(0, np.sqrt(N0 / 2), M) +
                            1j * np.random.normal(0, np.sqrt(N0 / 2), M)
                    )

                    for cr in compression_rates:
                        # Compress and reconstruct
                        compressed_data = technique.compress(original_vector, cr)
                        reconstructed = technique.reconstruct(compressed_data, M)

                        # Calculate NMSE
                        nmse = self.normalized_mse(original_vector, reconstructed)
                        nmse_values[cr].append(nmse)

                # Average NMSE for this technique
                results[N0][technique_name] = {
                    cr: np.mean(nmse_values[cr]) for cr in compression_rates
                }

        return results

    def plot_compression_performance(self, results):
        """
        Plot NMSE vs Compression Rate for different techniques and noise variances
        """
        plt.figure(figsize=(12, 8))

        # Create subplot for each noise variance
        num_N0 = len(results)
        fig, axs = plt.subplots(1, num_N0, figsize=(5 * num_N0, 5), sharey=True)

        # Ensure axs is always a list, even if there's only one subplot
        if num_N0 == 1:
            axs = [axs]

        for i, (N0, technique_results) in enumerate(results.items()):
            for technique_name, nmse_dict in technique_results.items():
                compression_rates = list(nmse_dict.keys())
                nmse_values = list(nmse_dict.values())

                axs[i].plot(compression_rates, nmse_values,
                            marker='o', label=technique_name)

            axs[i].set_title(f'N0 = {N0}')
            axs[i].set_xlabel('Compression Rate')
            axs[i].set_xscale('linear')
            axs[i].set_yscale('log')
            axs[i].grid(True)
            axs[i].legend()

        axs[0].set_ylabel('Normalized MSE')
        plt.tight_layout()
        plt.show()
class PCACompression(CompressionTechnique):
    """
    PCA-based compression technique
    """

    def __init__(self):
        self.pca_model = None

    def compress(self, vectors, compression_rate):
        """
        Compress a set of vectors using PCA

        Parameters:
        -----------
        vectors : np.ndarray
            Matrix of shape (n_samples, n_features) containing all samples
        compression_rate : float
            Fraction of principal components to retain

        Returns:
        --------
        compressed_data : tuple
            Compressed representation (PCA projections and model)
        """
        n_components = int(vectors.shape[1] * (1 - compression_rate))
        self.pca_model = PCA(n_components=n_components)
        compressed_vectors = self.pca_model.fit_transform(vectors)
        return compressed_vectors, self.pca_model

    def reconstruct(self, compressed_data, original_length):
        """
        Reconstruct vectors from PCA-compressed representation

        Parameters:
        -----------
        compressed_data : tuple
            Compressed PCA projections and model
        original_length : int
            Not used, as PCA reconstructs based on its fitted components

        Returns:
        --------
        reconstructed_vectors : np.ndarray
            Reconstructed set of vectors
        """
        compressed_vectors, pca_model = compressed_data
        return pca_model.inverse_transform(compressed_vectors)
class CompressionAnalyzerUpdated(CompressionAnalyzer):
    """
    Extended analyzer to support Monte Carlo and single-run analysis.
    """

    def single_run_compression(self, matrix, compression_rates):
        """
        Perform a single compression and reconstruction analysis on a matrix.

        Parameters:
        -----------
        matrix : np.ndarray
            Input matrix of shape (N_samples, N_features), where each row is a sample.
        compression_rates : list
            List of compression rates to test.

        Returns:
        --------
        results : dict
            NMSE results for each compression technique and rate averaged over samples.
        """
        n_samples, n_features = matrix.shape
        results = {}

        for technique_name, technique in self.techniques.items():
            nmse_values = {cr: [] for cr in compression_rates}
            for cr in compression_rates:
                for row in matrix:
                    # Compress and reconstruct each row
                    compressed_data = technique.compress(row, cr)
                    reconstructed = technique.reconstruct(compressed_data, n_features)

                    # Calculate NMSE for the row
                    nmse = self.normalized_mse(row, reconstructed)
                    nmse_values[cr].append(nmse)

            # Average NMSE over all samples for this technique and rate
            results[technique_name] = {cr: np.mean(nmse_values[cr]) for cr in compression_rates}

        return results

    def monte_carlo_compression(self, M, N, N0_values, compression_rates):
        """
        Perform Monte Carlo simulation of compression techniques with shared samples.

        Parameters:
        -----------
        M : int
            Length of the vectors.
        N : int
            Number of vectors to simulate.
        N0_values : list
            List of noise power levels.
        compression_rates : list
            List of compression rates to test.

        Returns:
        --------
        results : dict
            NMSE results for each noise level, technique, and compression rate.
        """
        results = {}

        for N0 in N0_values:
            results[N0] = {}

            # Generate shared data for all techniques
            original_vectors = np.sqrt(1 / 2) * (
                np.random.normal(0, np.sqrt(N0 / 2), (N, M)) +
                1j * np.random.normal(0, np.sqrt(N0 / 2), (N, M))
            )

            for technique_name, technique in self.techniques.items():
                nmse_values = {cr: [] for cr in compression_rates}

                for cr in compression_rates:
                    for vector in original_vectors:
                        # Compress and reconstruct
                        compressed_data = technique.compress(vector, cr)
                        reconstructed = technique.reconstruct(compressed_data, M)

                        # Calculate NMSE
                        nmse = self.normalized_mse(vector, reconstructed)
                        nmse_values[cr].append(nmse)

                # Average NMSE for this technique
                results[N0][technique_name] = {
                    cr: np.mean(nmse_values[cr]) for cr in compression_rates
                }

        return results

    def analyze_and_plot(self, mode, *args, **kwargs):
        """
        Analyze data and plot results for a single run or Monte Carlo.

        Parameters:
        -----------
        mode : str
            Analysis mode, 'single' or 'montecarlo'.
        *args, **kwargs :
            Arguments for the respective analysis method.

        Returns:
        --------
        results : dict
            Analysis results.
        """
        if mode == 'single':
            vector = kwargs.get('vector')
            compression_rates = kwargs.get('compression_rates')
            results = self.single_run_compression(vector, compression_rates)
            self.plot_single_run_results(results)
        elif mode == 'montecarlo':
            results = self.monte_carlo_compression(*args, **kwargs)
            self.plot_compression_performance(results)
        else:
            raise ValueError("Invalid mode. Choose 'single' or 'montecarlo'.")
        return results

    def plot_single_run_results(self, results):
        """
        Plot NMSE vs Compression Rate for a single run.

        Parameters:
        -----------
        results : dict
            NMSE results for each technique and compression rate.
        """
        plt.figure(figsize=(8, 6))
        for technique_name, nmse_dict in results.items():
            compression_rates = list(nmse_dict.keys())
            nmse_values = list(nmse_dict.values())
            plt.plot(compression_rates, nmse_values, marker='o', label=technique_name)

        plt.xlabel('Compression Rate')
        plt.ylabel('Normalized MSE')
        plt.yscale('log')
        plt.grid(True)
        plt.legend()
        plt.title('Single Run Compression Analysis')
        plt.show()




if __name__ == "__main__":
    # Simulation parameters
    M = 50 # Vector length
    N = 5000  # Monte Carlo simulations
    N0_values = [0.5]  # Noise variance values
    compression_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    analyzer = CompressionAnalyzerUpdated(techniques={
        "DCT Compression": DCTCompression(),
        "DFT Compression": DFTCompression(),
        "Variable Length Encoding": VariableLengthEncodingCompression()
    })

    # results_montecarlo = analyzer.analyze_and_plot(
    #     mode='montecarlo',
    #     M=256, N=100,
    #     N0_values=[1],
    #     compression_rates=compression_rates
    # )


    # Use this cfg variable whenever we need to access some constant
    cfg = utils.Config(
        num_rx_antennas=1,
        num_tx_antennas=64,
        num_subcarriers=160,
        train_test_split=0.8,
        data_root="../data/dataset1",
        # duplicate_data=1,
        # data_snr=-1
    )


    #load the dataset
    train_set, test_set = dataset.load_data(cfg)

    csis = utils.reshape_tensor(train_set.csi_samples, K=1)

    print(f"csis shape: {csis.shape}")

    results_single = analyzer.analyze_and_plot(
        mode='single',
        vector=csis,
        compression_rates=compression_rates
    )



    #
    # # Create updated analyzer
    # analyzer_updated = CompressionAnalyzerUpdated(techniques_updated)
    #
    # # Run updated simulation
    # results_updated = analyzer_updated.monte_carlo_compression(M, N, N0_values, compression_rates)
    #
    # # Plot results
    # analyzer_updated.plot_compression_performance(results_updated)