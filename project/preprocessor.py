import numpy as np


class ComplexVectorPreprocessor:
    def __init__(self,normalization = "mean_std",conversion_method='real_imag'):
        """
        Initialize the preprocessor with conversion method and normalization tracking

        Args:
            conversion_method (str): Method to convert complex numbers
                                     ('real_imag' or 'amplitude_angle')
        """
        self.conversion_method = conversion_method
        self.normalization = normalization
        self.normalization_factors = {}

    def convert_complex_to_features(self, complex_data):
        """
        Convert complex data to real features based on selected method

        Args:
            complex_data (np.ndarray): Input complex data of shape (Nsamples, Nfeatures)

        Returns:
            np.ndarray: Converted real-valued features
        """
        if self.conversion_method == 'real_imag':
            return np.column_stack([complex_data.real, complex_data.imag])
        elif self.conversion_method == 'amplitude_angle':
            return np.column_stack([np.abs(complex_data), np.angle(complex_data)])
        else:
            raise ValueError("Invalid conversion method. Choose 'real_imag' or 'amplitude_angle'")

    def fit_normalization(self, features):
        """
        Compute and store normalization factors from training data

        Args:
            features (np.ndarray): Input features to compute normalization factors

        Returns:
            self: Allows method chaining
        """

        if(self.normalization == 'mean_std'):
            for i in range(features.shape[1]):
                mean = np.mean(features[:, i])
                std = np.std(features[:, i])
                self.normalization_factors[i] = {"mean":mean,"std":std}
        else:
            for i in range(features.shape[1]):
                max_val = np.max(features[:, i])
                self.normalization_factors[i] = max_val

        return self

    def normalize_features(self, features, apply_existing=False):
        """
        Normalize features using stored or computed normalization factors

        Args:
            features (np.ndarray): Input features to normalize
            apply_existing (bool): If True, use existing normalization factors

        Returns:
            np.ndarray: Normalized features
        """
        normalized_features = np.zeros_like(features)

        if(self.normalization == "max"):
            for i in range(features.shape[1]):
                if apply_existing:
                    # Use pre-computed normalization factor from training data
                    if i not in self.normalization_factors:
                        raise ValueError(f"Normalization factor for feature {i} not found. Call fit_normalization() first.")
                    max_val = self.normalization_factors[i]
                else:
                    # Compute new normalization factor
                    max_val = np.max(features[:, i])
                    self.normalization_factors[i] = max_val

                # Normalize using the selected max value
                normalized_features[:, i] = features[:, i] / (max_val + 1e-12)
        else:
            for i in range(features.shape[1]):
                if apply_existing:
                    # Use pre-computed normalization factor from training data
                    if i not in self.normalization_factors:
                        raise ValueError(
                            f"Normalization factor for feature {i} not found. Call fit_normalization() first.")
                    mean = self.normalization_factors[i]["mean"]
                    std = self.normalization_factors[i]["std"]
                else:
                    mean = np.mean(features[:, i])
                    std = np.std(features[:, i])
                    self.normalization_factors[i] = {"mean": mean, "std": std}

                # Normalize using the selected max value
                normalized_features[:, i] = (features[:, i]-mean) / std

        return normalized_features

    def denormalize_features(self, normalized_features):
        """
        Convert normalized features back to original scale using stored normalization factors

        Args:
            normalized_features (np.ndarray): Normalized input features

        Returns:
            np.ndarray: Denormalized features
        """
        denormalized_features = np.zeros_like(normalized_features)

        if(self.normalization == "max"):
            for i in range(normalized_features.shape[1]):
                max_val = self.normalization_factors[i]
                denormalized_features[:, i] = normalized_features[:, i] * (max_val + 1e-8)
        else:
            for i in range(normalized_features.shape[1]):
                mean = self.normalization_factors[i]["mean"]
                std = self.normalization_factors[i]["std"]
                denormalized_features[:, i] = normalized_features[:, i] * std + mean

        return denormalized_features

    def reconstruct_complex_data(self, denormalized_features):
        """
        Reconstruct complex data from denormalized features

        Args:
            denormalized_features (np.ndarray): Denormalized features

        Returns:
            np.ndarray: Reconstructed complex data
        """
        if self.conversion_method == 'real_imag':
            # Split features into real and imaginary parts
            half_features = denormalized_features.shape[1] // 2
            return denormalized_features[:, :half_features] + \
                1j * denormalized_features[:, half_features:]

        elif self.conversion_method == 'amplitude_angle':
            # Reconstruct from magnitude and phase
            half_features = denormalized_features.shape[1] // 2
            magnitudes = denormalized_features[:, :half_features]
            phases = denormalized_features[:, half_features:]
            return magnitudes * np.exp(1j * phases)

        else:
            raise ValueError("Invalid conversion method")

    def create_windowed_samples(self, normalized_data, window_size):
        """
        Create windowed samples with M-1 steps as input and Mth step as label

        Args:
            normalized_data (np.ndarray): Normalized input data
            window_size (int): Size of the sliding window

        Returns:
            tuple: X (input samples), y (labels)
        """
        X, y = [], []
        for i in range(len(normalized_data) - window_size):
            X.append(normalized_data[i:i + window_size])
            y.append(normalized_data[i + window_size])

        return np.array(X), np.array(y)

