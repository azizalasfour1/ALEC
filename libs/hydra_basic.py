"""
Hydra Feature Extraction Implementation

Based on the original Hydra implementation by Dempster et al. (2022)
https://github.com/angus924/hydra/blob/main/code/hydra.py

This module provides a minimal implementation of the Hydra feature extraction method
for time series classification. Hydra uses random convolutional kernels with different
dilations to extract features from time series data, followed by global max/min pooling
and sparse-aware scaling.

Key Components:
    - Hydra: Main feature extractor class using random dilated convolutions
    - SparseScaler: Specialized scaler designed for sparse feature matrices

The implementation is GPU-aware and optimized for batch processing of time series data.

References:
    Based on the Hydra paper implementation with optimizations for practical use.

Author: Research implementation for time series feature extraction
"""

# Enable future annotations for better type hinting compatibility
from __future__ import annotations

# Type hinting imports
from typing import Optional

# Third-party scientific computing imports
import numpy as np  # Numerical operations and mathematical functions
import torch  # PyTorch for tensor operations and neural networks
import torch.nn as nn  # Neural network modules and layers
import torch.nn.functional as F  # Functional interface for neural operations


# Default device selection: use CUDA if available, otherwise CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from config import HYDRA_DETERMINISTIC, HYDRA_DETERMINISTIC_STRICT


def _configure_torch_determinism() -> None:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.backends.cuda.matmul.fp32_precision = "ieee"     # disables TF32 for matmul
    except Exception:
        # Skip if this version of Torch doesn't expose this setting
        pass
    try:
        torch.backends.cudnn.conv.fp32_precision = "ieee"      # disables TF32 for conv
    except Exception:
        # Skip if this version of Torch doesn't expose this setting
        pass

    if HYDRA_DETERMINISTIC_STRICT:
        torch.use_deterministic_algorithms(True)


class Hydra(nn.Module):
    """
    Minimal Hydra feature extractor for time series classification.
    
    Hydra extracts features from time series using random convolutional kernels
    with different dilations. The method applies these kernels to both the original
    time series and its first-order difference, then performs global max/min pooling
    to create fixed-size feature representations.
    
    The implementation mirrors the original paper code with GPU-awareness and
    batch processing capabilities for efficient computation.
    
    Attributes:
        k (int): Number of random kernels per group
        g (int): Number of kernel groups (total kernels = k * g)
        device (torch.device): Device for computation (CPU or CUDA)
        dilations (torch.Tensor): Dilation rates for convolutions
        paddings (torch.Tensor): Padding sizes for each dilation
        W (torch.Tensor): Random convolutional kernel weights
        
    Example:
        >>> hydra = Hydra(input_length=100, k=8, g=64)
        >>> features = hydra(time_series_tensor)
        >>> print(features.shape)  # [batch_size, feature_dim]
    """

    def __init__(self, input_length: int, k: int = 8, g: int = 64, seed: Optional[int] = None, device: Optional[torch.device] = None):
        """
        Initialize the Hydra feature extractor.
        
        Args:
            input_length (int): Length of input time series
            k (int, optional): Number of kernels per group. Defaults to 8.
            g (int, optional): Number of kernel groups. Defaults to 64.
            seed (Optional[int], optional): Random seed for reproducibility. Defaults to None.
            device (Optional[torch.device], optional): Computation device. Defaults to None (auto-select).
            
        Note:
            The total number of features extracted will be approximately 2 * k * g * num_dilations,
            where num_dilations depends on the input_length.
        """
        super().__init__()
        
        # Configure reproducibility settings if requested
        if HYDRA_DETERMINISTIC:
            _configure_torch_determinism()
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        # Set computation device (GPU if available, otherwise CPU)
        self.device = device or DEVICE

        # Store hyperparameters
        self.k = k  # Number of kernels per group
        self.g = g  # Number of groups

        # Calculate dilation rates based on input length
        # Dilations are powers of 2, ensuring kernels capture patterns at different scales
        max_exponent = np.log2((input_length - 1) / (9 - 1))  # 9 is kernel size
        self.dilations = (2 ** torch.arange(int(max_exponent) + 1, device=self.device)).long()
        self.num_dilations = len(self.dilations)
        
        # Calculate padding to maintain output size for each dilation
        # Padding = (kernel_size - 1) * dilation / 2
        self.paddings = torch.div((9 - 1) * self.dilations, 2, rounding_mode="floor").int()

        # Set up kernel organization
        # divisor determines how many different input types we use (original + diff1)
        self.divisor = min(2, self.g)  # Use both original and diff if g >= 2
        self.h = self.g // self.divisor  # Number of groups per input type

        # Generate random convolutional kernels
        # Shape: [num_dilations, divisor, k*h, 1, 9]
        # - num_dilations: different dilation rates
        # - divisor: original series vs. first-order difference
        # - k*h: total kernels per (dilation, input_type) combination
        # - 1: input channels (univariate time series)
        # - 9: kernel size
        W = torch.randn(self.num_dilations, self.divisor, self.k * self.h, 1, 9, device=self.device)
        
        # Normalize kernels: zero-mean and unit L1 norm
        W = W - W.mean(-1, keepdims=True)  # Zero mean across kernel dimension
        W = W / W.abs().sum(-1, keepdims=True)  # Unit L1 norm
        
        # Register as buffer (part of model state but not a parameter)
        self.register_buffer("W", W)

    def batch(self, X: torch.Tensor, batch_size: int = 256) -> torch.Tensor:
        """
        Process input tensor in batches for memory efficiency.
        
        This method is useful when processing large datasets that don't fit in GPU memory.
        It splits the input into smaller batches and processes them sequentially.
        
        Args:
            X (torch.Tensor): Input tensor of shape [n_samples, n_channels, length]
            batch_size (int, optional): Maximum batch size. Defaults to 256.
            
        Returns:
            torch.Tensor: Extracted features of shape [n_samples, feature_dim]
            
        Note:
            If the input size is smaller than batch_size, processes everything at once.
        """
        n = X.shape[0]  # Number of samples
        
        # If input is small enough, process all at once
        if n <= batch_size:
            return self(X)
        
        # Process in batches
        out = []
        for idx in torch.arange(n, device=X.device).split(batch_size):
            out.append(self(X[idx]))  # Process current batch
        
        # Concatenate results from all batches
        return torch.cat(out)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: extract Hydra features from input time series.
        
        The method applies random convolutional kernels at different dilations
        to both the original time series and its first-order difference, then performs
        global max and min pooling to create feature representations.
        
        Args:
            X (torch.Tensor): Input time series of shape [batch_size, 1, length]
            
        Returns:
            torch.Tensor: Extracted features of shape [batch_size, feature_dim]
            
        Process:
            1. Compute first-order difference of input
            2. For each dilation rate:
                a. Apply convolutions to original and/or diff1 series
                b. Reshape to separate kernels
                c. Perform global max and min pooling
                d. Count occurrences of max/min values
            3. Concatenate all features
        """
        # Move input to correct device
        X = X.to(self.device)
        n = X.shape[0]  # Batch size

        # Compute first-order difference if using multiple input types
        if self.divisor > 1:
            diff_X = torch.diff(X)  # First-order difference: X[t] - X[t-1]

        # List to store features from all dilations and input types
        Z = []
        
        # Process each dilation rate
        for di in range(self.num_dilations):
            d = int(self.dilations[di].item())  # Current dilation
            p = int(self.paddings[di].item())   # Current padding
            
            # Process each input type (original series, diff1)
            for df in range(self.divisor):
                # Select input source: original (df=0) or diff1 (df=1)
                src = X if df == 0 else diff_X
                
                # Apply dilated convolution
                # Output shape: [batch_size, k*h, output_length]
                _Z = F.conv1d(src, self.W[di, df], dilation=d, padding=p)
                
                # Reshape to separate kernels: [batch_size, h, k, output_length]
                _Z = _Z.view(n, self.h, self.k, -1)

                # Global max pooling across time dimension
                max_values, max_indices = _Z.max(2)  # Max over kernel dimension
                count_max = torch.zeros(n, self.h, self.k, device=X.device)
                
                # Global min pooling across time dimension  
                min_values, min_indices = _Z.min(2)  # Min over kernel dimension
                count_min = torch.zeros(n, self.h, self.k, device=X.device)

                # Count max occurrences: scatter max values to their kernel positions
                count_max.scatter_add_(-1, max_indices, max_values)
                
                # Count min occurrences: scatter ones to min positions (frequency count)
                count_min.scatter_add_(-1, min_indices, torch.ones_like(min_values))

                # Add both max and min features to the feature list
                Z.append(count_max)
                Z.append(count_min)

        # Concatenate all features and flatten to 2D
        # Final shape: [batch_size, total_features]
        return torch.cat(Z, 1).view(n, -1)


class SparseScaler:
    """
    Specialized scaler for sparse feature matrices from Hydra.
    
    This scaler is designed to handle the sparse nature of Hydra features,
    which often contain many zero values. It applies square root transformation
    to reduce the impact of large values and uses adaptive epsilon based on
    sparsity levels.
    
    The scaler handles zero values specially by masking them during normalization,
    preserving the sparse structure of the feature matrix.
    
    Attributes:
        mask (bool): Whether to mask zero values during scaling
        exponent (int): Exponent for adaptive epsilon calculation
        fitted (bool): Whether the scaler has been fitted to data
        epsilon (torch.Tensor): Adaptive epsilon values per feature
        mu (torch.Tensor): Mean values per feature
        sigma (torch.Tensor): Standard deviation values per feature
        
    Example:
        >>> scaler = SparseScaler()
        >>> scaled_features = scaler.fit_transform(hydra_features)
    """

    def __init__(self, mask: bool = True, exponent: int = 4):
        """
        Initialize the SparseScaler.
        
        Args:
            mask (bool, optional): Whether to mask zero values during scaling. Defaults to True.
            exponent (int, optional): Exponent for adaptive epsilon calculation. Defaults to 4.
            
        Note:
            The exponent controls how much the epsilon adapts to sparsity levels.
            Higher values make epsilon more sensitive to the proportion of zeros.
        """
        self.mask = mask          # Whether to preserve zero structure
        self.exponent = exponent  # Controls epsilon adaptation to sparsity
        self.fitted = False       # Track fitting status

    def fit(self, X: torch.Tensor):
        """
        Fit the scaler to the input data.
        
        Computes statistics needed for scaling: mean, standard deviation,
        and adaptive epsilon based on the sparsity pattern of the data.
        
        Args:
            X (torch.Tensor): Input feature matrix to fit on
            
        Raises:
            RuntimeError: If the scaler has already been fitted
            
        Process:
            1. Apply square root transformation to reduce large value impact
            2. Calculate sparsity-adaptive epsilon per feature
            3. Compute mean and standard deviation per feature
            4. Add epsilon to standard deviation for numerical stability
        """
        if self.fitted:
            raise RuntimeError("SparseScaler already fitted")
        
        # Apply square root transformation and clamp negative values to 0
        X = X.clamp(0).sqrt()
        
        # Calculate adaptive epsilon based on sparsity level per feature
        # Higher sparsity (more zeros) leads to larger epsilon
        self.epsilon = (X == 0).float().mean(0) ** self.exponent + 1e-8
        
        # Compute feature-wise statistics
        self.mu = X.mean(0)      # Mean per feature
        # Use unbiased=False to avoid degrees of freedom warning when n_samples=1
        # Also suppress the warning since it's expected for very sparse features
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='std.*degrees of freedom')
            self.sigma = X.std(0, unbiased=False) + self.epsilon  # Std dev + adaptive epsilon
        
        # Mark as fitted
        self.fitted = True

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        Transform input data using fitted statistics.
        
        Applies the same square root transformation and normalization
        used during fitting, with optional masking of zero values.
        
        Args:
            X (torch.Tensor): Input feature matrix to transform
            
        Returns:
            torch.Tensor: Scaled feature matrix
            
        Raises:
            RuntimeError: If the scaler hasn't been fitted yet
            
        Process:
            1. Apply square root transformation
            2. Subtract mean and divide by standard deviation
            3. Optionally mask zero values to preserve sparsity
        """
        if not self.fitted:
            raise RuntimeError("SparseScaler not fitted")
        
        # Apply same transformation as during fitting
        X = X.clamp(0).sqrt()
        
        if self.mask:
            # Preserve zero structure: only scale non-zero values
            return ((X - self.mu) * (X != 0)) / self.sigma
        else:
            # Standard scaling without masking
            return (X - self.mu) / self.sigma

    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        Fit the scaler and transform the data in one step.
        
        Convenience method that combines fitting and transformation.
        Equivalent to calling fit(X) followed by transform(X).
        
        Args:
            X (torch.Tensor): Input feature matrix to fit and transform
            
        Returns:
            torch.Tensor: Scaled feature matrix
            
        Example:
            >>> scaler = SparseScaler()
            >>> scaled_data = scaler.fit_transform(features)
        """
        self.fit(X)           # Fit scaler to data
        return self.transform(X)  # Transform using fitted statistics
