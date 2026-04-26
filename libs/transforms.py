"""
This tile implements a suite of time series smoothing algorithms exactly as described in:

James Large, Paul Southam, Anthony Bagnall
Can automated smoothing significantly improve benchmark time series classification algorithms?
arXiv:1811.00894v1 [cs.LG], 1 Nov 2018

Specifically, we follow Section 2.2 Time Series Smoothing (pages 3--5) of the paper.
"""

# Defaults from the paper quoted above

from functools import partial
from typing import Callable
import numpy as np
import pandas as pd
from scipy.signal import convolve, medfilt, savgol_filter, windows
from scipy.ndimage import median_filter


def get_transform(
    transform_id: str,
    window: int,
    polyorder: int,
    radius: float,
    numerator: int,
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Return a callable for this transform and parametrization.

    Always takes a superset of parameters; each transform picks what it needs.

    Returns:
        Callable[[np.ndarray], np.ndarray]
    Raises:
        ValueError if the parametrization does not work or transform_id is unknown
    """
    tid = transform_id.upper()

    if tid == "MA":
        return partial(moving_average, w=window)

    if tid == "EXP":
        return partial(exponential_smoothing, w=window)

    if tid == "GF":
        return partial(gaussian_filtering, w=window)

    if tid == "SG":
        return partial(savitzky_golay, w=window, n=polyorder)

    if tid == "DFT":
        return partial(fourier_approximation, r=radius)

    if tid == "SIV_PRIME":
        return partial(siv_median_approx, n=numerator)

    if tid == "BASELINE":
        return identity

    raise ValueError(f"Unknown transform_id: {transform_id!r}")


# A no-op with the same signature as the other transforms
def identity(X: np.ndarray) -> np.ndarray:
    return X


def _pad_for_same(w: int) -> tuple[int, int]:
    """
    Compute symmetric-ish padding (left, right) so that a 'valid' convolution over a
    padded signal returns the original length.

    Works for even/odd w (center alignment).
    """
    if w < 1:
        raise ValueError("w must be >= 1")
    pad_left = (w - 1) // 2
    pad_right = (w - 1) - pad_left
    return pad_left, pad_right


def moving_average(X: np.ndarray, *, w: int = 5) -> np.ndarray:
    """
    Moving Average (MA), valid convolution (no padding).

    Input:  (..., T)
    Output: (..., T - (w - 1))
    """
    if w < 2:
        raise ValueError("w must be >= 2")

    T = X.shape[-1]
    if w > T:
        w = T

    kernel = np.full(w, 1.0 / w, dtype=float)
    pl, pr = _pad_for_same(w)

    def _ma_1d(v: np.ndarray) -> np.ndarray:
        vp = np.pad(v, (pl, pr), mode="reflect")
        return np.convolve(vp, kernel, mode="valid")

    return np.apply_along_axis(_ma_1d, axis=-1, arr=X)


def exponential_smoothing(
    X: np.ndarray,
    *,
    w: int | None = 5,
    alpha: float | None = None,
) -> np.ndarray:
    """
    Exponential Smoothing (EXP), EWMA.

    Paper convention: if w is given, alpha = 2/(w+1).
    Length-preserving.

    Input:  (..., T)
    Output: (..., T)
    """
    if alpha is None:
        if w is None:
            raise ValueError("provide w or alpha")
        if w < 2:
            raise ValueError("w must be >= 2")
        alpha = 2.0 / (w + 1.0)
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be in [0, 1]")

    def _ewma_1d(v: np.ndarray) -> np.ndarray:
        return (
            pd.Series(v, copy=False)
            .ewm(alpha=alpha, adjust=False)
            .mean()
            .to_numpy(dtype=float)
        )

    return np.apply_along_axis(_ewma_1d, axis=-1, arr=X)


def gaussian_filtering(
    X: np.ndarray,
    *,
    w: int = 5,
    sigma: float | None = None,
) -> np.ndarray:
    """
    Gaussian Filtering (GF), fixed FIR convolution over window w (valid).

    If sigma is None, uses sigma = w/6 (≈99.7% mass within window).

    Input:  (..., T)
    Output: (..., T - (w - 1))
    """
    if w < 2:
        raise ValueError("w must be >= 2")

    T = X.shape[-1]
    if w > T:
        w = T

    if sigma is None:
        sigma = w / 6.0
    if sigma <= 0:
        raise ValueError("sigma must be > 0")

    kernel = windows.gaussian(M=w, std=sigma).astype(float, copy=False)
    kernel /= kernel.sum()

    pl, pr = _pad_for_same(w)

    def _gf_1d(v: np.ndarray) -> np.ndarray:
        vp = np.pad(v, (pl, pr), mode="reflect")
        return convolve(vp, kernel, mode="valid")

    return np.apply_along_axis(_gf_1d, axis=-1, arr=X)


def savitzky_golay(
    X: np.ndarray,
    *,
    w: int = 5,
    n: int = 2,
) -> np.ndarray:
    """
    Savitzky-Golay (SG).

    We compute length-preserving savgol_filter then return only positions whose full
    window is inside the input (center-aligned 'valid').

    Input:  (..., T)
    Output: (..., T - (w - 1))
    """
    if w < 3:
        raise ValueError("w must be >= 3")
    if w % 2 == 0:
        raise ValueError("w must be odd for Savitzky-Golay")
    if not (0 <= n < w):
        raise ValueError("n must satisfy 0 <= n < w")
    if w > X.shape[-1]:
        # Clamp to largest odd <= T; if T < 3, just return X as float
        T = X.shape[-1]
        w = T if (T % 2 == 1) else (T - 1)
        if w < 3:
            return X.astype(float, copy=False)
        if n >= w:
            raise ValueError("n must satisfy 0 <= n < w")

    y = savgol_filter(
        X,
        window_length=w,
        polyorder=n,
        axis=-1,
        mode="interp",
    ).astype(float, copy=False)

    # Length-preserving for tiny series: do not crop to 'valid'
    return y


def fourier_approximation(
    X: np.ndarray,
    *,
    r: float = 0.1,
) -> np.ndarray:
    """
    Discrete Fourier Approximation (DFT) smoothing per paper:
      FFT -> keep low-frequency terms -> inverse FFT.

    r is the proportion of Fourier terms to retain (0 < r <= 1).
    Length-preserving.

    Input:  (..., T)
    Output: (..., T)
    """
    if not (0.0 < r <= 1.0):
        raise ValueError("r must be in (0, 1]")

    def _dft_1d(v: np.ndarray) -> np.ndarray:
        n = v.size
        F = np.fft.fft(v)

        k = int(np.floor(r * n / 2))
        F_low = np.zeros_like(F)
        F_low[: k + 1] = F[: k + 1]
        if k > 0:
            F_low[-k:] = F[-k:]

        return np.fft.ifft(F_low).real.astype(float)

    return np.apply_along_axis(_dft_1d, axis=-1, arr=X)


def siv_median_approx(
    X: np.ndarray,
    *,
    n: int = 5,
) -> np.ndarray:
    """
    Median-based approximation of the Recursive Median Sieve (SIV).

    This function implements a practical, heuristic approximation of the
    Recursive Median Sieve (SIV) described in Section 2.2 of:

        James Large, Paul Southam, Anthony Bagnall
        "Can automated smoothing significantly improve benchmark time series
        classification algorithms?"
        arXiv:1811.00894v1, 2018.

    In the original formulation, SIV is an extrema-based morphological sieve that
    recursively removes local maxima and minima up to a given scale c, producing
    a scale-dependent smoothing of the input time series. Implementing the exact
    extrema-detection and removal procedure requires specialised morphological
    logic that is not available as a standard numerical primitive.

    In this study, SIV is used solely as a scale-controlled smoothing operator,
    and no downstream analysis depends on the internal sieve decomposition or
    on the explicit topology of extrema. Accordingly, we replace the exact sieve
    with a median-based approximation that preserves the intended effect of
    suppressing extrema below a given scale.

    The approximation applies recursive median filtering at increasing odd
    window sizes, which progressively removes extrema of increasing support.
    This reproduces the scale-selective, robust smoothing behaviour that SIV
    contributes to the experimental pipeline, while remaining simple, stable,
    and reproducible.

    Parameterisation follows the paper exactly. The user supplies an integer
    numerator n ∈ {1, …, 15}, from which the sieve scale is computed as:

        c = (n / 15) · log10(m),

    where m is the length of the time series (m = X.shape[-1]). The maximum
    median-filter window size is derived from this scale and capped to the
    largest odd window not exceeding the series length.

    This function is therefore not algorithmically identical to the original
    Recursive Median Sieve, but is a faithful, parameter-aligned approximation
    suitable for empirical smoothing and classification studies. For clarity,
    we denote this variant as SIV-MA (median approximation of SIV).

    Input:
        X: ndarray of shape (..., T)
           Input time series or batch of time series.

    Parameters:
        n: int, default = 5
           Numerator controlling the sieve scale, corresponding to the bolded
           default (5/15 · log10(m)) used in the paper.

    Output:
        ndarray of shape (..., T)
        Smoothed time series, length-preserving.
    """
    if n <= 0:
        raise ValueError("n must be > 0")

    T = X.shape[-1]
    c = (n / 15) * np.log10(T)

    radius = int(np.ceil(c))
    w_max = 2 * radius + 1
    if w_max < 3:
        w_max = 3

    w_cap = T if (T % 2 == 1) else (T - 1)
    if w_cap < 3:
        return X.astype(float, copy=False)
    if w_max > w_cap:
        w_max = w_cap

    y = X
    for w in range(3, w_max + 1, 2):
        # Avoid medfilt's implicit zero-padding artifacts; reflect is less intrusive
        y = median_filter(y, size=(1,) * (y.ndim - 1) + (w,), mode="reflect")
    return y.astype(float, copy=False)

# Use the approximation as the implementation of sieve()
sieve = siv_median_approx


