"""
Color transfer functions with CuPy-backed acceleration where available.

The ``gpu_`` prefix is historical: these entry points accept numpy arrays
and return numpy arrays regardless of whether CuPy is installed.  When
CuPy is absent every function falls back to the CPU path transparently.
Only ``gpu_mean_std_transfer`` and ``gpu_lab_transfer`` exercise genuine GPU
kernels; the
remaining functions route through scikit-image on the CPU and are kept
under their existing names so callers do not need to change import paths
when GPU kernels land.

Use ``is_gpu_available()`` to check at runtime whether the accelerator
will be used.
"""

import numpy as np

from colorcast.processing.transfer_methods import (
    _EPSILON,
    _LAB_AB_BOUNDS,
    _LAB_L_BOUNDS,
    _meanstd_transfer,
    validate_and_resize_images,
)

# Try to import CuPy, fallback gracefully if not available
try:
    import cupy as cp

    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None  # Type stub for mypy


def is_gpu_available() -> bool:
    """Check whether CuPy is installed and a usable CUDA device is present."""
    if not HAS_CUPY:
        return False
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except cp.cuda.runtime.CUDARuntimeError:
        return False


def gpu_histogram_matching(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Histogram matching (CPU-only until a native CuPy kernel lands).

    Both CuPy-available and fallback paths currently route through
    scikit-image on the CPU.  The function is kept as a named entry point
    so callers do not need to change when GPU acceleration arrives.

    Args:
        source: Source image array (H, W, 3), any numeric dtype.
        reference: Reference image array (H, W, 3), any numeric dtype.

    Returns:
        Matched image array (H, W, 3), float32 in range [0, 1].
    """
    from skimage import exposure

    source, reference = validate_and_resize_images(source, reference)

    result = np.empty_like(source)
    for i in range(3):
        result[:, :, i] = exposure.match_histograms(source[:, :, i], reference[:, :, i])
    return result.astype(source.dtype)


def gpu_mean_std_transfer(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    GPU-accelerated mean and standard deviation transfer using CuPy.

    Args:
        source: Source image array (H, W, 3) in range [0, 1]
        reference: Reference image array (H, W, 3) in range [0, 1]

    Returns:
        Transferred image array (H, W, 3) in range [0, 1]
    """
    source, reference = validate_and_resize_images(source, reference)

    if not is_gpu_available():
        result = _meanstd_transfer(source, reference)
        return np.clip(result, 0, 1).astype(source.dtype)

    # Convert to GPU arrays (images are already float32 in [0, 1])
    source_gpu = cp.asarray(source, dtype=cp.float32)
    reference_gpu = cp.asarray(reference, dtype=cp.float32)

    # Compute mean and std transfer per channel on GPU
    result_gpu = cp.empty_like(source_gpu)
    for i in range(3):
        source_channel = source_gpu[:, :, i]
        reference_channel = reference_gpu[:, :, i]

        # Compute statistics on GPU
        source_mean = cp.mean(source_channel)
        source_std = cp.std(source_channel)
        ref_mean = cp.mean(reference_channel)
        ref_std = cp.std(reference_channel)

        # Apply transfer
        result_gpu[:, :, i] = (
            (source_channel - source_mean) * (ref_std / (source_std + _EPSILON))
        ) + ref_mean

    # Clip to [0, 1]
    result_gpu = cp.clip(result_gpu, 0, 1)
    result = cp.asnumpy(result_gpu)
    result = np.clip(result, 0, 1).astype(source.dtype)

    return result


def gpu_lab_transfer(
    source: np.ndarray,
    reference: np.ndarray,
    alpha: float = 1.0,
) -> np.ndarray:
    """
    GPU-accelerated Lab color space transfer using CuPy.

    Lab conversion and back-conversion run on CPU (no CuPy equivalent in
    scikit-image).  The per-channel mean/standard-deviation transfer,
    channel-wise clipping, and alpha blending are offloaded to CuPy when a
    GPU is present.

    Args:
        source: Source image array (H, W, 3) in range [0, 1]
        reference: Reference image array (H, W, 3) in range [0, 1]
        alpha: Transfer strength (0.0 to 1.0)

    Returns:
        Transferred image array (H, W, 3) in range [0, 1]
    """
    source, reference = validate_and_resize_images(source, reference)
    alpha = float(np.clip(alpha, 0.0, 1.0))

    from skimage import color

    source_lab = color.rgb2lab(source)
    reference_lab = color.rgb2lab(reference)

    if is_gpu_available():
        source_gpu = cp.asarray(source_lab, dtype=cp.float32)
        reference_gpu = cp.asarray(reference_lab, dtype=cp.float32)
        result_gpu = cp.empty_like(source_gpu)
        for i in range(3):
            s_ch = source_gpu[:, :, i]
            r_ch = reference_gpu[:, :, i]
            result_gpu[:, :, i] = (
                (s_ch - cp.mean(s_ch)) * (cp.std(r_ch) / (cp.std(s_ch) + _EPSILON))
            ) + cp.mean(r_ch)
        result_gpu[:, :, 0] = cp.clip(result_gpu[:, :, 0], *_LAB_L_BOUNDS)
        result_gpu[:, :, 1] = cp.clip(result_gpu[:, :, 1], *_LAB_AB_BOUNDS)
        result_gpu[:, :, 2] = cp.clip(result_gpu[:, :, 2], *_LAB_AB_BOUNDS)
        if alpha < 1.0:
            result_gpu = source_gpu * (1.0 - alpha) + result_gpu * alpha
        result_lab = cp.asnumpy(result_gpu)
    else:
        result_lab = _meanstd_transfer(source_lab, reference_lab)
        result_lab[:, :, 0] = np.clip(result_lab[:, :, 0], *_LAB_L_BOUNDS)
        result_lab[:, :, 1] = np.clip(result_lab[:, :, 1], *_LAB_AB_BOUNDS)
        result_lab[:, :, 2] = np.clip(result_lab[:, :, 2], *_LAB_AB_BOUNDS)
        if alpha < 1.0:
            result_lab = source_lab * (1.0 - alpha) + result_lab * alpha

    result_rgb = color.lab2rgb(result_lab)
    return np.clip(result_rgb, 0, 1).astype(source.dtype)


def gpu_histogram_matching_multichannel(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Multichannel histogram matching (CPU-only until a native CuPy kernel lands).

    Convenience alias for :func:`gpu_histogram_matching` that accepts and
    returns the same float32 [0, 1] contract as the rest of the package.

    Args:
        source: Source image array (H, W, 3), any numeric dtype.
        reference: Reference image array (H, W, 3), any numeric dtype.

    Returns:
        Matched image array (H, W, 3), float32 in range [0, 1].
    """
    return gpu_histogram_matching(source, reference)


# Export functions for module API
__all__ = [
    "gpu_histogram_matching",
    "gpu_mean_std_transfer",
    "gpu_lab_transfer",
    "gpu_histogram_matching_multichannel",
    "is_gpu_available",
]
