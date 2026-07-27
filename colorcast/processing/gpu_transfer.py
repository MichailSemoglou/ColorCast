"""
GPU-accelerated color transfer using CuPy.

This module provides GPU implementations of color transfer algorithms
for improved performance on large images and batch processing.
"""

import numpy as np

# Try to import CuPy, fallback gracefully if not available
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None  # Type stub for mypy
    import warnings
    warnings.warn("CuPy is not installed. GPU acceleration will be disabled.")


def gpu_histogram_matching(
    source: np.ndarray,
    reference: np.ndarray
) -> np.ndarray:
    """
    GPU-accelerated histogram matching using CuPy.

    Note: The CuPy path currently copies channel data to the CPU for
    histogram matching via scikit-image and transfers results back.
    It is therefore no faster than the CPU fallback.  A native GPU
    histogram implementation is planned.

    Args:
        source: Source image array (H, W, 3) in range [0, 255]
        reference: Reference image array (H, W, 3) in range [0, 255]

    Returns:
        Matched image array (H, W, 3) in range [0, 255]
    """
    if not HAS_CUPY:
        # Fallback to CPU implementation
        from skimage import exposure
        result = np.empty_like(source)
        for i in range(3):
            result[:, :, i] = exposure.match_histograms(
                source[:, :, i], reference[:, :, i]
            )
        return result.astype(source.dtype)
    
    # Convert to GPU arrays (float32)
    source_gpu = cp.asarray(source, dtype=cp.float32)
    reference_gpu = cp.asarray(reference, dtype=cp.float32)
    
    # Compute histogram matching per channel on GPU
    matched_gpu = cp.empty_like(source_gpu)
    for i in range(3):
        # Extract channels
        source_channel = source_gpu[:, :, i]
        reference_channel = reference_gpu[:, :, i]
        
        # Match histograms using scikit-image compatible implementation
        # For now, use CPU for histogram matching as it's complex
        # but move data back to CPU for the matching operation
        source_cpu = cp.asnumpy(source_channel)
        reference_cpu = cp.asnumpy(reference_channel)
        
        from skimage import exposure
        matched_channel = exposure.match_histograms(
            source_cpu, reference_cpu
        )
        
        matched_gpu[:, :, i] = cp.asarray(matched_channel, dtype=cp.float32)
    
    # Transfer result back to CPU
    result = cp.asnumpy(matched_gpu).astype(source.dtype)
    return result


def gpu_mean_std_transfer(
    source: np.ndarray,
    reference: np.ndarray
) -> np.ndarray:
    """
    GPU-accelerated mean and standard deviation transfer using CuPy.
    
    Args:
        source: Source image array (H, W, 3) in range [0, 1]
        reference: Reference image array (H, W, 3) in range [0, 1]

    Returns:
        Transferred image array (H, W, 3) in range [0, 1]
    """
    if not HAS_CUPY:
        # Fallback to CPU implementation
        result = np.empty_like(source)
        for i in range(3):
            source_mean = np.mean(source[:, :, i])
            source_std = np.std(source[:, :, i])
            ref_mean = np.mean(reference[:, :, i])
            ref_std = np.std(reference[:, :, i])
            
            epsilon = 1e-8
            result[:, :, i] = (
                (source[:, :, i] - source_mean)
                * (ref_std / (source_std + epsilon))
            ) + ref_mean
        
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
        epsilon = 1e-8
        result_gpu[:, :, i] = (
            (source_channel - source_mean)
            * (ref_std / (source_std + epsilon))
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

    Note: CuPy has no rgb2lab/lab2rgb equivalent, so both the GPU and
    CPU branches route through scikit-image on the CPU.  A native GPU
    Lab pipeline is planned.

    Args:
        source: Source image array (H, W, 3) in range [0, 1]
        reference: Reference image array (H, W, 3) in range [0, 1]
        alpha: Transfer strength (0.0 to 1.0)

    Returns:
        Transferred image array (H, W, 3) in range [0, 1]
    """
    if not HAS_CUPY:
        # Fallback to CPU implementation
        from skimage import color

        # Clamp alpha
        alpha = np.clip(alpha, 0.0, 1.0)

        # Convert RGB to Lab color space
        source_lab = color.rgb2lab(source)
        reference_lab = color.rgb2lab(reference)

        # Compute statistics in Lab space
        result_lab = np.empty_like(source_lab)
        for i in range(3):
            # Compute mean and std for source and reference
            source_mean = np.mean(source_lab[:, :, i])
            source_std = np.std(source_lab[:, :, i])
            ref_mean = np.mean(reference_lab[:, :, i])
            ref_std = np.std(reference_lab[:, :, i])

            # Apply statistical transfer
            epsilon = 1e-8
            result_lab[:, :, i] = (
                (source_lab[:, :, i] - source_mean)
                * (ref_std / (source_std + epsilon))
            ) + ref_mean

        # Clip to valid Lab ranges
        result_lab[:, :, 0] = np.clip(result_lab[:, :, 0], 0, 100)  # L channel
        result_lab[:, :, 1] = np.clip(result_lab[:, :, 1], -128, 127)  # a channel
        result_lab[:, :, 2] = np.clip(result_lab[:, :, 2], -128, 127)  # b channel

        # Apply alpha blending for partial transfer
        if alpha < 1.0:
            result_lab = source_lab * (1 - alpha) + result_lab * alpha

        # Convert Lab back to RGB
        result_rgb = color.lab2rgb(result_lab)

        # Ensure result is in valid range [0, 1]
        result_rgb = np.clip(result_rgb, 0, 1).astype(source.dtype)

        return result_rgb

    # CuPy has no rgb2lab/lab2rgb equivalent — both branches route through
    # scikit-image on the CPU.  This block exists to preserve the API shape
    # until a native GPU Lab pipeline lands.
    from skimage import color

    alpha = float(alpha)
    alpha = max(0.0, min(1.0, alpha))

    source_lab = color.rgb2lab(source)
    reference_lab = color.rgb2lab(reference)

    result_lab = np.empty_like(source_lab)
    for i in range(3):
        source_mean = np.mean(source_lab[:, :, i])
        source_std = np.std(source_lab[:, :, i])
        ref_mean = np.mean(reference_lab[:, :, i])
        ref_std = np.std(reference_lab[:, :, i])

        epsilon = 1e-8
        result_lab[:, :, i] = (
            (source_lab[:, :, i] - source_mean)
            * (ref_std / (source_std + epsilon))
        ) + ref_mean

    result_lab[:, :, 0] = np.clip(result_lab[:, :, 0], 0, 100)
    result_lab[:, :, 1] = np.clip(result_lab[:, :, 1], -128, 127)
    result_lab[:, :, 2] = np.clip(result_lab[:, :, 2], -128, 127)

    if alpha < 1.0:
        result_lab = source_lab * (1 - alpha) + result_lab * alpha

    result_rgb = color.lab2rgb(result_lab)
    return np.clip(result_rgb, 0, 1).astype(source.dtype)


def gpu_histogram_matching_multichannel(
    source: np.ndarray,
    reference: np.ndarray
) -> np.ndarray:
    """
    GPU-accelerated multichannel histogram matching using CuPy.
    
    Args:
        source: Source image array (H, W, 3) in range [0, 255]
        reference: Reference image array (H, W, 3) in range [0, 255]

    Returns:
        Matched image array (H, W, 3) in range [0, 255]
    """
    if not HAS_CUPY:
        # Fallback to CPU implementation
        from colorcast.processing.transfer_methods import match_histograms_multichannel
        return match_histograms_multichannel(source, reference)
    
    return gpu_histogram_matching(source, reference)


def is_gpu_available() -> bool:
    """Check if GPU acceleration is available."""
    return HAS_CUPY


# Export functions for module API
__all__ = [
    "gpu_histogram_matching",
    "gpu_mean_std_transfer",
    "gpu_lab_transfer",
    "gpu_histogram_matching_multichannel",
    "is_gpu_available",
]