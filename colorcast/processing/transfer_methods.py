"""Color transfer methods and algorithms."""

from typing import Literal
import numpy as np
from skimage import exposure, transform, color
from colorcast.utils.exceptions import InvalidImageFormatError


def validate_and_resize_images(source: np.ndarray, reference: np.ndarray) -> tuple:
    """
    Validate and ensure both images are compatible for processing.

    Args:
        source: Source image array (H, W, 3)
        reference: Reference image array (H, W, 3)

    Returns:
        Tuple of (validated_source, validated_reference)

    Raises:
        InvalidImageFormatError: If images have incorrect dimensions
    """
    # Validate source
    if source.ndim != 3 or source.shape[2] != 3:
        raise InvalidImageFormatError(
            "Source image is not 3-channel RGB after preprocessing."
        )

    # Validate reference
    if reference.ndim != 3 or reference.shape[2] != 3:
        raise InvalidImageFormatError(
            "Reference image is not 3-channel RGB after preprocessing."
        )

    # Resize if needed
    if source.shape != reference.shape:
        reference = transform.resize(
            reference, source.shape, anti_aliasing=True, preserve_range=True
        )

    return source, reference


def match_histograms_multichannel(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Match histograms between images per channel.

    Based on histogram matching algorithm described in:
    Gonzalez, R. C., & Woods, R. E. (2017).
    Digital Image Processing (4th ed.). Pearson.
    pp. 156-163

    The algorithm maps cumulative distribution function (CDF)
    of source image to CDF of reference image,
    preserving statistical properties of reference.

    Args:
        source: Source image array (H, W, 3)
        reference: Reference image array (H, W, 3)

    Returns:
        Matched image array (H, W, 3)
    """
    source, reference = validate_and_resize_images(source, reference)

    matched = np.empty_like(source)
    for i in range(3):
        matched[:, :, i] = exposure.match_histograms(source[:, :, i], reference[:, :, i])
    
    return matched.astype(source.dtype)


def color_transfer_meanstd(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Transfer color using mean and standard deviation matching per channel.

    This is a simplified per-channel RGB normalization.  For the canonical
    Lab-space statistical transfer, see `color_transfer_lab` (Reinhard et al.,
    2001).

    This method matches first and second moments (mean and
    standard deviation) of each color channel independently,
    providing natural-looking color transfers.

    Mathematical formulation:
    result = ((source - μ_source) × (σ_ref / σ_source)) + μ_ref

    where μ is mean and σ is standard deviation per channel.

    Args:
        source: Source image array (H, W, 3)
        reference: Reference image array (H, W, 3)

    Returns:
        Color-transferred image array (H, W, 3) in range [0, 1]
    """
    source, reference = validate_and_resize_images(source, reference)

    result = np.empty_like(source)
    for i in range(3):
        source_mean = np.mean(source[:, :, i])
        source_std = np.std(source[:, :, i])
        ref_mean = np.mean(reference[:, :, i])
        ref_std = np.std(reference[:, :, i])

        # Avoid division by zero
        epsilon = 1e-8
        result[:, :, i] = (
            (source[:, :, i] - source_mean) * (ref_std / (source_std + epsilon))
        ) + ref_mean

    return np.clip(result, 0, 1).astype(source.dtype)


def lut_transfer_with_curve(
    source: np.ndarray,
    reference: np.ndarray,
    curve_type: Literal["linear", "s-curve", "contrast"] = "linear",
) -> np.ndarray:
    """
    LUT-based transfer with tone curve adjustment.

    Combines histogram matching with tone curve adjustments for
    enhanced contrast and visual appeal.

    Args:
        source: Source image array (H, W, 3)
        reference: Reference image array (H, W, 3)
        curve_type: Type of tone curve to apply ('linear', 's-curve', 'contrast')

    Returns:
        Transferred image array (H, W, 3) in range [0, 1]
    """
    from colorcast.processing.curves import apply_curve

    source, reference = validate_and_resize_images(source, reference)

    matched = np.empty_like(source)
    for i in range(3):
        matched[:, :, i] = exposure.match_histograms(source[:, :, i], reference[:, :, i])
        matched[:, :, i] = apply_curve(matched[:, :, i], curve_type)

    return np.clip(matched, 0, 1).astype(source.dtype)


def color_transfer_lab(
    source: np.ndarray,
    reference: np.ndarray,
    alpha: float = 1.0,
) -> np.ndarray:
    """
    Transfer color using Lab color space with statistical matching (Reinhard method).

    Based on seminal paper:
    Reinhard, E., Ashikhmin, M., Gooch, B., & Shirley, P. (2001).
    Color transfer between images.
    IEEE Computer Graphics and Applications, 21(5), 34-41.
    DOI: 10.1109/38.946629.

    This method operates in L*a*b* color space, which is
    perceptually uniform and separates luminance (L) from chromaticity (a, b).
    The algorithm:

    1. Convert RGB images to Lab color space
    2. Compute mean and standard deviation for each channel
    3. Apply statistical transfer directly in Lab space
    4. Convert back to RGB

    Mathematical formulation (in Lab space):
    result_lab = ((source_lab - μ_source) × (σ_ref / σ_source)) + μ_ref

    where μ is the mean and σ is the standard deviation per channel.

    Advantages:
    - Perceptually uniform results
    - Better color preservation than RGB methods
    - More natural-looking transfers
    - Industry-standard in color grading

    Args:
        source: Source image array (H, W, 3) in range [0, 1]
        reference: Reference image array (H, W, 3) in range [0, 1]
        alpha: Strength of transfer (0.0 to 1.0), where 1.0 is full transfer.
               Values outside this range are clamped.

    Returns:
        Color-transferred image array (H, W, 3) in range [0, 1]

    Raises:
        InvalidImageFormatError: If images have incorrect dimensions
    """
    source, reference = validate_and_resize_images(source, reference)
    
    # Clamp alpha to valid range [0, 1]
    alpha = np.clip(alpha, 0.0, 1.0)

    # Convert RGB to Lab color space
    # Lab uses float64 internally, values: L in [0, 100], a, b in [-128, 127]
    source_lab = color.rgb2lab(source)
    reference_lab = color.rgb2lab(reference)

    # Compute statistics in Lab space for each channel
    result_lab = np.empty_like(source_lab)
    for i in range(3):
        # Compute mean and std for source and reference
        source_mean = np.mean(source_lab[:, :, i])
        source_std = np.std(source_lab[:, :, i])
        ref_mean = np.mean(reference_lab[:, :, i])
        ref_std = np.std(reference_lab[:, :, i])

        # Avoid division by zero
        epsilon = 1e-8

        # Apply statistical transfer directly in Lab space
        # This preserves color relationships while matching statistics
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
    result_rgb = np.clip(result_rgb, 0, 1)
    
    # Preserve original dtype
    result_rgb = result_rgb.astype(source.dtype)
    
    return result_rgb


def selective_color_transfer(
    source: np.ndarray,
    reference: np.ndarray,
    mode: Literal["full", "shadows", "midtones", "highlights"] = "full",
    shadow_threshold: float = 0.3,
    highlight_threshold: float = 0.7,
) -> np.ndarray:
    """
    Transfer colors selectively based on luminance regions.

    This function transfers color statistics only in specific tonal ranges,
    allowing for precise control over which parts of image are affected.

    The luminance calculation uses ITU-R BT.601 standard coefficients:
    - Red: 0.299
    - Green: 0.587
    - Blue: 0.114

    These weights approximate human perception of brightness, where green
    contributes most to perceived luminance.

    Args:
        source: Source image array (H, W, 3) in range [0, 1]
        reference: Reference image array (H, W, 3) in range [0, 1]
        mode: Transfer mode ('full', 'shadows', 'midtones', 'highlights')
        shadow_threshold: Luminance threshold for shadow region (0-1)
        highlight_threshold: Luminance threshold for highlight region (0-1)

    Returns:
        Color-transferred image array (H, W, 3) in range [0, 1]

    Raises:
        ValueError: If mode is invalid or images have incorrect dimensions
    """
    source, reference = validate_and_resize_images(source, reference)

    # Calculate luminance using ITU-R BT.601 coefficients
    source_lum = (
        0.299 * source[:, :, 0] + 0.587 * source[:, :, 1] + 0.114 * source[:, :, 2]
    )

    # Create binary mask based on selected tonal region
    # Masks are continuous (0.0 to 1.0) for smooth blending at boundaries
    if mode == "full":
        mask = np.ones_like(source_lum)
    elif mode == "shadows":
        mask = (source_lum < shadow_threshold).astype(float)
    elif mode == "midtones":
        mask = (
            (source_lum >= shadow_threshold) & (source_lum <= highlight_threshold)
        ).astype(float)
    elif mode == "highlights":
        mask = (source_lum > highlight_threshold).astype(float)
    else:
        raise ValueError(f"Invalid mode: {mode}")

    # Expand mask to 3 channels for RGB processing
    mask = np.stack([mask, mask, mask], axis=2)

    # Apply histogram matching to all channels
    matched = np.empty_like(source)
    for i in range(3):
        matched[:, :, i] = exposure.match_histograms(
            source[:, :, i], reference[:, :, i]
        )

    # Blend original and matched images using mask
    # Original image shows where mask=0, matched where mask=1
    result = source * (1 - mask) + matched * mask

    return np.clip(result, 0, 1).astype(source.dtype)