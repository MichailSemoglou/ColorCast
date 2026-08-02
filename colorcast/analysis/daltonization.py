"""Daltonization — re-encode lost chromatic information for dichromatic observers.

This module provides Daltonization: it takes the signed
error map produced by :mod:`~colorcast.analysis.error_map` and shifts the lost
chromatic information into perceptual channels that remain functional for the
chosen deficiency type.

Public API
----------
- :func:`apply_daltonization` — correct an image using a pre-computed
  :class:`~colorcast.analysis.error_map.ErrorMap`.
- :func:`daltonize` — convenience wrapper that runs simulation, error analysis,
  and correction in one call.

Lab-space correction
--------------------
The correction operates in the CIE Lab* (a*, b*) chromaticity plane rather
than gamma-encoded sRGB.  For each deficiency type, the lost chromatic
information — measured by the signed (a*, b*) difference in the error map —
is rotated into the surviving chromatic axis:

- ``deuteranopia`` / ``protanopia`` — red-green (a*) error is redirected
  into the blue-yellow (b*) channel.
- ``tritanopia`` — blue-yellow (b*) error is redirected into the red-green
  (a*) channel.

Correction is spatially weighted by the chromaticity error map so
low-error pixels remain largely unchanged, and the original L* channel is
restored to prevent brightness drift.

For the color-vision science behind this module, see the project wiki:
"Color-Vision Background".

References
----------
- Commission Internationale de l'Éclairage. (2004). Colorimetry (3rd ed.)
  (CIE Publication No. 15:2004). CIE.
- Huang, J.-B., Chen, C.-S., Jen, T.-C., & Wang, S.-J. (2009). Image
  recolorization for the colorblind. In Proceedings of the IEEE
  International Conference on Acoustics, Speech and Signal Processing
  (ICASSP), pp. 1161-1164. DOI: 10.1109/ICASSP.2009.4959795.
- Rasche, K., Geist, R., & Westall, J. (2005). Re-coloring images for
  gamuts of lower dimension. Computer Graphics Forum, 24(3), 423-432.
  DOI: 10.1111/j.1467-8659.2005.00867.x.
- Smith, V. C., & Pokorny, J. (1975). Spectral sensitivity of the foveal cone
  photopigments between 400 and 500 nm. Vision Research, 15(2), 161-171.
  DOI: 10.1016/0042-6989(75)90203-5.
"""

from __future__ import annotations

import numpy as np
from skimage import color as skcolor

from colorcast.analysis.error_map import ErrorMap, get_error_map
from colorcast.processing.image_loader import normalize_to_float32

# ---------------------------------------------------------------------------
# Lab-space shift coefficients
# ---------------------------------------------------------------------------
#
# Each deficiency maps a 2D (a*, b*) error vector to a 2D correction vector
# that redirects lost chromaticity into the surviving axis.
#
#   correction_a = coeff[0] * error_a + coeff[1] * error_b
#   correction_b = coeff[2] * error_a + coeff[3] * error_b
#
# For red-green deficiencies the a* error is the dominant loss; it is
# redirected into b*.  For blue-yellow deficiency the b* error is
# redirected into a*.

_LAB_SHIFT_COEFFICIENTS: dict[str, tuple[float, float, float, float]] = {
    # Deuteranopia (M-cone missing): redirect a* → b*
    "deuteranopia": (0.0, 0.0, 0.35, 0.0),
    # Protanopia (L-cone missing): redirect a* → b* — slightly stronger
    # coefficient because L-cone contributes more to the a* axis.
    "protanopia": (0.0, 0.0, 0.40, 0.0),
    # Tritanopia (S-cone missing): redirect b* → a*
    "tritanopia": (0.0, 0.35, 0.0, 0.0),
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _compute_chromaticity_weight(
    error_map: ErrorMap,
) -> np.ndarray:
    """Derive a per-pixel spatial weight from the chroma error map.

    The chroma_error array is subsampled to keep the percentile calculation
    constant-time regardless of image size.  Weight is ``clamp(ce / p95, 0, 1)``
    so that only pixels with high chromatic error receive the full correction.

    Returns (H, W, 1) float32 array.
    """
    ce = error_map.chroma_error
    stride = max(1, ce.size // 40_000)
    p95 = float(np.percentile(ce.flat[::stride], 95))
    if p95 < 1e-6:
        return np.zeros((*ce.shape, 1), dtype=np.float32)
    weight = np.clip(ce / p95, 0.0, 1.0)
    return weight[:, :, np.newaxis]


# ---------------------------------------------------------------------------
# Convenience end-to-end pipeline
# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------


def apply_daltonization(
    original_img: np.ndarray,
    error_map: ErrorMap,
    deficiency_type: str,
    intensity: float = 1.0,
    *,
    simulated_img: np.ndarray | None = None,
) -> np.ndarray:
    """Apply the Daltonization correction to *original_img*.

    The correction operates in the CIE Lab* (a*, b*) chromaticity plane:
    the lost chromatic information measured by the error map is rotated
    into the surviving chromatic axis for the given deficiency type, and
    applied to the *simulated* image so that the color-blind observer
    perceives the re-encoded information.

    When ``simulated_img`` is None the correction is applied to the
    original image instead (backward-compatible behavior).

    Correction is spatially weighted by the chromaticity error map:
    pixels where the simulation barely changed anything are left nearly
    untouched; pixels where large chromatic information was lost receive
    the strongest shift.  L* is preserved throughout.

    Parameters
    ----------
    original_img :
        Source image, shape (H, W, 3), any numeric dtype.
    error_map :
        Result of :func:`~colorcast.analysis.error_map.get_error_map`.
    deficiency_type :
        One of ``"deuteranopia"``, ``"protanopia"``, or ``"tritanopia"``.
    intensity :
        Global correction strength in [0.0, 1.0].
    simulated_img :
        The dichromatic simulation of ``original_img``.  When provided,
        the correction is applied to this image so the color-blind
        observer sees the re-encoded information.

    Returns
    -------
    np.ndarray
        Daltonized image, float32, shape (H, W, 3), values in [0, 1].

    Raises
    ------
    ValueError
        If ``deficiency_type`` is not supported.
    """
    if deficiency_type not in _LAB_SHIFT_COEFFICIENTS:
        raise ValueError(
            f"Unknown deficiency type {deficiency_type!r}. "
            f"Supported: {sorted(_LAB_SHIFT_COEFFICIENTS)}"
        )

    intensity = float(np.clip(intensity, 0.0, 1.0))
    base = normalize_to_float32(simulated_img if simulated_img is not None else original_img)
    if intensity < 1e-6:
        return base.copy()

    base_lab = skcolor.rgb2lab(base)

    weight = _compute_chromaticity_weight(error_map)
    if float(np.max(weight)) < 1e-6:
        return base.copy()

    # Signed (a*, b*) error: the chromatic information lost in simulation
    ab_error = error_map.signed_chroma_ab  # (H, W, 2)

    # Lab-space shift coefficients
    caa, cab, cba, cbb = _LAB_SHIFT_COEFFICIENTS[deficiency_type]
    correction_a = caa * ab_error[:, :, 0] + cab * ab_error[:, :, 1]
    correction_b = cba * ab_error[:, :, 0] + cbb * ab_error[:, :, 1]

    # Apply spatial weight and intensity
    w = weight[:, :, 0] * intensity
    base_lab[:, :, 1] = base_lab[:, :, 1] + correction_a * w
    base_lab[:, :, 2] = base_lab[:, :, 2] + correction_b * w

    corrected_rgb = skcolor.lab2rgb(base_lab)
    return np.clip(corrected_rgb, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Convenience end-to-end pipeline
# ---------------------------------------------------------------------------


def daltonize(
    original_img: np.ndarray,
    deficiency_type: str,
    intensity: float = 1.0,
) -> np.ndarray:
    """Full Daltonization pipeline: simulate → measure → correct.

    A convenience wrapper that chains all three phases into a single call:

     1. **Color vision simulation**: Generate what the image looks like through the
        dichromat's visual system using the Smith-Pokorny cone model.
     2. **Error map**: Measure the signed chromatic difference between
        original and simulated in both RGB and CIE Lab* space.
     3. **Daltonization**: Re-inject the lost information via the
       appropriate shift matrix, weighted by the per-pixel chroma error, with
       luminance preservation.

    Parameters
    ----------
    original_img :
        Source image, shape (H, W, 3), any numeric dtype.
    deficiency_type :
        One of ``"deuteranopia"``, ``"protanopia"``, or ``"tritanopia"``.
    intensity :
        Correction strength in [0.0, 1.0].

    Returns
    -------
    np.ndarray
        Daltonized image, float32, shape (H, W, 3), values in [0, 1].

    Example
    -------
    >>> from colorcast.analysis.daltonization import daltonize
    >>> corrected = daltonize(original, "deuteranopia", intensity=0.9)
    """
    from colorcast.processing.image_loader import normalize_to_float32
    from colorcast.processing.simulation import ColorBlindSimulator

    intensity = float(np.clip(intensity, 0.0, 1.0))
    if deficiency_type not in _LAB_SHIFT_COEFFICIENTS:
        raise ValueError(
            f"Unknown deficiency type {deficiency_type!r}. "
            f"Supported: {sorted(_LAB_SHIFT_COEFFICIENTS)}"
        )
    if intensity < 1e-6:
        return normalize_to_float32(original_img)

    simulated = ColorBlindSimulator().transform_color_space(original_img, deficiency_type)  # type: ignore[arg-type]
    em = get_error_map(original_img, simulated)
    return apply_daltonization(
        original_img, em, deficiency_type, intensity=intensity, simulated_img=simulated
    )
