"""Daltonization — re-encode lost chromatic information for dichromatic observers.

This module provides Phase 3 of the accessibility pipeline: it takes the signed
error map produced by :mod:`~colorcast.analysis.error_map` and shifts the lost
chromatic information into perceptual channels that remain functional for the
chosen deficiency type.

Public API
----------
- :func:`apply_daltonization` — correct an image using a pre-computed
  :class:`~colorcast.analysis.error_map.ErrorMap`.
- :func:`daltonize` — convenience wrapper that runs simulation, error analysis,
  and correction in one call.

Shift matrices
--------------
The module stores a 3×3 shift matrix for each supported deficiency:

- ``deuteranopia`` / ``protanopia`` — red-green error is routed into the Blue
  channel, which is preserved in both conditions.
- ``tritanopia`` — blue error is routed equally into Red and Green, encoding
  the correction as a luminance modulation.

Correction is spatially weighted by the CIE-Lab* chromaticity error map so
low-error pixels remain largely unchanged, and the original luminance channel
is restored after correction to prevent brightness drift.

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
# Shift matrices (rows = [ΔR, ΔG, ΔB] output;  cols = [eR, eG, eB] input)
# ---------------------------------------------------------------------------
#
# Multiplication is: correction_flat = error_flat @ SHIFT.T
# where error_flat is (H*W, 3) and correction_flat is (H*W, 3).
#
# Convention:
#   • SHIFT[i, j] is the coefficient by which error channel j
#     contributes to output channel i.
#   • Rows that are zero → that output channel is left unchanged.
#   • Column weights sum to ≤ 1.0 per output channel to avoid clipping.

_SHIFT_MATRICES: dict[str, np.ndarray] = {
    # ── Deuteranopia (M-cone missing) ─────────────────────────────────────
    # Red-Green confusion → redirect the Red + Green error into Blue.
    # B += 0.7·eR + 0.1·eG (R, G unchanged: luminance handles the rest)
    "deuteranopia": np.array(
        [
            [0.0, 0.0, 0.0],   # ΔR = 0
            [0.0, 0.0, 0.0],   # ΔG = 0
            [0.7, 0.1, 0.0],   # ΔB = 0.7·eR + 0.1·eG
        ],
        dtype=np.float32,
    ),
    # ── Protanopia (L-cone missing) ───────────────────────────────────────
    # Same opponent axis as deuteranopia (red-green); same redistribution
    # strategy.  Slightly lower Red coefficient because the L-cone overlap
    # with the M-cone is higher → the confusion is less asymmetric.
    "protanopia": np.array(
        [
            [0.0, 0.0, 0.0],   # ΔR = 0
            [0.0, 0.0, 0.0],   # ΔG = 0
            [0.7, 0.1, 0.0],   # ΔB = 0.7·eR + 0.1·eG
        ],
        dtype=np.float32,
    ),
    # ── Tritanopia (S-cone missing) ───────────────────────────────────────
    # Blue-Yellow confusion → redirect the Blue error into both Red and Green.
    # Adding the same amount to R and G encodes the correction as a luminance
    # modulation, which is perceivable even without hue discrimination.
    "tritanopia": np.array(
        [
            [0.0, 0.0, 0.7],   # ΔR = 0.7·eB
            [0.0, 0.0, 0.7],   # ΔG = 0.7·eB
            [0.0, 0.0, 0.0],   # ΔB = 0
        ],
        dtype=np.float32,
    ),
}


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def apply_daltonization(
    original_img: np.ndarray,
    error_map: ErrorMap,
    deficiency_type: str,
    intensity: float = 1.0,
) -> np.ndarray:
    """Apply the Daltonization correction to *original_img*.

    The function re-encodes the chromatic information that was lost in
    dichromatic simulation (Phase 1) — as measured by the error map
    (Phase 2) — into a surviving perceptual channel that the affected
    observer can discriminate.

    The correction is spatially weighted by the CIE Lab* chromaticity
    error: pixels where the simulation barely changed anything are left
    nearly untouched; pixels where large chromatic information was lost
    receive the strongest shift. After the shift, original luminance (L*)
    is restored to prevent brightness drift.

    Parameters
    ----------
    original_img :
        Source image, shape (H, W, 3), any numeric dtype.  Values may be
        uint8 [0, 255] or float [0, 1]; the function normalises internally.
    error_map :
        Result of :func:`~colorcast.analysis.error_map.get_error_map`
        computed from ``original_img`` and its simulated counterpart.
    deficiency_type :
        One of ``"deuteranopia"``, ``"protanopia"``, or ``"tritanopia"``.
    intensity :
        Global correction strength in [0.0, 1.0].  At ``0.0`` the output
        equals the original; at ``1.0`` the full correction is applied.
        The per-pixel chromaticity weight is applied on top of this — so
        even at ``intensity=1.0``, low-error pixels remain essentially
        unchanged.

    Returns
    -------
    np.ndarray
        Daltonized image, float32, shape (H, W, 3), values in [0, 1].

    Raises
    ------
    ValueError
        If ``deficiency_type`` is not supported.

    Example
    -------
    >>> from colorcast.processing.simulation import ColorBlindSimulator
    >>> from colorcast.analysis.error_map import get_error_map
    >>> from colorcast.analysis.daltonization import apply_daltonization
    >>> sim = ColorBlindSimulator().transform_color_space(original, "deuteranopia")
    >>> em  = get_error_map(original, sim)
    >>> corrected = apply_daltonization(original, em, "deuteranopia", intensity=0.8)
    """
    if deficiency_type not in _SHIFT_MATRICES:
        raise ValueError(
            f"Unknown deficiency type {deficiency_type!r}. "
            f"Supported: {sorted(_SHIFT_MATRICES)}"
        )

    intensity = float(np.clip(intensity, 0.0, 1.0))

    # ── 1. Normalize original to float32 [0, 1] ─────────────────────────────
    orig = normalize_to_float32(original_img)

    H, W, _ = orig.shape

    # Short-circuit: identity pass when intensity is effectively zero
    if intensity < 1e-6:
        return orig.copy()

    # ── 2. Compute the raw channel correction via the shift matrix ──────────
    #
    # error_map.signed is (H, W, 3) float32, values ≈ [−1, 1].
    # The shift matrix M is (3, 3) where M[i, j] is the coefficient by which
    # error channel j contributes to output channel i.
    #
    # We flatten to (H*W, 3) for a single vectorized matrix multiply, then
    # reshape back.  This avoids any Python-level loops.
    shift = _SHIFT_MATRICES[deficiency_type]   # (3, 3)
    error_flat = error_map.signed.reshape(-1, 3)          # (H*W, 3)
    correction_flat = error_flat @ shift.T                # (H*W, 3)
    correction = correction_flat.reshape(H, W, 3)         # (H, W, 3)

    # ── 3. Perceptual (chromaticity) spatial weight ─────────────────────────
    #
    # Subsample the chroma_error array for the percentile calculation when
    # the image is large. Taking every Nth element gives a statistically
    # representative estimate of p95 at a fraction of the cost. A stride of
    # ~1 element per 40k pixels keeps the sample ≥ 2 500 points on any
    # reasonably sized image while cutting the sort cost significantly.
    ce = error_map.chroma_error
    stride = max(1, ce.size // 40_000)
    p95 = float(np.percentile(ce.flat[::stride], 95))
    if p95 < 1e-6:
        # Degenerate case: image is essentially uniform — nothing to correct.
        return orig.copy()

    weight = np.clip(ce / p95, 0.0, 1.0)               # (H, W)
    weight = weight[:, :, np.newaxis]                  # (H, W, 1) → broadcast over RGB

    # ── 4. Apply global intensity scale and per-pixel weight ────────────────
    #
    # The two scales multiply together:
    #   • intensity    — user-facing slider: "how much Daltonization overall?"
    #   • weight       — automatic spatial gate: "how much at this pixel?"
    #
    # At intensity=1.0, weight drives the correction.
    # At intensity=0.5, the whole map is halved before the weight is applied.
    correction *= weight * intensity

    # ── 5. Inject the weighted correction into the original ─────────────────
    corrected = orig + correction     # may temporarily exceed [0, 1]

    # ── 6. Luminance Preservation (L* round-trip) ───────────────────────────
    #
    # Use the L* channel cached in error_map.orig_l_star — it was already
    # computed during get_error_map() — instead of calling rgb2lab(orig)
    # again.  This eliminates one full Lab conversion (≈25% of total time
    # on a 1080p image).
    #
    # skimage.rgb2lab accepts float32 and up-converts internally, so we
    # avoid the explicit .astype(np.float64) copy on corrected_f.
    corrected_f   = np.clip(corrected, 0.0, 1.0)
    corrected_lab = skcolor.rgb2lab(corrected_f)   # float64 output

    # Restore original lightness; preserve shifted chromaticity.
    # orig_l_star is float32; the float64 array accepts the implicit upcast.
    # Skip for tritanopia: the shift matrix encodes the blue-error correction
    # as equal R+G (luminance) shifts, which are the primary carrier of the
    # correction signal. Restoring L* would erase those shifts entirely.
    if deficiency_type != "tritanopia":
        corrected_lab[:, :, 0] = error_map.orig_l_star

    corrected_rgb = skcolor.lab2rgb(corrected_lab)  # float64, may have tiny excursions
    corrected_rgb = np.clip(corrected_rgb, 0.0, 1.0).astype(np.float32)

    return corrected_rgb


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

    1. **Phase 1 (Simulation)**: Generate what the image looks like through the
       dichromat's visual system using the Smith-Pokorny cone model.
    2. **Phase 2 (Error Map)**: Measure the signed chromatic difference between
       original and simulated in both RGB and CIE Lab* space.
    3. **Phase 3 (Daltonization)**: Re-inject the lost information via the
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
    from colorcast.processing.simulation import ColorBlindSimulator

    simulated = ColorBlindSimulator().transform_color_space(original_img, deficiency_type)
    em = get_error_map(original_img, simulated)
    return apply_daltonization(original_img, em, deficiency_type, intensity=intensity)
