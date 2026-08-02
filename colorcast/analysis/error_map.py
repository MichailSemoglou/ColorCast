"""Compute error maps between an original image and its dichromatic simulation.

This module computes error maps: it measures the
per-pixel and chromaticity-only difference between an original RGB image and
the output of :class:`~colorcast.processing.simulation.ColorBlindSimulator`.

Public API
----------
- :class:`ErrorMap` — named tuple that holds signed, absolute, and
  chromaticity-only error arrays, plus cached CIE-Lab* metadata.
- :func:`get_error_map` — compute the full error analysis between two images.
- :func:`plot_error_heatmap` — render the error analysis as a Matplotlib figure.
- :func:`summarize_error_map` — return scalar summary statistics.

The signed RGB difference ``original - simulated`` is the direct input to the
Daltonization correction step. Chromaticity-only error is measured in the CIE
Lab* (a*, b*) plane after zeroing L*, so luminance changes do not dominate the
accessibility analysis.

For the color-vision science behind this module, see the project wiki:
"Color-Vision Background".

References
----------
- Brettel, H., Viénot, F., & Mollon, J. D. (1997). Computerized simulation
  of color appearance for dichromats. Journal of the Optical Society of
  America A, 14(10), 2647-2655. DOI: 10.1364/josaa.14.002647.
- Commission Internationale de l'Éclairage. (2004). Colorimetry (3rd ed.)
  (CIE Publication No. 15:2004). CIE.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from skimage import color as skcolor

from colorcast.processing.image_loader import normalize_to_float32

# ---------------------------------------------------------------------------
# Public data class
# ---------------------------------------------------------------------------


class ErrorMap(NamedTuple):
    """Container returned by :func:`get_error_map`.

    Attributes
    ----------
    signed : np.ndarray
        Signed per-pixel RGB difference ``original − simulated``,
        shape (H, W, 3), dtype float32, range roughly [−1, 1].
        **This is the direct input to Daltonization.**
        Pass it to :func:`~colorcast.analysis.daltonization.apply_daltonization`
        together with the original image, the deficiency type, and an
        intensity value.

    absolute : np.ndarray
        Absolute magnitude ``|signed|``, shape (H, W, 3), dtype float32,
        range [0, 1].  Useful for summary statistics and greyscale heatmaps.

    chroma_error : np.ndarray
        Chromaticity-only error, shape (H, W), dtype float32.
        Computed as the CIE-Lab* Euclidean distance in the (a*, b*) plane,
        ignoring luminance (L*).  Range [0, ∞), typically ≤ 100.
        Higher values indicate stronger colour confusion that Daltonization
        must correct.  The spatial
        pattern of ``chroma_error`` tells the corrector *where* and *how
        much* to shift hues.

    chroma_error_dE00 : np.ndarray
        Per-pixel CIEDE2000 color difference, shape (H, W), dtype float32.
        Computed with L* set to 50 (mid-gray) so only chromaticity
        contributes.  More perceptually uniform than the Euclidean
        ``chroma_error``.  Range [0, ∞), typically ≤ 100.

    signed_chroma_ab : np.ndarray
        Signed (a*, b*) difference in Lab* space, shape (H, W, 2),
        dtype float32.  Needed if the correction is applied directly in Lab* rather
        than RGB.

    orig_l_star : np.ndarray
        The L* (lightness) channel of the *original* image in CIE Lab* space,
        shape (H, W), dtype float32.  Cached here so that
        :func:`~colorcast.analysis.daltonization.apply_daltonization` can
        restore original luminance without recomputing ``rgb2lab(original)``.
    """

    signed: np.ndarray
    absolute: np.ndarray
    chroma_error: np.ndarray
    chroma_error_dE00: np.ndarray
    signed_chroma_ab: np.ndarray
    orig_l_star: np.ndarray


# ---------------------------------------------------------------------------
# Core analysis function
# ---------------------------------------------------------------------------


def _compute_chroma_error_dE00(orig_lab: np.ndarray, sim_lab: np.ndarray) -> np.ndarray:
    """Compute CIEDE2000 chromaticity error from Lab* inputs.

    Both ``orig_lab`` and ``sim_lab`` are expected to be arrays with shape
    ``(H, W, 3)`` containing Lab* values in channel order ``(L*, a*, b*)``.
    The function pins the L* channel to ``50`` for both inputs and computes the
    CIEDE2000 distance from the resulting a* and b* values.  The returned
    array has shape ``(H, W)`` and dtype ``float32``.
    """
    lab1 = np.empty(orig_lab.shape, dtype=np.float64)
    lab2 = np.empty(sim_lab.shape, dtype=np.float64)
    lab1[:, :, 0] = 50.0
    lab2[:, :, 0] = 50.0
    lab1[:, :, 1:] = orig_lab[:, :, 1:]
    lab2[:, :, 1:] = sim_lab[:, :, 1:]
    return skcolor.deltaE_ciede2000(lab1, lab2).astype(np.float32)


def get_error_map(
    original_img: np.ndarray,
    simulated_img: np.ndarray,
    compute_dE00: bool = False,
) -> ErrorMap:
    """Compute the full error analysis between *original_img* and *simulated_img*.

    Both images must be float32 (or convertible) in the [0, 1] range and of
    the same spatial dimensions (H, W, 3).  This is automatically satisfied
    when *simulated_img* comes from
    :meth:`~colorcast.processing.simulation.ColorBlindSimulator.transform_color_space`.

    The function is fully vectorized — no Python loops.

    Parameters
    ----------
    original_img :
        The unmodified RGB image, shape (H, W, 3), any numeric dtype.
    simulated_img :
        The dichromat-simulated RGB image, shape (H, W, 3), any numeric dtype.
    compute_dE00 : bool, default False
        When ``True``, the returned :class:`ErrorMap` includes
        ``chroma_error_dE00`` as a per-pixel ``(H, W)`` array of dtype
        ``float32`` computed from the Lab* channels.  When ``False``, that
        field is left unavailable as ``np.nan`` values so callers can still
        unpack the result without a second code path.

    Returns
    -------
    ErrorMap
        Named tuple with ``signed``, ``absolute``, ``chroma_error``,
        ``chroma_error_dE00``, ``signed_chroma_ab``, and ``orig_l_star``
        arrays.  See :class:`ErrorMap` for details.

    Raises
    ------
    ValueError
        If the images differ in shape or are not 3-channel.

    Example
    -------
    >>> from colorcast.processing.simulation import ColorBlindSimulator
    >>> from colorcast.analysis.error_map import get_error_map
    >>> sim = ColorBlindSimulator()
    >>> simulated = sim.transform_color_space(original, "deuteranopia")
    >>> em = get_error_map(original, simulated)
    >>> # Daltonization (see apply_daltonization in daltonization.py):
    >>> from colorcast.analysis.daltonization import apply_daltonization
    >>> corrected = apply_daltonization(original, em, "deuteranopia", intensity=0.8)
    """
    # -- Input normalisation --------------------------------------------------
    orig = normalize_to_float32(np.asarray(original_img))
    sim = normalize_to_float32(np.asarray(simulated_img))

    if orig.shape != sim.shape:
        raise ValueError(f"Shape mismatch: original {orig.shape} vs simulated {sim.shape}.")
    if orig.ndim != 3 or orig.shape[2] != 3:
        raise ValueError(f"Expected (H, W, 3) images; got shape {orig.shape}.")

    # -- 1. Signed difference (RGB) ------------------------------------------
    # The direction is important: positive → original was stronger in that
    # channel.  Daltonization will *add* a fraction of this back to the original.
    signed = orig - sim  # float32, range ≈ [−1, 1]

    # -- 2. Absolute magnitude (RGB) -----------------------------------------
    absolute = np.abs(signed)  # unsigned "heat", float32, range [0, 1]

    # -- 3. Luminance Masking — isolate chromaticity error in Lab* -----------
    # Convert both images to CIE L*a*b* (D65 illuminant, sRGB primaries).
    #   L*        ∈ [0, 100]  — perceived lightness
    #   a*        ∈ [−128, 127] — red(+) / green(−) opponent axis
    #   b*        ∈ [−128, 127] — yellow(+) / blue(−) opponent axis
    #
    # We care only about a* and b* because:
    #   • L* differences are mostly brightness changes, not colour confusion.
    #   • Dichromats' brightness perception is largely intact (especially for
    #     Deuteranopia/Tritanopia).
    #   • The accessibility problem — colours that *look different* to normal
    #     observers but *look the same* to a dichromat — lives entirely in the
    #     (a*, b*) chromaticity plane.
    #
    # Note: skimage.color.rgb2lab expects float32/float64 in [0, 1].
    orig_lab = skcolor.rgb2lab(orig)  # (H, W, 3) float64
    sim_lab = skcolor.rgb2lab(sim)  # (H, W, 3) float64

    # Signed difference in Lab*: keep all three channels for signed_chroma_ab
    diff_lab = orig_lab - sim_lab  # (H, W, 3)

    # Zero out L* — we explicitly ignore luminance change.
    # What remains in a* and b* is the pure chromatic information lost.
    a_diff = diff_lab[:, :, 1].astype(np.float32)  # red–green axis loss
    b_diff = diff_lab[:, :, 2].astype(np.float32)  # blue–yellow axis loss

    signed_chroma_ab = np.stack([a_diff, b_diff], axis=-1)  # (H, W, 2)

    # Euclidean distance in (a*, b*) plane = total chromatic confusion magnitude.
    # This scalar map is the *blueprint* for Daltonization:
    #   • High values → pixels where the dichromat cannot discriminate colours
    #     that a trichromat can; these pixels need the most correction.
    #   • Low values  → pixels where the simulation barely differs; little or
    #     no correction needed.
    # Daltonization uses this map (or signed_chroma_ab) to decide where and how
    # strongly to boost colour contrasts.
    chroma_error = np.sqrt(a_diff**2 + b_diff**2)  # (H, W) float32

    # -- CIEDE2000 chromaticity error -----------------------------------------
    # Compute dE00 with L* pinned to 50 to isolate chromaticity contribution.
    # By default this optional output is disabled, so the result contains
    # NaN values instead of a computed metric.
    chroma_error_dE00 = (
        _compute_chroma_error_dE00(orig_lab, sim_lab)
        if compute_dE00
        else np.full(chroma_error.shape, np.nan, dtype=np.float32)
    )

    # Cache the original L* channel so downstream callers can
    # restore luminance without recomputing rgb2lab on the original image.
    orig_l_star = orig_lab[:, :, 0].astype(np.float32)  # (H, W)

    return ErrorMap(
        signed=signed,
        absolute=absolute,
        chroma_error=chroma_error,
        chroma_error_dE00=chroma_error_dE00,
        signed_chroma_ab=signed_chroma_ab,
        orig_l_star=orig_l_star,
    )


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------


def plot_error_heatmap(
    error_map: ErrorMap,
    original_img: np.ndarray | None = None,
    simulated_img: np.ndarray | None = None,
    title: str = "Colour-Blindness Error Map",
    colormap: str = "magma",
    figsize: tuple[int, int] = (14, 5),
):
    """Render the error analysis as an annotated Matplotlib figure.

    The heatmap uses the *chromaticity* error (Euclidean distance in the (a*, b*)
    plane of Lab* space) to highlight pixels where the simulation diverges in pure
    colour terms, ignoring brightness.  Areas glowing bright on the 'magma' colormap are
    where a dichromat's colour discrimination is most impaired — exactly the
    regions that Daltonization must correct most aggressively.

    Parameters
    ----------
    error_map :
        Result of :func:`get_error_map`.
    original_img :
        Optional original image to display alongside the heatmap.
    simulated_img :
        Optional simulated image to display alongside the heatmap.
    title :
        Figure super-title.
    colormap :
        Matplotlib colormap name for the heatmap (default ``"magma"``).
        ``"hot"``, ``"inferno"``, and ``"YlOrRd"`` also work well.
    figsize :
        Figure size (width, height) in inches.

    Returns
    -------
    matplotlib.figure.Figure
        The Matplotlib figure.  The caller is responsible for saving or
        closing it (e.g. ``fig.savefig(...); plt.close(fig)``).
    """
    import matplotlib.pyplot as plt

    show_images = original_img is not None and simulated_img is not None
    n_cols = 4 if show_images else 2
    fig, axes = plt.subplots(1, n_cols, figsize=figsize)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    col = 0

    # -- Optional: show original and simulated --------------------------------
    if show_images:
        orig_disp = normalize_to_float32(np.asarray(original_img, dtype=np.float32))
        sim_disp = normalize_to_float32(np.asarray(simulated_img, dtype=np.float32))

        axes[col].imshow(orig_disp)
        axes[col].set_title("Original", fontsize=11)
        axes[col].axis("off")
        col += 1

        axes[col].imshow(sim_disp)
        axes[col].set_title("Simulated", fontsize=11)
        axes[col].axis("off")
        col += 1

    # -- Chromaticity error heatmap -------------------------------------------
    # chroma_error is the Euclidean (a*, b*) distance: pure colour loss.
    # Bright pixels = regions of high colour confusion → high priority for
    # Daltonization correction.
    im = axes[col].imshow(error_map.chroma_error, cmap=colormap, interpolation="nearest")
    axes[col].set_title("Chromaticity Error\n(|Δa*|² + |Δb*|²)^½", fontsize=11)
    axes[col].axis("off")
    fig.colorbar(im, ax=axes[col], fraction=0.046, pad=0.04, label="chroma error (a*, b*)")
    col += 1

    # -- Absolute RGB magnitude heatmap (greyscale per-channel) ---------------
    # Sum across R, G, B channels to get a single scalar "total RGB loss" map.
    rgb_magnitude = error_map.absolute.sum(axis=2)  # (H, W)
    im2 = axes[col].imshow(rgb_magnitude, cmap=colormap, interpolation="nearest")
    axes[col].set_title("Total RGB Magnitude\n|ΔR| + |ΔG| + |ΔB|", fontsize=11)
    axes[col].axis("off")
    fig.colorbar(im2, ax=axes[col], fraction=0.046, pad=0.04, label="Σ|ΔRGB|")

    fig.tight_layout()
    return fig


def summarize_error_map(error_map: ErrorMap) -> dict[str, float]:
    """Return scalar summary statistics for the error map.

    Useful for logging, printing, or comparing how severely different
    deficiency types affect a given image.

    Parameters
    ----------
    error_map :
        Result of :func:`get_error_map`.

    Returns
    -------
    dict
        Keys: ``mean_chroma_error``, ``max_chroma_error``,
        ``p95_chroma_error``, ``mean_rgb_error``, ``max_rgb_error``.
    """
    ce = error_map.chroma_error
    ab = error_map.absolute.sum(axis=2)
    return {
        "mean_chroma_error": float(ce.mean()),
        "max_chroma_error": float(ce.max()),
        "p95_chroma_error": float(np.percentile(ce, 95)),
        "mean_rgb_error": float(ab.mean()),
        "max_rgb_error": float(ab.max()),
    }
