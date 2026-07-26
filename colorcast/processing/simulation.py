"""Simulate dichromatic color vision in linear sRGB.

This module provides dichromatic color vision simulation: it transforms an
RGB image to approximate how a person with Protanopia, Deuteranopia, or
Tritanopia would perceive it.

Public API
----------
- :class:`ColorBlindSimulator` — vectorized simulator for the three main
  dichromatic deficiencies.
- :data:`DeficiencyType` — ``Literal["protanopia", "deuteranopia",
  "tritanopia"]``.
- :data:`SUPPORTED_DEFICIENCIES`: runtime tuple of the supported deficiency
  names, derived from :data:`DeficiencyType`.
- ``simulate_protanopia``, ``simulate_deuteranopia``,
  ``simulate_tritanopia`` — convenience wrappers on
  :meth:`ColorBlindSimulator.transform_color_space`.

Pipeline
--------
1. Normalize input to float32 in [0, 1] (nonlinear sRGB assumed).
2. Gamma-decode sRGB to linear RGB.
3. Flatten to (N, 3) for vectorized matrix operations.
4. Apply a deficiency-specific 3×3 matrix in linear RGB space.
   Protanopia and Deuteranopia use the single-matrix method of
   Viénot, Brettel, & Mollon (1999). Tritanopia uses the two-half-plane
   construction of Brettel, Viénot, & Mollon (1997).
5. Gamma-encode linear RGB back to sRGB.
6. Clip to [0, 1] and return a float32 (H, W, 3) array.

The linear RGB matrices are the combined RGB → LMS → projection → LMS → RGB
forms published by DaltonLens. Each row sums to 1, so achromatic whites and
grays are preserved by construction.

For the color-vision science behind this module, see the project wiki:
"Color-Vision Background".

References
----------
- Brettel, H., Viénot, F., & Mollon, J. D. (1997). Computerized simulation of
  color appearance for dichromats. Journal of the Optical Society of America A,
  14(10), 2647-2655. DOI: 10.1364/josaa.14.002647.
- DaltonLens. (2021). Accurate SVG filters for color blindness simulation.
  https://daltonlens.org/cvd-simulation-svg-filters/.
- Viénot, F., Brettel, H., & Mollon, J. D. (1999). Digital video colourmaps
  for checking the legibility of displays by dichromats.
  Color Research & Application, 24(4), 243-252.
  DOI: 10.1002/(SICI)1520-6378(199908)24:4<243::AID-COL5>3.0.CO;2-3.
"""

from __future__ import annotations

from typing import Callable, ClassVar, Literal, get_args

import numpy as np
from colorcast.processing.image_loader import normalize_to_float32

# Type alias used in public method signatures. This Literal is the single
# source of truth for the supported deficiencies; Python 3.10 cannot unpack a
# tuple into Literal[...], so the alias comes first and the runtime tuple is
# derived from it.
DeficiencyType = Literal["protanopia", "deuteranopia", "tritanopia"]

# Deficiency names accepted at runtime, derived from DeficiencyType so that
# validation and static typing cannot drift apart.
SUPPORTED_DEFICIENCIES: tuple[str, ...] = get_args(DeficiencyType)


def _srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    """Convert nonlinear sRGB values in [0, 1] to linear RGB.

    Uses the IEC 61966-2-1 sRGB transfer function.
    """
    return np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        np.power((rgb + 0.055) / 1.055, 2.4),
    )


def _linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    """Convert linear RGB values to nonlinear sRGB in [0, 1].

    Uses the IEC 61966-2-1 sRGB transfer function.  Values below zero are
    clipped before the power operation to avoid NaN warnings from
    out-of-gamut linear values; the final result is clipped to [0, 1] by
    the caller.
    """
    return np.where(
        linear <= 0.0031308,
        linear * 12.92,
        1.055 * np.power(np.maximum(linear, 0.0), 1.0 / 2.4) - 0.055,
    )


def _tritan_projection(linear_pixels: np.ndarray) -> np.ndarray:
    """Project (N, 3) linear-RGB pixels for tritanopia (Brettel 1997).

    Tritanopia cannot be approximated well by a single matrix; each pixel is
    projected onto one of two half-planes depending on which side of the
    separation plane it falls. Both projections are computed and merged with
    ``np.where`` at negligible extra cost. Uses the plane matrices defined
    on :class:`ColorBlindSimulator`.
    """
    plane1: np.ndarray = linear_pixels @ ColorBlindSimulator._TRITAN_PLANE1_RGB.T
    plane2: np.ndarray = linear_pixels @ ColorBlindSimulator._TRITAN_PLANE2_RGB.T
    on_plane1 = (linear_pixels @ ColorBlindSimulator._TRITAN_SEPARATION_RGB) >= 0
    return np.where(on_plane1[:, np.newaxis], plane1, plane2)


class ColorBlindSimulator:
    """Simulate the three main dichromatic colour-vision deficiencies.

    All simulation methods are fully vectorized (no Python loops) so they
    scale to high-resolution images efficiently.  Every public method returns
    a **float32** NumPy array in [0, 1], making it straightforward to
    compute per-pixel error maps in a downstream analysis step::

        original  = image_array.astype(np.float32) / 255
        simulated = simulator.transform_color_space(original, "protanopia")
        error_map = original - simulated   # signed difference, shape (H,W,3)
    """

    # L cone absent ("red-blind"). Single-matrix model of Viénot, Brettel &
    # Mollon (1999), combined linear-RGB form published by DaltonLens (2021).
    # Rows sum to 1, so achromatic whites and grays are preserved.
    _PROTAN_RGB: np.ndarray = np.array(
        [
            [0.10889, 0.89111, -0.00000],
            [0.10889, 0.89111, 0.00000],
            [0.00447, -0.00447, 1.00000],
        ],
        dtype=np.float64,
    )

    # M cone absent ("green-blind"). Single-matrix model of Viénot, Brettel &
    # Mollon (1999), combined linear-RGB form published by DaltonLens (2021).
    # Rows sum to 1, so achromatic whites and grays are preserved.
    _DEUTAN_RGB: np.ndarray = np.array(
        [
            [0.29031, 0.70969, -0.00000],
            [0.29031, 0.70969, -0.00000],
            [-0.02197, 0.02197, 1.00000],
        ],
        dtype=np.float64,
    )

    # S cone absent ("blue-blind"), first half-plane. Two-plane model of
    # Brettel, Viénot & Mollon (1997), combined linear-RGB form published by
    # DaltonLens (2021).
    _TRITAN_PLANE1_RGB: np.ndarray = np.array(
        [
            [1.01354, 0.14268, -0.15622],
            [-0.01181, 0.87561, 0.13619],
            [0.07707, 0.81208, 0.11085],
        ],
        dtype=np.float64,
    )

    # S cone absent ("blue-blind"), second half-plane. Two-plane model of
    # Brettel, Viénot & Mollon (1997), combined linear-RGB form published by
    # DaltonLens (2021).
    _TRITAN_PLANE2_RGB: np.ndarray = np.array(
        [
            [0.93337, 0.19999, -0.13336],
            [0.05809, 0.82565, 0.11626],
            [-0.37923, 1.13825, 0.24098],
        ],
        dtype=np.float64,
    )

    # Separation-plane normal in linear RGB. A pixel selects Plane 1 when
    # dot(rgb, _TRITAN_SEPARATION_RGB) >= 0 and Plane 2 otherwise.
    _TRITAN_SEPARATION_RGB: np.ndarray = np.array(
        [7.92482, -5.66475, -2.26007],
        dtype=np.float64,
    )

    # Strategy registry: maps each name in SUPPORTED_DEFICIENCIES to either a
    # single linear-RGB projection matrix or a callable that transforms an
    # (N, 3) linear-RGB array. A new deficiency needs one entry here plus its
    # name in DeficiencyType.
    _PROJECTION: ClassVar[
        dict[str, np.ndarray | Callable[[np.ndarray], np.ndarray]]
    ] = {
        "protanopia": _PROTAN_RGB,
        "deuteranopia": _DEUTAN_RGB,
        "tritanopia": _tritan_projection,
    }

    def transform_color_space(
        self,
        image_array: np.ndarray,
        deficiency_type: DeficiencyType,
    ) -> np.ndarray:
        """Simulate *deficiency_type* colour-vision loss on *image_array*.

        This is the unified, modular entry point for all three dichromacy
        simulations.  Internally it:

        1. Converts the input to a **float32** array normalized to [0, 1].
        2. Validates the (H, W, 3) shape, raising ``ValueError`` otherwise.
        3. Flattens (H, W, 3) → (N, 3) for fully vectorized batch ops.
        4. Gamma-decodes nonlinear sRGB → linear RGB.
        5. Applies the appropriate linear RGB transformation:
           Viénot 1999 for protanopia/deuteranopia, Brettel 1997 two-plane
           for tritanopia.
        6. Gamma-encodes linear RGB → nonlinear sRGB and clips to [0, 1].
        7. Returns the result as a **float32** (H, W, 3) array.

        Returning float32 (not saving to disk) is intentional: the caller
        can directly compute a signed error / difference map::

            error_map = original_float32 - simulated_float32

        Args:
            image_array: RGB image, shape (H, W, 3), any numeric dtype.
            deficiency_type: One of ``"protanopia"``, ``"deuteranopia"``,
                or ``"tritanopia"``.

        Returns:
            Simulated image as float32 in [0, 1], shape (H, W, 3).

        Raises:
            ValueError: If *deficiency_type* is not one of
                :data:`SUPPORTED_DEFICIENCIES`.
            ValueError: If *image_array* is not an (H, W, 3) RGB image.
        """
        if deficiency_type not in SUPPORTED_DEFICIENCIES:
            raise ValueError(
                f"Unknown deficiency type: {deficiency_type!r}. "
                f"Choose from {list(SUPPORTED_DEFICIENCIES)}."
            )

        img = normalize_to_float32(image_array)

        h, w, _ = img.shape
        pixels = img.reshape(-1, 3).astype(np.float64)  # upcast for precision
        linear_pixels = _srgb_to_linear(pixels)

        strategy = self._PROJECTION[deficiency_type]
        if callable(strategy):
            rgb_linear_sim = strategy(linear_pixels)
        else:
            rgb_linear_sim = linear_pixels @ strategy.T

        rgb_sim = _linear_to_srgb(rgb_linear_sim)
        rgb_sim = np.clip(rgb_sim, 0.0, 1.0).astype(np.float32)
        return rgb_sim.reshape(h, w, 3)

    # -- Convenience wrappers (kept for back-compat with existing GUI code) -- #

    def simulate_protanopia(self, image_array: np.ndarray) -> np.ndarray:
        """Convenience wrapper — simulate L-cone (red) blindness."""
        return self.transform_color_space(image_array, "protanopia")

    def simulate_deuteranopia(self, image_array: np.ndarray) -> np.ndarray:
        """Convenience wrapper — simulate M-cone (green) blindness."""
        return self.transform_color_space(image_array, "deuteranopia")

    def simulate_tritanopia(self, image_array: np.ndarray) -> np.ndarray:
        """Convenience wrapper — simulate S-cone (blue) blindness."""
        return self.transform_color_space(image_array, "tritanopia")
