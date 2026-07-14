"""Simulate dichromatic color vision in linear sRGB.

This module provides Phase 1 of the accessibility pipeline: it transforms an
RGB image to approximate how a person with Protanopia, Deuteranopia, or
Tritanopia would perceive it.

Public API
----------
- :class:`ColorBlindSimulator` — vectorized simulator for the three main
  dichromatic deficiencies.
- :data:`DeficiencyType` — ``Literal["protanopia", "deuteranopia",
  "tritanopia"]``.
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

from typing import Literal

import numpy as np
from colorcast.processing.image_loader import normalize_to_float32

# Type alias used in public method signatures.
DeficiencyType = Literal["protanopia", "deuteranopia", "tritanopia"]


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


class ColorBlindSimulator:
    """Simulate the three main dichromatic colour-vision deficiencies.

    All simulation methods are fully vectorized (no Python loops) so they
    scale to high-resolution images efficiently.  Every public method returns
    a **float32** NumPy array in [0, 1], making it straightforward to
    compute per-pixel error maps in a Phase-2 analysis step::

        original  = image_array.astype(np.float32) / 255
        simulated = simulator.transform_color_space(original, "protanopia")
        error_map = original - simulated   # signed difference, shape (H,W,3)
    """

    # ------------------------------------------------------------------ #
    # Viénot 1999 single-matrix linear RGB transformations.             #
    #                                                                    #
    # These are the combined RGB → LMS → projection → LMS → RGB matrices #
    # dumped from DaltonLens-Python (linearRGB colour interpolation).    #
    # Each row sums to 1, preserving achromatic whites and grays.        #
    # ------------------------------------------------------------------ #

    # Protanopia — L cone absent ("red-blind")
    _PROTAN_RGB: np.ndarray = np.array(
        [
            [0.10889, 0.89111, -0.00000],
            [0.10889, 0.89111, 0.00000],
            [0.00447, -0.00447, 1.00000],
        ],
        dtype=np.float64,
    )

    # Deuteranopia — M cone absent ("green-blind")
    _DEUTAN_RGB: np.ndarray = np.array(
        [
            [0.29031, 0.70969, -0.00000],
            [0.29031, 0.70969, -0.00000],
            [-0.02197, 0.02197, 1.00000],
        ],
        dtype=np.float64,
    )

    # ------------------------------------------------------------------ #
    # Brettel 1997 two-half-plane tritanopia transformation.            #
    #                                                                    #
    # Tritanopia cannot be approximated well by a single matrix; the     #
    # accurate model projects onto one of two planes depending on which  #
    # side of the neutral diagonal the colour falls.  The matrices and   #
    # separation-plane normal below are the pre-computed linear RGB      #
    # forms published by DaltonLens.                                     #
    # ------------------------------------------------------------------ #

    _TRITAN_PLANE1_RGB: np.ndarray = np.array(
        [
            [1.01354, 0.14268, -0.15622],
            [-0.01181, 0.87561, 0.13619],
            [0.07707, 0.81208, 0.11085],
        ],
        dtype=np.float64,
    )

    _TRITAN_PLANE2_RGB: np.ndarray = np.array(
        [
            [0.93337, 0.19999, -0.13336],
            [0.05809, 0.82565, 0.11626],
            [-0.37923, 1.13825, 0.24098],
        ],
        dtype=np.float64,
    )

    # Separation-plane normal in linear RGB.  A pixel selects Plane 1 when
    # dot(rgb, _TRITAN_SEPARATION_RGB) >= 0 and Plane 2 otherwise.
    _TRITAN_SEPARATION_RGB: np.ndarray = np.array(
        [7.92482, -5.66475, -2.26007],
        dtype=np.float64,
    )

    # Lookup table — maps deficiency name → its linear RGB matrix.
    # Tritanopia is handled separately because it needs two planes.
    _PROJECTION: dict[str, np.ndarray] = {
        "protanopia": _PROTAN_RGB,
        "deuteranopia": _DEUTAN_RGB,
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
        2. Flattens (H, W, 3) → (N, 3) for fully vectorized batch ops.
        3. Gamma-decodes nonlinear sRGB → linear RGB.
        4. Applies the appropriate linear RGB transformation:
           Viénot 1999 for protanopia/deuteranopia, Brettel 1997 two-plane
           for tritanopia.
        5. Gamma-encodes linear RGB → nonlinear sRGB and clips to [0, 1].
        6. Returns the result as a **float32** (H, W, 3) array.

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
            ValueError: If *deficiency_type* is not recognised.
        """
        valid_types = ["protanopia", "deuteranopia", "tritanopia"]
        if deficiency_type not in valid_types:
            raise ValueError(
                f"Unknown deficiency type: {deficiency_type!r}. "
                f"Choose from {valid_types}."
            )

        # Step 1: convert to float32, normalize to [0, 1] --------------------
        img = normalize_to_float32(image_array)

        h, w, _ = img.shape

        # Step 2: flatten to (N, 3) for vectorized matrix operations ----------
        pixels = img.reshape(-1, 3).astype(np.float64)  # upcast for precision

        # Step 3: gamma-decode nonlinear sRGB → linear RGB --------------------
        linear_pixels = _srgb_to_linear(pixels)

        # Step 4: apply the deficiency-specific linear RGB transformation -----
        if deficiency_type == "tritanopia":
            # Brettel 1997: choose one of two projection planes per pixel.
            # The separation plane is evaluated in linear RGB; the matrices
            # already include the RGB → LMS → projection → LMS → RGB chain.
            dot = linear_pixels @ self._TRITAN_SEPARATION_RGB  # (N,)
            plane1_mask = dot >= 0

            rgb_linear_sim = np.empty_like(linear_pixels)
            if np.any(plane1_mask):
                rgb_linear_sim[plane1_mask] = (
                    linear_pixels[plane1_mask] @ self._TRITAN_PLANE1_RGB.T
                )
            if np.any(~plane1_mask):
                rgb_linear_sim[~plane1_mask] = (
                    linear_pixels[~plane1_mask] @ self._TRITAN_PLANE2_RGB.T
                )
        else:
            projection = self._PROJECTION[deficiency_type]
            rgb_linear_sim = linear_pixels @ projection.T  # (N, 3)

        # Step 5: gamma-encode linear RGB → nonlinear sRGB --------------------
        rgb_sim = _linear_to_srgb(rgb_linear_sim)

        # Step 6: clip, reshape, and downcast back to float32 -----------------
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
