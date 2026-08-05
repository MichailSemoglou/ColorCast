"""Shared color-space utilities.

This module provides low-level functions for color-space conversions that
are used across multiple analysis and processing modules.
"""

from __future__ import annotations

import numpy as np


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    """Convert sRGB gamma-encoded values to linear light.

    Implements the IEC 61966-2-1 sRGB transfer function (inverse).

    Args:
        rgb: Array of sRGB values in [0, 1] range.

    Returns:
        Array of linear light values in [0, 1] range, same shape as input.

    Note:
        This function is used by both CVD simulation (simulation.py) and
        appearance-space conversion (appearance.py) to ensure consistent
        color handling across the codebase.
    """
    # sRGB uses a piecewise transfer function:
    # - Linear segment for very dark values (≤ 0.04045)
    # - Power curve for brighter values
    linear = np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4,
    )
    return linear
