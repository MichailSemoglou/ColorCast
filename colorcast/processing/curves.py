"""Tone curve utilities for color transfer."""

from typing import Literal
import numpy as np


def apply_curve(values: np.ndarray, curve_type: Literal["linear", "s-curve", "contrast"] = "linear") -> np.ndarray:
    """
    Apply tone curve to values.

    Args:
        values: Input values (typically in range [0, 1])
        curve_type: Type of curve to apply ('linear', 's-curve', 'contrast')

    Returns:
        Values with curve applied
    """
    if curve_type == "linear":
        return values
    elif curve_type == "s-curve":
        # S-curve for smooth midtone enhancement
        return 0.5 + 0.5 * np.sin(np.pi * (values - 0.5))
    elif curve_type == "contrast":
        # Power curve for increased contrast
        return np.power(values, 0.8)
    else:
        raise ValueError(f"Unknown curve type: {curve_type}")