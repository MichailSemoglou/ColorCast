"""Image blending utilities."""

import numpy as np


def blend_images(original: np.ndarray, styled: np.ndarray, intensity: float) -> np.ndarray:
    """
    Blend original and styled images based on intensity.

    Args:
        original: Original image array (H, W, 3) in range [0, 1]
        styled: Styled image array (H, W, 3) in range [0, 1]
        intensity: Blending intensity (0.0 to 1.0), where 0.0 = original,
                  1.0 = fully styled

    Returns:
        Blended image array (H, W, 3) in range [0, 1]
    """
    intensity = np.clip(intensity, 0.0, 1.0)
    blended = original * (1 - intensity) + styled * intensity
    return np.clip(blended, 0.0, 1.0).astype(original.dtype)