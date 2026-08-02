"""Image processing modules for ColorCast."""

from colorcast.processing.blending import blend_images
from colorcast.processing.curves import apply_curve
from colorcast.processing.image_loader import ensure_rgb, load_image
from colorcast.processing.registry import registry
from colorcast.processing.simulation import ColorBlindSimulator
from colorcast.processing.transfer_methods import (
    color_transfer_meanstd,
    lut_transfer_with_curve,
    match_histograms_multichannel,
    selective_color_transfer,
)

__all__ = [
    "load_image",
    "ensure_rgb",
    "match_histograms_multichannel",
    "color_transfer_meanstd",
    "lut_transfer_with_curve",
    "selective_color_transfer",
    "apply_curve",
    "blend_images",
    "registry",
    "ColorBlindSimulator",
]
