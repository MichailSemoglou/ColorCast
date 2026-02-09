"""Image processing modules for ColorCast."""

from colorcast.processing.image_loader import load_image, ensure_rgb
from colorcast.processing.transfer_methods import (
    match_histograms_multichannel,
    color_transfer_meanstd,
    lut_transfer_with_curve,
    selective_color_transfer,
)
from colorcast.processing.curves import apply_curve
from colorcast.processing.blending import blend_images
from colorcast.processing.registry import registry

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
]