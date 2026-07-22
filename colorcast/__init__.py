"""ColorCast - Advanced Color Transfer Suite

A sophisticated toolkit for color and style transfer between images.

ColorCast provides multiple algorithms including:
- Histogram matching
- Mean/standard deviation transfer
- LUT-based curve adjustments
- Selective regional color transfer (shadows/midtones/highlights)

The package supports:
- RGB, grayscale, and RGBA images (automatically converted to RGB)
- Plugin architecture for custom transfer methods
- LRU caching for performance optimization
- Batch processing with parallel execution
- Configuration management

Version: 2.2.0
Author: Michail Semoglou
License: MIT
"""

__version__ = "2.2.0"
__author__ = "Michail Semoglou"
__email__ = "m.semoglou@tongji.edu.cn"
__license__ = "MIT"

from colorcast.processing.image_loader import load_image, ensure_rgb, save_image
from colorcast.processing.transfer_methods import (
    match_histograms_multichannel,
    color_transfer_meanstd,
    color_transfer_lab,
    lut_transfer_with_curve,
    selective_color_transfer,
)
from colorcast.processing.blending import blend_images
from colorcast.processing.curves import apply_curve
from colorcast.utils.config import ColorCastConfig
from colorcast.processing.registry import registry

__all__ = [
    "load_image",
    "save_image",
    "ensure_rgb",
    "match_histograms_multichannel",
    "color_transfer_meanstd",
    "color_transfer_lab",
    "lut_transfer_with_curve",
    "selective_color_transfer",
    "blend_images",
    "apply_curve",
    "ColorCastConfig",
    "registry",
]
