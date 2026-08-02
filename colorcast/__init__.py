"""ColorCast — color and style transfer between images.

Algorithms include histogram matching, mean/standard deviation transfer,
Lab color space transfer (Reinhard), LUT-based curve adjustments, and
selective regional transfer (shadows/midtones/highlights). The package
also provides colour-blindness simulation and Daltonization correction.

The package supports:
- RGB, grayscale, and RGBA images (automatically converted to RGB)
- Plugin architecture for custom transfer methods
- Configurable LRU caching
- Batch processing with parallel execution
- Configuration management

Version: 2.5.0
Author: Michail Semoglou
License: MIT
"""

from colorcast._version import __version__

__author__ = "Michail Semoglou"
__email__ = "m.semoglou@tongji.edu.cn"
__license__ = "MIT"

from colorcast.analysis.comparison import MethodComparison
from colorcast.analysis.daltonization import daltonize
from colorcast.analysis.error_map import ErrorMap, get_error_map
from colorcast.processing.blending import blend_images
from colorcast.processing.curves import apply_curve
from colorcast.processing.image_loader import ensure_rgb, load_image, save_image
from colorcast.processing.registry import registry
from colorcast.processing.simulation import ColorBlindSimulator
from colorcast.processing.transfer_methods import (
    color_transfer_lab,
    color_transfer_meanstd,
    lut_transfer_with_curve,
    match_histograms_multichannel,
    selective_color_transfer,
)
from colorcast.utils.config import ColorCastConfig

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
    "__version__",
    "ColorCastConfig",
    "registry",
    "ColorBlindSimulator",
    "daltonize",
    "get_error_map",
    "ErrorMap",
    "MethodComparison",
]
