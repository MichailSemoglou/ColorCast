"""Analysis utilities for ColorCast.

This package provides tools for:
- Comparing different transfer methods using quantitative metrics
- Visualizing transfer results
- Experiment tracking for academic research
"""

from colorcast.analysis.comparison import MethodComparison
from colorcast.analysis.visualization import (
    visualize_transfer_result,
    visualize_method_comparison,
    visualize_color_channels,
    create_side_by_side_comparison,
)

__all__ = [
    'MethodComparison',
    'visualize_transfer_result',
    'visualize_method_comparison',
    'visualize_color_channels',
    'create_side_by_side_comparison',
]