"""Analysis utilities for ColorCast.

This package provides tools for:
- Comparing different transfer methods using quantitative metrics
- Visualizing transfer results
- Experiment tracking for academic research
- Phase 2: Error map / difference analysis for colour-blindness simulation
- Phase 3: Daltonization — re-encoding lost chromatic information
"""

from colorcast.analysis.comparison import MethodComparison
from colorcast.analysis.visualization import (
    visualize_transfer_result,
    visualize_method_comparison,
    visualize_color_channels,
    create_side_by_side_comparison,
)
from colorcast.analysis.error_map import (
    ErrorMap,
    get_error_map,
    plot_error_heatmap,
    summarize_error_map,
)
from colorcast.analysis.daltonization import (
    apply_daltonization,
    daltonize,
)

__all__ = [
    'MethodComparison',
    'visualize_transfer_result',
    'visualize_method_comparison',
    'visualize_color_channels',
    'create_side_by_side_comparison',
    'ErrorMap',
    'get_error_map',
    'plot_error_heatmap',
    'summarize_error_map',
    'apply_daltonization',
    'daltonize',
]