"""Analysis utilities for ColorCast.

This package provides tools for:
- Comparing different transfer methods using quantitative metrics
- Visualizing transfer results
- Experiment tracking for academic research
- Error map / difference analysis for colour-blindness simulation
- Daltonization: re-encoding lost chromatic information
- CVD accessibility dashboard: compare deficiencies at once
- Perceptually uniform appearance spaces (ICtCp, CIELAB)
"""

from colorcast.analysis.appearance import (
    AppearanceDelta,
    AppearanceSpace,
    CIELABSpace,
    ICtCpSpace,
    make_appearance_space,
)
from colorcast.analysis.comparison import MethodComparison
from colorcast.analysis.daltonization import (
    apply_daltonization,
    daltonize,
)
from colorcast.analysis.dashboard import (
    DashboardResult,
    compute_dashboard,
    generate_dashboard_report,
)
from colorcast.analysis.error_map import (
    ErrorMap,
    get_error_map,
    plot_error_heatmap,
    summarize_error_map,
)
from colorcast.analysis.visualization import (
    create_side_by_side_comparison,
    visualize_color_channels,
    visualize_method_comparison,
    visualize_transfer_result,
)

__all__ = [
    "AppearanceSpace",
    "AppearanceDelta",
    "CIELABSpace",
    "ICtCpSpace",
    "make_appearance_space",
    "MethodComparison",
    "visualize_transfer_result",
    "visualize_method_comparison",
    "visualize_color_channels",
    "create_side_by_side_comparison",
    "ErrorMap",
    "get_error_map",
    "plot_error_heatmap",
    "summarize_error_map",
    "apply_daltonization",
    "daltonize",
    "DashboardResult",
    "compute_dashboard",
    "generate_dashboard_report",
]
