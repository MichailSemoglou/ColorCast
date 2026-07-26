"""Visualization tools for color transfer analysis.

This module provides utilities to visualize and compare color transfer
results, including side-by-side comparisons, histograms, and difference maps.
"""

from typing import Dict, Optional
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def visualize_transfer_result(
    source: np.ndarray,
    reference: np.ndarray,
    result: np.ndarray,
    method_name: str,
    show_histograms: bool = True,
    show_difference: bool = False,
    figsize: Optional[tuple] = None,
) -> plt.Figure:
    """
    Create comprehensive visualization of transfer result.
    
    Shows source, reference, and result images side by side,
    plus optional histograms for each color channel and difference maps.
    
    Args:
        source: Source image (H, W, 3) in range [0, 1]
        reference: Reference image (H, W, 3) in range [0, 1]
        result: Transferred image (H, W, 3) in range [0, 1]
        method_name: Name of transfer method
        show_histograms: Whether to show color histograms
        show_difference: Whether to show difference from reference
        figsize: Optional figure size (width, height)
    
    Returns:
        Matplotlib Figure object
    
    Example:
        >>> fig = visualize_transfer_result(
        ...     source, style, result, "Histogram Matching"
        ... )
        >>> fig.savefig("comparison.png", dpi=150, bbox_inches='tight')
        >>> plt.close(fig)
    """
    if figsize is None:
        if show_histograms:
            figsize = (18, 14) if show_difference else (18, 12)
        else:
            figsize = (15, 6) if show_difference else (15, 5)
    
    # Create figure
    if show_histograms:
        n_rows = 5 if show_difference else 4
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(n_rows, 3, figure=fig, hspace=0.3, wspace=0.2)
    else:
        n_rows = 2 if show_difference else 1
        fig, axes = plt.subplots(n_rows, 3, figsize=figsize)
        if n_rows == 1:
            axes = np.array([axes])
        gs = None
    
    # Helper to get subplot
    def subplot(row, col, rowspan=1, colspan=1):
        if gs is not None:
            return fig.add_subplot(gs[row:row+rowspan, col:col+colspan])
        else:
            return axes[row, col]
    
    # Images (top row)
    ax_source = subplot(0, 0)
    ax_ref = subplot(0, 1)
    ax_result = subplot(0, 2)
    
    ax_source.imshow(source)
    ax_source.set_title('Source Image', fontsize=12, fontweight='bold')
    ax_source.axis('off')
    
    ax_ref.imshow(reference)
    ax_ref.set_title('Reference (Style) Image', fontsize=12, fontweight='bold')
    ax_ref.axis('off')
    
    ax_result.imshow(result)
    ax_result.set_title(f'Result ({method_name})', fontsize=12, fontweight='bold')
    ax_result.axis('off')
    
    # Difference from reference (if requested)
    if show_difference:
        ax_diff = subplot(1, 0, colspan=3)
        diff = np.abs(result - reference)
        ax_diff.imshow(diff)
        ax_diff.set_title('Difference from Reference', fontsize=12)
        ax_diff.axis('off')
        fig.colorbar(ax_diff.images[0], ax=ax_diff, fraction=0.046, pad=0.04)
    
    # Histograms
    if show_histograms:
        colors = ['r', 'g', 'b']
        channel_names = ['Red', 'Green', 'Blue']
        start_row = 2 if show_difference else 1
        
        for i, (color, name) in enumerate(zip(colors, channel_names)):
            ax_hist = subplot(start_row + i, 0, colspan=3)
            
            for img, label in [
                (source, 'Source'),
                (reference, 'Reference'),
                (result, 'Result')
            ]:
                hist, bins = np.histogram(img[:, :, i], bins=50, range=(0, 1))
                ax_hist.plot(bins[:-1], hist, color=color, alpha=0.7, label=label, linewidth=2)
            
            ax_hist.set_title(f'{name} Channel Histogram', fontsize=11, fontweight='bold')
            ax_hist.set_xlabel('Intensity', fontsize=10)
            ax_hist.set_ylabel('Count', fontsize=10)
            ax_hist.legend(fontsize=9)
            ax_hist.grid(True, alpha=0.3)
    
    fig.suptitle('Color Transfer Analysis', fontsize=16, fontweight='bold', y=0.995)
    return fig


def visualize_method_comparison(
    images: Dict[str, np.ndarray],
    reference: np.ndarray,
    show_difference: bool = True,
    show_histograms: bool = False,
    figsize: Optional[tuple] = None,
) -> plt.Figure:
    """
    Visualize comparison of multiple transfer methods.
    
    Displays results from multiple methods side by side for easy comparison,
    with optional difference maps from reference.
    
    Args:
        images: Dict of {method_name: result_image}
        reference: Reference image (H, W, 3) in range [0, 1]
        show_difference: Whether to show difference from reference
        show_histograms: Whether to show histograms (not recommended for many methods)
        figsize: Optional figure size
    
    Returns:
        Matplotlib Figure object
    
    Example:
        >>> results = {
        ...     'Histogram': hist_result,
        ...     'Mean/Std': meanstd_result,
        ...     'Lab': lab_result,
        ... }
        >>> fig = visualize_method_comparison(results, reference)
        >>> fig.savefig("method_comparison.png", dpi=150, bbox_inches='tight')
        >>> plt.close(fig)
    """
    n_methods = len(images)
    
    if figsize is None:
        n_rows = 2 if show_difference else 1
        figsize = (5 * (n_methods + 1), 5 * n_rows)
    
    fig, axes = plt.subplots(n_rows, n_methods + 1, figsize=figsize)
    if n_rows == 1:
        axes = np.array([axes])
    
    # First column: reference
    axes[0, 0].imshow(reference)
    axes[0, 0].set_title('Reference Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Add reference to bottom row if difference shown
    if show_difference:
        axes[1, 0].axis('off')
    
    # Display each method
    for i, (name, img) in enumerate(images.items(), start=1):
        # Top row: results
        axes[0, i].imshow(img)
        axes[0, i].set_title(name, fontsize=11, fontweight='bold')
        axes[0, i].axis('off')
        
        # Bottom row: difference from reference
        if show_difference:
            diff = np.abs(img - reference)
            axes[1, i].imshow(diff)
            axes[1, i].set_title(f'{name} (Difference)', fontsize=10)
            axes[1, i].axis('off')
    
    fig.suptitle('Transfer Method Comparison', fontsize=14, fontweight='bold')
    fig.tight_layout()
    return fig


def visualize_color_channels(
    image: np.ndarray,
    title: str = "Color Channel Visualization",
    figsize: tuple = (16, 4),
) -> plt.Figure:
    """
    Visualize individual RGB channels of an image.
    
    Useful for understanding how color transfer affects each channel separately.
    
    Args:
        image: Image (H, W, 3) in range [0, 1]
        title: Figure title
        figsize: Figure size (width, height)
    
    Returns:
        Matplotlib Figure object
    
    Example:
        >>> fig = visualize_color_channels(result, "After Transfer")
        >>> fig.savefig("channels.png", dpi=150)
        >>> plt.close(fig)
    """
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    
    # Full RGB
    axes[0].imshow(image)
    axes[0].set_title('RGB', fontweight='bold')
    axes[0].axis('off')
    
    # Individual channels
    channel_names = ['Red', 'Green', 'Blue']
    for i, name in enumerate(channel_names):
        axes[i+1].imshow(image[:, :, i], cmap='gray')
        axes[i+1].set_title(name, fontweight='bold')
        axes[i+1].axis('off')
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    fig.tight_layout()
    return fig


def create_side_by_side_comparison(
    image1: np.ndarray,
    image2: np.ndarray,
    label1: str = "Before",
    label2: str = "After",
    figsize: tuple = (12, 5),
) -> plt.Figure:
    """
    Create simple side-by-side comparison of two images.
    
    Args:
        image1: First image (H, W, 3) in range [0, 1]
        image2: Second image (H, W, 3) in range [0, 1]
        label1: Label for first image
        label2: Label for second image
        figsize: Figure size
    
    Returns:
        Matplotlib Figure object
    
    Example:
        >>> fig = create_side_by_side_comparison(source, result)
        >>> fig.savefig("before_after.png", dpi=150)
        >>> plt.close(fig)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    ax1.imshow(image1)
    ax1.set_title(label1, fontsize=13, fontweight='bold')
    ax1.axis('off')
    
    ax2.imshow(image2)
    ax2.set_title(label2, fontsize=13, fontweight='bold')
    ax2.axis('off')
    
    fig.tight_layout()
    return fig