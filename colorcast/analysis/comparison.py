"""Algorithm comparison utilities for ColorCast.

This module provides tools to compare different color transfer methods
using quantitative metrics like PSNR, SSIM, and color distance.
"""

from typing import Dict, Callable, List, Tuple
import numpy as np
from skimage.metrics import structural_similarity as ssim


class MethodComparison:
    """Compare transfer methods using various quality metrics."""
    
    @staticmethod
    def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Compute Peak Signal-to-Noise Ratio.
        
        PSNR measures the quality of reconstruction. Higher values
        indicate better quality. Typically 20-40 dB range for images.
        
        Args:
            img1: First image (H, W, C) in range [0, 1]
            img2: Second image (H, W, C) in range [0, 1]
        
        Returns:
            PSNR value in dB, infinity if images are identical
        
        Example:
            >>> psnr = MethodComparison.compute_psnr(img1, img2)
            >>> print(f"PSNR: {psnr:.2f} dB")
        """
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * np.log10(1.0 / np.sqrt(mse))
    
    @staticmethod
    def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Compute Structural Similarity Index.
        
        SSIM measures perceived quality based on structural information.
        Values range from -1 to 1, with 1 indicating perfect match.
        Values above 0.9 typically indicate good perceptual quality.
        
        Args:
            img1: First image (H, W, C) in range [0, 1]
            img2: Second image (H, W, C) in range [0, 1]
        
        Returns:
            SSIM value in range [-1, 1]
        
        Example:
            >>> ssim_val = MethodComparison.compute_ssim(img1, img2)
            >>> print(f"SSIM: {ssim_val:.4f}")
        """
        return ssim(img1, img2, channel_axis=-1, data_range=1.0)
    
    @staticmethod
    def compute_color_distance(source: np.ndarray, result: np.ndarray) -> float:
        """
        Compute average Euclidean color distance.
        
        Measures how much colors have shifted from original.
        Lower values indicate more subtle color changes.
        
        Args:
            source: Original image (H, W, C) in range [0, 1]
            result: Transferred image (H, W, C) in range [0, 1]
        
        Returns:
            Average color distance (lower = more subtle)
        
        Example:
            >>> dist = MethodComparison.compute_color_distance(source, result)
            >>> print(f"Color distance: {dist:.4f}")
        """
        diff = source - result
        return np.mean(np.sqrt(np.sum(diff ** 2, axis=2)))
    
    @staticmethod
    def compute_histogram_distance(
        img1: np.ndarray,
        img2: np.ndarray,
        bins: int = 256
    ) -> float:
        """
        Compute Earth Mover's Distance between histograms.
        
        Measures difference in color distributions. Lower values
        indicate more similar histograms.
        
        Args:
            img1: First image (H, W, C) in range [0, 1]
            img2: Second image (H, W, C) in range [0, 1]
            bins: Number of histogram bins
        
        Returns:
            Average histogram distance across channels
        """
        distances = []
        for i in range(img1.shape[2]):
            hist1, _ = np.histogram(img1[:, :, i], bins=bins, range=(0, 1))
            hist2, _ = np.histogram(img2[:, :, i], bins=bins, range=(0, 1))
            
            # Normalize histograms
            hist1 = hist1 / hist1.sum()
            hist2 = hist2 / hist2.sum()
            
            # Compute L1 distance
            distance = np.sum(np.abs(hist1 - hist2))
            distances.append(distance)
        
        return np.mean(distances)
    
    def compare_methods(
        self,
        source: np.ndarray,
        reference: np.ndarray,
        methods: Dict[str, Callable],
        include_baseline: bool = True,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple transfer methods with various metrics.
        
        Args:
            source: Source image (H, W, 3) in range [0, 1]
            reference: Reference image (H, W, 3) in range [0, 1]
            methods: Dict of {method_name: transfer_function}
            include_baseline: Whether to include source-to-reference comparison
        
        Returns:
            Dict of metrics for each method
        
        Example:
            >>> comparison = MethodComparison()
            >>> methods = {
            ...     'histogram': match_histograms_multichannel,
            ...     'meanstd': color_transfer_meanstd,
            ... }
            >>> results = comparison.compare_methods(source, reference, methods)
            >>> for method, metrics in results.items():
            ...     print(f"{method}: PSNR={metrics['psnr']:.2f}")
        """
        results = {}
        
        # Include baseline (source vs reference) if requested
        if include_baseline:
            results['baseline'] = {
                'psnr': self.compute_psnr(source, reference),
                'ssim': self.compute_ssim(source, reference),
                'color_distance': self.compute_color_distance(source, reference),
                'histogram_distance': self.compute_histogram_distance(source, reference),
            }
        
        # Compare each method
        for name, method in methods.items():
            result = method(source, reference)
            
            results[name] = {
                'psnr': self.compute_psnr(result, reference),
                'ssim': self.compute_ssim(result, reference),
                'color_distance': self.compute_color_distance(source, result),
                'histogram_distance': self.compute_histogram_distance(result, reference),
            }
        
        return results
    
    def rank_methods(
        self,
        comparison_results: Dict[str, Dict[str, float]],
        primary_metric: str = 'ssim',
        ascending: bool = False,
    ) -> List[Tuple[str, float]]:
        """
        Rank methods by a specific metric.
        
        Args:
            comparison_results: Results from compare_methods()
            primary_metric: Metric to rank by ('psnr', 'ssim', 'color_distance', 'histogram_distance')
            ascending: If True, lower values are better (for distances)
        
        Returns:
            List of (method_name, metric_value) tuples, sorted by rank
        
        Example:
            >>> comparison = MethodComparison()
            >>> results = comparison.compare_methods(source, reference, methods)
            >>> ranking = comparison.rank_methods(results, 'ssim')
            >>> for rank, (name, value) in enumerate(ranking, 1):
            ...     print(f"{rank}. {name}: {value:.4f}")
        """
        return sorted(
            [(name, metrics[primary_metric]) for name, metrics in comparison_results.items()],
            key=lambda x: x[1],
            reverse=not ascending
        )
    
    def generate_comparison_report(
        self,
        comparison_results: Dict[str, Dict[str, float]],
    ) -> str:
        """
        Generate human-readable comparison report.
        
        Args:
            comparison_results: Results from compare_methods()
        
        Returns:
            Formatted report string
        
        Example:
            >>> comparison = MethodComparison()
            >>> results = comparison.compare_methods(source, reference, methods)
            >>> report = comparison.generate_comparison_report(results)
            >>> print(report)
        """
        report = []
        report.append("Color Transfer Method Comparison")
        report.append("=" * 60)
        
        # Header
        report.append(f"{'Method':<25} {'PSNR (dB)':<12} {'SSIM':<10} {'Color Dist':<12} {'Hist Dist':<12}")
        report.append("-" * 60)
        
        # Results
        for method, metrics in comparison_results.items():
            report.append(
                f"{method:<25} "
                f"{metrics['psnr']:>10.2f}  "
                f"{metrics['ssim']:>8.4f}  "
                f"{metrics['color_distance']:>10.4f}  "
                f"{metrics['histogram_distance']:>10.4f}"
            )
        
        # Rankings
        report.append("\nRankings:")
        report.append("-" * 60)
        
        for metric, ascending in [
            ('ssim', False),
            ('psnr', False),
            ('color_distance', True),
            ('histogram_distance', True),
        ]:
            ranking = self.rank_methods(comparison_results, metric, ascending)
            report.append(f"\nBy {metric}:")
            for i, (name, value) in enumerate(ranking[:3], 1):
                report.append(f"  {i}. {name}: {value:.4f}")
        
        return "\n".join(report)
    
    def find_best_method(
        self,
        comparison_results: Dict[str, Dict[str, float]],
        metric: str = 'ssim',
    ) -> Tuple[str, float]:
        """
        Find the best method according to a specific metric.
        
        Args:
            comparison_results: Results from compare_methods()
            metric: Metric to optimize ('psnr', 'ssim', 'color_distance', 'histogram_distance')
        
        Returns:
            Tuple of (method_name, metric_value)
        
        Example:
            >>> comparison = MethodComparison()
            >>> results = comparison.compare_methods(source, reference, methods)
            >>> best_method, best_value = comparison.find_best_method(results, 'ssim')
            >>> print(f"Best method: {best_method} (SSIM={best_value:.4f})")
        """
        ranking = self.rank_methods(comparison_results, metric)
        return ranking[0]