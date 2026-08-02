"""Performance benchmarks for ColorCast transfer methods."""

import time

import numpy as np
import pytest

from colorcast import (
    color_transfer_meanstd,
    lut_transfer_with_curve,
    match_histograms_multichannel,
    selective_color_transfer,
)
from colorcast.processing.cache import StyleTransferCache


@pytest.fixture
def small_image():
    """Small image for fast testing (256x256)."""
    return np.random.rand(256, 256, 3).astype(np.float32)


@pytest.fixture
def medium_image():
    """Medium image (1024x1024)."""
    return np.random.rand(1024, 1024, 3).astype(np.float32)


@pytest.fixture
def large_image():
    """Large image (2048x2048)."""
    return np.random.rand(2048, 2048, 3).astype(np.float32)


@pytest.fixture
def style_image():
    """Style image for transfers."""
    return np.random.rand(1024, 1024, 3).astype(np.float32) * 0.5 + 0.25


class TestMethodPerformance:
    """Benchmark individual transfer method performance."""

    def test_histogram_matching_small(self, small_image, style_image, benchmark):
        """Benchmark histogram matching on small image."""
        benchmark(
            match_histograms_multichannel,
            small_image,
            style_image,
        )

    def test_histogram_matching_medium(self, medium_image, style_image, benchmark):
        """Benchmark histogram matching on medium image."""
        benchmark(
            match_histograms_multichannel,
            medium_image,
            style_image,
        )

    def test_mean_std_small(self, small_image, style_image, benchmark):
        """Benchmark mean/std transfer on small image."""
        benchmark(
            color_transfer_meanstd,
            small_image,
            style_image,
        )

    def test_mean_std_medium(self, medium_image, style_image, benchmark):
        """Benchmark mean/std transfer on medium image."""
        benchmark(
            color_transfer_meanstd,
            medium_image,
            style_image,
        )

    def test_lut_linear_small(self, small_image, style_image, benchmark):
        """Benchmark LUT with linear curve on small image."""
        benchmark(
            lut_transfer_with_curve,
            small_image,
            style_image,
            "linear",
        )

    def test_lut_scurve_small(self, small_image, style_image, benchmark):
        """Benchmark LUT with S-curve on small image."""
        benchmark(
            lut_transfer_with_curve,
            small_image,
            style_image,
            "s-curve",
        )

    def test_lut_contrast_small(self, small_image, style_image, benchmark):
        """Benchmark LUT with contrast curve on small image."""
        benchmark(
            lut_transfer_with_curve,
            small_image,
            style_image,
            "contrast",
        )

    def test_selective_full_small(self, small_image, style_image, benchmark):
        """Benchmark selective full transfer on small image."""
        benchmark(
            selective_color_transfer,
            small_image,
            style_image,
            "full",
        )

    def test_selective_shadows_small(self, small_image, style_image, benchmark):
        """Benchmark selective shadows transfer on small image."""
        benchmark(
            selective_color_transfer,
            small_image,
            style_image,
            "shadows",
        )

    def test_selective_midtones_small(self, small_image, style_image, benchmark):
        """Benchmark selective midtones transfer on small image."""
        benchmark(
            selective_color_transfer,
            small_image,
            style_image,
            "midtones",
        )

    def test_selective_highlights_small(self, small_image, style_image, benchmark):
        """Benchmark selective highlights transfer on small image."""
        benchmark(
            selective_color_transfer,
            small_image,
            style_image,
            "highlights",
        )


class TestCachePerformance:
    """Benchmark cache performance."""

    def test_cache_hit_small(self, small_image, style_image, benchmark):
        """Benchmark cache hit on small image."""
        cache = StyleTransferCache(max_size=10)

        # Pre-fill cache
        cache.get_or_compute(
            key="test",
            compute_func=lambda: match_histograms_multichannel(small_image, style_image),
        )

        # Benchmark cache hit
        benchmark(
            cache.get_or_compute,
            key="test",
            compute_func=lambda: match_histograms_multichannel(small_image, style_image),
        )

    def test_cache_miss_small(self, small_image, style_image, benchmark):
        """Benchmark cache miss on small image."""
        cache = StyleTransferCache(max_size=10)

        # Benchmark cache miss
        benchmark(
            cache.get_or_compute,
            key="test",
            compute_func=lambda: match_histograms_multichannel(small_image, style_image),
        )

    def test_cache_effectiveness(self, medium_image, style_image):
        """Measure cache effectiveness."""
        cache = StyleTransferCache(max_size=5)

        # First access - miss
        start = time.time()
        result1 = cache.get_or_compute(
            key="test1",
            compute_func=lambda: match_histograms_multichannel(medium_image, style_image),
        )
        time_miss = time.time() - start

        # Second access - hit
        start = time.time()
        result2 = cache.get_or_compute(
            key="test1",
            compute_func=lambda: match_histograms_multichannel(medium_image, style_image),
        )
        time_hit = time.time() - start

        # Verify same result
        assert np.allclose(result1, result2)

        # Cache should be faster
        speedup = time_miss / time_hit
        assert speedup > 10, f"Cache speedup too low: {speedup:.2f}x"


class TestScalingPerformance:
    """Benchmark performance scaling with image size."""

    def test_histogram_scaling(self, small_image, medium_image, large_image, style_image):
        """Test performance scaling for histogram matching."""
        times = {}

        for name, img in [("small", small_image), ("medium", medium_image)]:
            start = time.time()
            match_histograms_multichannel(img, style_image)
            times[name] = time.time() - start

        # Medium should be slower than small
        assert times["medium"] > times["small"]

        # Calculate scaling factor
        # Medium has 16x more pixels (1024^2 / 256^2 = 16)
        # But due to NumPy vectorization and modern CPU optimization,
        # scaling is sub-linear. Expect 1.5x to 8x slowdown.
        scaling = times["medium"] / times["small"]
        assert 1.2 < scaling < 10, f"Unexpected scaling: {scaling:.2f}x"

    @pytest.mark.skip(
        reason="Micro-benchmarks with time.time() are unreliable; use pytest-benchmark instead"
    )
    def test_mean_std_scaling(self, small_image, medium_image, style_image):
        """Test performance scaling for mean/std transfer."""
        # Warmup run to initialize NumPy/caches
        color_transfer_meanstd(small_image, style_image)

        times = {}

        for name, img in [("small", small_image), ("medium", medium_image)]:
            start = time.time()
            color_transfer_meanstd(img, style_image)
            times[name] = time.time() - start

        # Medium should be slower than small
        assert times["medium"] > times["small"]

        # Calculate scaling factor
        # Due to vectorized operations, expect sub-linear scaling
        scaling = times["medium"] / times["small"]
        assert 1.2 < scaling < 10, f"Unexpected scaling: {scaling:.2f}x"


class TestMemoryPerformance:
    """Test memory efficiency."""

    def test_large_image_processing(self, large_image, style_image):
        """Test processing of large image without memory issues."""
        # This should complete without memory errors
        result = match_histograms_multichannel(large_image, style_image)
        assert result.shape == large_image.shape

    def test_multiple_operations_memory(self, medium_image, style_image):
        """Test multiple operations don't leak memory."""
        import gc

        # Perform many operations
        for _i in range(10):
            result = match_histograms_multichannel(medium_image, style_image)
            assert result.shape == medium_image.shape

        # Force garbage collection
        gc.collect()

        # Should still work
        result = match_histograms_multichannel(medium_image, style_image)
        assert result.shape == medium_image.shape
