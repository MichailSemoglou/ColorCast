"""Integration tests for ColorCast."""

import numpy as np
import pytest
from pathlib import Path

from colorcast import (
    load_image,
    match_histograms_multichannel,
    color_transfer_meanstd,
    lut_transfer_with_curve,
    selective_color_transfer,
    blend_images,
    registry,
)
from colorcast.processing.batch import BatchProcessor
from colorcast.processing.cache import LRUCache


@pytest.fixture
def sample_image_rgb():
    """Create a sample RGB image."""
    return np.random.rand(256, 256, 3).astype(np.float32)


@pytest.fixture
def sample_style_image():
    """Create a sample style image with different colors."""
    return np.random.rand(256, 256, 3).astype(np.float32) * 0.5 + 0.25


class TestIntegrationWorkflow:
    """Test complete workflows integrating multiple components."""

    def test_complete_histogram_workflow(self, sample_image_rgb, sample_style_image):
        """Test complete histogram matching workflow."""
        # Apply transfer
        result = match_histograms_multichannel(sample_image_rgb, sample_style_image)

        # Verify result
        assert result.shape == sample_image_rgb.shape
        assert result.dtype == sample_image_rgb.dtype
        assert np.all(result >= 0) and np.all(result <= 1)

    def test_complete_meanstd_workflow(self, sample_image_rgb, sample_style_image):
        """Test complete mean/std transfer workflow."""
        result = color_transfer_meanstd(sample_image_rgb, sample_style_image)

        assert result.shape == sample_image_rgb.shape
        assert result.dtype == sample_image_rgb.dtype
        assert np.all(result >= 0) and np.all(result <= 1)

    def test_complete_lut_workflow(self, sample_image_rgb, sample_style_image):
        """Test complete LUT workflow with curves."""
        for curve_type in ["linear", "s-curve", "contrast"]:
            result = lut_transfer_with_curve(
                sample_image_rgb, sample_style_image, curve_type
            )
            assert result.shape == sample_image_rgb.shape
            assert result.dtype == sample_image_rgb.dtype
            assert np.all(result >= 0) and np.all(result <= 1)

    def test_complete_selective_workflow(self, sample_image_rgb, sample_style_image):
        """Test complete selective transfer workflow."""
        modes = ["shadows", "midtones", "highlights", "full"]
        for mode in modes:
            result = selective_color_transfer(
                sample_image_rgb, sample_style_image, mode=mode
            )
            assert result.shape == sample_image_rgb.shape
            assert result.dtype == sample_image_rgb.dtype
            assert np.all(result >= 0) and np.all(result <= 1)

    def test_complete_blending_workflow(self, sample_image_rgb, sample_style_image):
        """Test complete workflow with intensity blending."""
        # Apply transfer
        styled = match_histograms_multichannel(sample_image_rgb, sample_style_image)

        # Blend with different intensities
        intensities = [0.0, 0.25, 0.5, 0.75, 1.0]
        for intensity in intensities:
            result = blend_images(sample_image_rgb, styled, intensity)
            assert result.shape == sample_image_rgb.shape
            assert result.dtype == sample_image_rgb.dtype
            assert np.all(result >= 0) and np.all(result <= 1)

            # Verify intensity affects result
            if intensity == 0.0:
                assert np.allclose(result, sample_image_rgb, atol=1e-10)
            elif intensity == 1.0:
                assert np.allclose(result, styled, atol=1e-10)


class TestIntegrationWithCache:
    """Test integration with caching system."""

    def test_cached_workflow(self, sample_image_rgb, sample_style_image):
        """Test workflow with caching enabled."""
        cache = LRUCache(max_size=10)

        # First call - should compute
        result1 = cache.get_or_compute(
            key="test",
            compute_func=lambda: match_histograms_multichannel(
                sample_image_rgb, sample_style_image
            ),
        )

        # Second call - should use cache
        result2 = cache.get_or_compute(
            key="test",
            compute_func=lambda: match_histograms_multichannel(
                sample_image_rgb, sample_style_image
            ),
        )

        # Verify same result
        assert np.allclose(result1, result2)

        # Verify cache was used
        assert cache.stats()["hits"] == 1
        assert cache.stats()["misses"] == 1


class TestIntegrationWithRegistry:
    """Test integration with plugin registry."""

    def test_registry_workflow(self, sample_image_rgb, sample_style_image):
        """Test workflow using registry methods."""
        # List available methods
        methods = registry.list_methods()
        assert len(methods) > 0

        # Test each method via registry
        for method_id in methods.keys():
            method = registry.get_method(method_id)
            result = method.transfer(sample_image_rgb, sample_style_image)

            assert result.shape == sample_image_rgb.shape
            assert result.dtype == sample_image_rgb.dtype
            assert np.all(result >= 0) and np.all(result <= 1)


class TestBatchIntegration:
    """Test batch processing integration."""

    def test_batch_directory_processing(
        self, tmp_path, sample_image_rgb, sample_style_image
    ):
        """Test batch processing of directory."""
        from colorcast.processing.image_loader import save_image

        # Create test directory
        content_dir = tmp_path / "content"
        content_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Save sample images
        for i in range(3):
            save_image(
                sample_image_rgb, str(content_dir / f"test_{i}.jpg")
            )
        save_image(sample_style_image, str(tmp_path / "style.jpg"))

        # Process batch
        processor = BatchProcessor(
            transfer_method=match_histograms_multichannel,
            max_workers=2,
        )
        results = processor.process_directory(
            content_dir=content_dir,
            style_image=tmp_path / "style.jpg",
            output_dir=output_dir,
            pattern="*.jpg",
        )

        # Verify results
        assert len(results) == 3
        for result_path in results:
            assert result_path.exists()
            assert result_path.suffix == ".jpg"

        # Check no failed files
        assert len(processor.failed_files) == 0


class TestMultiMethodPipeline:
    """Test complex pipelines with multiple methods."""

    def test_sequential_processing(self, sample_image_rgb, sample_style_image):
        """Test applying multiple methods sequentially."""
        # First transfer
        result1 = match_histograms_multichannel(sample_image_rgb, sample_style_image)

        # Second transfer on result
        result2 = color_transfer_meanstd(result1, sample_style_image)

        # Final blend
        final = blend_images(sample_image_rgb, result2, intensity=0.5)

        assert final.shape == sample_image_rgb.shape
        assert final.dtype == sample_image_rgb.dtype
        assert np.all(final >= 0) and np.all(final <= 1)

    def test_method_comparison(self, sample_image_rgb, sample_style_image):
        """Compare results from different methods."""
        methods = [
            match_histograms_multichannel,
            color_transfer_meanstd,
            lambda s, r: lut_transfer_with_curve(s, r, "s-curve"),
        ]

        results = []
        for method in methods:
            result = method(sample_image_rgb, sample_style_image)
            results.append(result)
            assert result.shape == sample_image_rgb.shape

        # Verify methods produce different results (excluding identical methods)
        # Mean/std should differ from histogram matching and LUT
        assert not np.allclose(results[0], results[1], atol=1e-5)
        assert not np.allclose(results[1], results[2], atol=1e-5)
