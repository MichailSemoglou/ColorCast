"""Comprehensive integration tests for ColorCast.

These tests verify end-to-end workflows and real-world usage scenarios.
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from colorcast import (
    blend_images,
    color_transfer_lab,
    color_transfer_meanstd,
    load_image,
    lut_transfer_with_curve,
    match_histograms_multichannel,
    save_image,
    selective_color_transfer,
)
from colorcast.processing.batch import BatchProcessor
from colorcast.processing.cache import StyleTransferCache


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    temp = tempfile.mkdtemp()
    yield Path(temp)
    shutil.rmtree(temp)


class TestIntegrationWorkflows:
    """Test complete end-to-end workflows."""

    @pytest.fixture
    def sample_images(self, temp_dir):
        """Create sample test images."""
        # Create test images
        content_img = Image.new("RGB", (512, 512), color=(100, 150, 200))
        style_img = Image.new("RGB", (512, 512), color=(200, 100, 150))

        content_path = temp_dir / "content.jpg"
        style_path = temp_dir / "style.jpg"

        content_img.save(content_path)
        style_img.save(style_path)

        return content_path, style_path

    def test_full_workflow_load_to_save(self, sample_images, temp_dir):
        """Test complete workflow from file loading to saving."""
        content_path, style_path = sample_images
        output_path = temp_dir / "output.jpg"

        # Execute full workflow
        content = load_image(str(content_path))
        style = load_image(str(style_path))
        assert content.shape == (512, 512, 3)
        assert style.shape == (512, 512, 3)

        # Apply transfer
        result = color_transfer_meanstd(content, style)
        assert result.shape == content.shape
        assert np.all(result >= 0) and np.all(result <= 1)

        # Blend with intensity
        result = blend_images(content, result, intensity=0.7)
        assert np.all(result >= 0) and np.all(result <= 1)

        # Save result
        save_image(result, str(output_path))
        assert output_path.exists()

        # Verify saved image
        loaded_result = load_image(str(output_path))
        assert loaded_result.shape == content.shape
        # Use looser tolerance for JPEG compression artifacts
        assert np.allclose(loaded_result, result, rtol=0.01, atol=0.02)

    def test_multiple_methods_workflow(self, sample_images, temp_dir):
        """Test applying multiple transfer methods sequentially."""
        content_path, style_path = sample_images

        content = load_image(str(content_path))
        style = load_image(str(style_path))

        # Apply different methods
        results = {}
        methods = {
            "histogram": match_histograms_multichannel,
            "meanstd": color_transfer_meanstd,
            "lab": lambda s, r: color_transfer_lab(s, r, alpha=0.8),
            "lut_linear": lambda s, r: lut_transfer_with_curve(s, r, "linear"),
            "lut_scurve": lambda s, r: lut_transfer_with_curve(s, r, "s-curve"),
        }

        for name, method in methods.items():
            result = method(content, style)
            results[name] = result

            # Verify each result
            assert result.shape == content.shape
            assert np.all(result >= 0) and np.all(result <= 1)

            # Save each result
            output_path = temp_dir / f"result_{name}.jpg"
            save_image(result, str(output_path))
            assert output_path.exists()

        # Verify all methods produced different results
        unique_results = {
            name for name, r in results.items() if not np.allclose(r, results["histogram"])
        }
        assert len(unique_results) > 1

    def test_caching_workflow(self, sample_images):
        """Test caching integration with workflow."""
        content_path, style_path = sample_images

        content = load_image(str(content_path))
        style = load_image(str(style_path))

        # Create cache
        cache = StyleTransferCache(max_size=5)

        # First call - cache miss
        result1 = cache.get_or_compute(
            key="test_key",
            compute_func=lambda: color_transfer_meanstd(content, style),
        )

        # Second call - cache hit
        result2 = cache.get_or_compute(
            key="test_key",
            compute_func=lambda: color_transfer_meanstd(content, style),
        )

        # Verify cache worked
        assert np.allclose(result1, result2)
        assert cache.stats()["hits"] == 1
        assert cache.stats()["misses"] == 1

    def test_batch_processing_workflow(self, temp_dir):
        """Test complete batch processing workflow."""
        # Create multiple content images
        content_dir = temp_dir / "content"
        content_dir.mkdir()

        for i in range(5):
            img = Image.new("RGB", (256, 256), color=(100 + i * 20, 150, 200 - i * 20))
            img.save(content_dir / f"img_{i}.jpg")

        # Create style image
        style_img = Image.new("RGB", (256, 256), color=(200, 100, 150))
        style_path = temp_dir / "style.jpg"
        style_img.save(style_path)

        # Process batch
        output_dir = temp_dir / "output"

        processor = BatchProcessor(
            transfer_method=color_transfer_meanstd,
            max_workers=2,
        )

        results = processor.process_directory(
            content_dir=content_dir,
            style_image=style_path,
            output_dir=output_dir,
            pattern="*.jpg",
        )

        # Verify results
        assert len(results) == 5
        assert all(r.exists() for r in results)
        assert processor.failed_files == []  # No failures

    def test_selective_transfer_workflow(self, sample_images, temp_dir):
        """Test selective transfer with different modes."""
        content_path, style_path = sample_images

        content = load_image(str(content_path))
        style = load_image(str(style_path))

        # Test all selective modes
        modes = ["shadows", "midtones", "highlights", "full"]

        for mode in modes:
            result = selective_color_transfer(content, style, mode=mode)

            assert result.shape == content.shape
            assert np.all(result >= 0) and np.all(result <= 1)

            # Save result
            output_path = temp_dir / f"selective_{mode}.jpg"
            save_image(result, str(output_path))
            assert output_path.exists()

    def test_intensity_blending_workflow(self, sample_images, temp_dir):
        """Test intensity blending workflow."""
        content_path, style_path = sample_images

        content = load_image(str(content_path))
        style = load_image(str(style_path))

        result = color_transfer_meanstd(content, style)

        # Test different intensities
        intensities = [0.0, 0.25, 0.5, 0.75, 1.0]

        for intensity in intensities:
            blended = blend_images(content, result, intensity=intensity)

            assert blended.shape == content.shape
            assert np.all(blended >= 0) and np.all(blended <= 1)

            # Verify blending behavior
            if intensity == 0.0:
                assert np.allclose(blended, content)
            elif intensity == 1.0:
                assert np.allclose(blended, result)

            # Save result
            output_path = temp_dir / f"blended_{intensity:.2f}.jpg"
            save_image(blended, str(output_path))
            assert output_path.exists()


class TestErrorRecovery:
    """Test error handling and recovery scenarios."""

    def test_invalid_file_handling(self):
        """Test handling of invalid files."""
        with pytest.raises((FileNotFoundError, Exception)):
            load_image("nonexistent_file.jpg")

    def test_different_size_images(self):
        """Test handling of images with different sizes."""
        content = np.random.rand(256, 256, 3)
        reference = np.random.rand(512, 512, 3)

        # Should handle different sizes by resizing
        result = match_histograms_multichannel(content, reference)
        assert result.shape == content.shape

    def test_grayscale_to_rgb(self, temp_dir):
        """Test handling of grayscale images."""
        # Create grayscale image
        gray_img = Image.new("L", (256, 256), color=128)
        gray_path = temp_dir / "gray.jpg"
        gray_img.save(gray_path)

        # Should convert to RGB automatically
        loaded = load_image(str(gray_path))
        assert loaded.ndim == 3
        assert loaded.shape[2] == 3

    def test_batch_with_errors(self, temp_dir):
        """Test batch processing with some failing files."""
        content_dir = temp_dir / "content"
        content_dir.mkdir()

        # Create valid images
        for i in range(3):
            img = Image.new("RGB", (128, 128), color=(100, 150, 200))
            img.save(content_dir / f"valid_{i}.jpg")

        # Create invalid file (corrupted JPEG)
        invalid_path = content_dir / "invalid.jpg"
        invalid_path.write_bytes(b"not a valid jpeg file")

        # Create style image
        style_img = Image.new("RGB", (128, 128), color=(200, 100, 150))
        style_path = temp_dir / "style.jpg"
        style_img.save(style_path)

        # Process batch
        output_dir = temp_dir / "output"

        processor = BatchProcessor(
            transfer_method=color_transfer_meanstd,
            max_workers=2,
        )

        results = processor.process_directory(
            content_dir=content_dir,
            style_image=style_path,
            output_dir=output_dir,
            pattern="*.jpg",
        )

        # Should process valid files despite one invalid
        assert len(results) == 3  # Only valid files
        assert len(processor.failed_files) >= 1  # At least the invalid file


class TestPerformanceIntegration:
    """Test performance-related integration scenarios."""

    def test_large_image_processing(self):
        """Test processing of large images."""
        # Create large image (4K)
        large_img = np.random.rand(4096, 4096, 3)
        style_img = np.random.rand(4096, 4096, 3)

        # Should process without errors
        result = color_transfer_meanstd(large_img, style_img)
        assert result.shape == large_img.shape

    def test_memory_efficiency(self):
        """Test memory efficiency with multiple operations."""
        images = [np.random.rand(512, 512, 3) for _ in range(10)]
        style = np.random.rand(512, 512, 3)

        # Process multiple images
        for img in images:
            result = color_transfer_meanstd(img, style)
            assert result.shape == img.shape

    def test_cache_effectiveness(self):
        """Test cache effectiveness with repeated operations."""
        cache = StyleTransferCache(max_size=10)
        content = np.random.rand(1024, 1024, 3)
        style = np.random.rand(1024, 1024, 3)

        # First computation
        result1 = cache.get_or_compute(
            key="test",
            compute_func=lambda: color_transfer_meanstd(content, style),
        )

        # Cache hit
        result2 = cache.get_or_compute(
            key="test",
            compute_func=lambda: color_transfer_meanstd(content, style),
        )

        # Verify cache worked
        assert np.allclose(result1, result2)
        assert cache.stats()["hits"] == 1
        assert cache.stats()["misses"] == 1


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_minimum_size_image(self):
        """Test processing of minimum size image."""
        tiny = np.random.rand(1, 1, 3)
        style = np.random.rand(1, 1, 3)

        result = color_transfer_meanstd(tiny, style)
        assert result.shape == (1, 1, 3)

    def test_uniform_image(self):
        """Test processing of uniform (flat) image."""
        uniform = np.ones((100, 100, 3)) * 0.5
        varied = np.random.rand(100, 100, 3)

        result = color_transfer_meanstd(uniform, varied)
        assert result.shape == uniform.shape
        assert np.all(result >= 0)

    def test_extreme_contrast(self):
        """Test processing of extreme high contrast image."""
        extreme = np.zeros((100, 100, 3))
        extreme[:, :, 0] = np.random.choice([0, 1], size=(100, 100))
        normal = np.random.rand(100, 100, 3)

        result = color_transfer_meanstd(extreme, normal)
        assert np.all(result >= 0) and np.all(result <= 1)

    def test_zero_intensity_blending(self):
        """Test blending with zero intensity."""
        content = np.random.rand(256, 256, 3)
        style = np.random.rand(256, 256, 3)
        result = color_transfer_meanstd(content, style)

        blended = blend_images(content, result, intensity=0.0)
        assert np.allclose(blended, content)

    def test_full_intensity_blending(self):
        """Test blending with full intensity."""
        content = np.random.rand(256, 256, 3)
        style = np.random.rand(256, 256, 3)
        result = color_transfer_meanstd(content, style)

        blended = blend_images(content, result, intensity=1.0)
        assert np.allclose(blended, result)
