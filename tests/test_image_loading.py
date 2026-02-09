"""Tests for image loading and preprocessing."""

import pytest
import numpy as np
from colorcast.processing.image_loader import ensure_rgb, load_image
from colorcast.utils.exceptions import InvalidImageFormatError


class TestEnsureRgb:
    """Tests for RGB conversion utility."""

    def test_ensure_rgb_from_grayscale(self):
        """Test conversion of grayscale to RGB."""
        grayscale = np.random.rand(100, 100)
        rgb = ensure_rgb(grayscale)

        assert rgb.shape == (100, 100, 3)
        assert np.all(rgb[:, :, 0] == rgb[:, :, 1])
        assert np.all(rgb[:, :, 1] == rgb[:, :, 2])

    def test_ensure_rgb_from_single_channel(self):
        """Test conversion of single channel image."""
        single = np.random.rand(100, 100, 1)
        rgb = ensure_rgb(single)

        assert rgb.shape == (100, 100, 3)
        assert np.all(rgb[:, :, 0] == rgb[:, :, 1])

    def test_ensure_rgb_from_rgb(self):
        """Test that RGB image is unchanged."""
        rgb = np.random.rand(100, 100, 3)
        result = ensure_rgb(rgb)

        assert result.shape == rgb.shape
        assert np.all(result == rgb)

    def test_ensure_rgb_from_rgba(self):
        """Test removal of alpha channel."""
        rgba = np.random.rand(100, 100, 4)
        rgb = ensure_rgb(rgba)

        assert rgb.shape == (100, 100, 3)
        assert np.all(rgb == rgba[:, :, :3])

    def test_ensure_rgb_invalid_channels_2d(self):
        """Test that 2D image is converted."""
        img = np.random.rand(100, 100)
        result = ensure_rgb(img)

        assert result.shape == (100, 100, 3)

    def test_ensure_rgb_invalid_channels_5ch(self):
        """Test that 5-channel image raises error."""
        img = np.random.rand(100, 100, 5)

        with pytest.raises(InvalidImageFormatError):
            ensure_rgb(img)

    def test_ensure_rgb_invalid_dimensions_1d(self):
        """Test that 1D array raises error."""
        img = np.random.rand(100)

        with pytest.raises(InvalidImageFormatError):
            ensure_rgb(img)

    def test_ensure_rgb_invalid_dimensions_4d(self):
        """Test that 4D array raises error."""
        img = np.random.rand(100, 100, 3, 2)

        with pytest.raises(InvalidImageFormatError):
            ensure_rgb(img)


class TestLoadImage:
    """Tests for image loading functionality."""

    def test_load_image_creates_test_files(self, tmp_path):
        """Test loading actual image files."""
        # Create test image files
        import tempfile
        from skimage import io

        test_img = (np.random.rand(50, 50, 3) * 255).astype(np.uint8)

        # Save as PNG
        png_path = tmp_path / "test.png"
        io.imsave(png_path, test_img)

        # Load and verify
        loaded = load_image(str(png_path))

        assert loaded.shape == (50, 50, 3)
        assert loaded.dtype == np.float64
        assert np.all(loaded >= 0) and np.all(loaded <= 1)

    def test_load_image_with_max_dimension(self, tmp_path):
        """Test loading with dimension limit."""
        from skimage import io

        # Create large image
        large_img = (np.random.rand(100, 200, 3) * 255).astype(np.uint8)
        img_path = tmp_path / "large.png"
        io.imsave(img_path, large_img)

        # Load with max dimension of 150
        loaded = load_image(str(img_path), max_dimension=150)

        # Should be resized so max dimension is 150
        assert max(loaded.shape[0], loaded.shape[1]) <= 150
        assert loaded.shape[2] == 3

    def test_load_image_with_max_pixels(self, tmp_path):
        """Test loading with pixel limit."""
        from skimage import io

        # Create moderately sized image
        img = (np.random.rand(100, 100, 3) * 255).astype(np.uint8)
        img_path = tmp_path / "test.png"
        io.imsave(img_path, img)

        # Load with reasonable pixel limit
        loaded = load_image(str(img_path), max_pixels=50000)

        assert loaded.shape[0] * loaded.shape[1] <= 50000