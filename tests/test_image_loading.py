"""Tests for image loading and preprocessing."""

import numpy as np
import pytest

from colorcast.processing.image_loader import ensure_rgb, load_image, save_image
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
        """RGBA input is composited onto a white background."""
        rgba = np.zeros((4, 4, 4), dtype=np.float32)
        rgba[:, :, 0] = 1.0  # red
        rgba[:, :, 3] = 1.0  # fully opaque

        rgb = ensure_rgb(rgba)

        assert rgb.shape == (4, 4, 3)
        np.testing.assert_allclose(rgb, rgba[:, :, :3], atol=1e-6)

        rgba[:, :, 3] = 0.0  # fully transparent becomes white
        rgb = ensure_rgb(rgba)
        np.testing.assert_allclose(rgb, 1.0, atol=1e-6)

        rgba[:, :, 3] = 0.5  # half transparent blends toward white
        rgb = ensure_rgb(rgba)
        np.testing.assert_allclose(rgb[:, :, 0], 1.0, atol=1e-6)
        np.testing.assert_allclose(rgb[:, :, 1], 0.5, atol=1e-6)

    def test_ensure_rgb_from_rgba_uint8(self):
        """Compositing preserves the input dtype for integer images."""
        rgba = np.zeros((2, 2, 4), dtype=np.uint8)
        rgba[:, :, 0] = 200
        rgba[:, :, 3] = 255  # fully opaque

        rgb = ensure_rgb(rgba)

        assert rgb.dtype == np.uint8
        np.testing.assert_array_equal(rgb, rgba[:, :, :3])

    @pytest.mark.parametrize("dtype", [np.int16, np.int32])
    def test_ensure_rgb_rgba_signed_integer_raises(self, dtype):
        """Signed integer RGBA input is rejected."""
        img = np.zeros((2, 2, 4), dtype=dtype)

        with pytest.raises(InvalidImageFormatError, match="Signed integer"):
            ensure_rgb(img)

    def test_ensure_rgb_rgba_uint64_raises(self):
        """64-bit integer RGBA input is rejected (float64 cannot hold it)."""
        img = np.zeros((2, 2, 4), dtype=np.uint64)

        with pytest.raises(InvalidImageFormatError, match="not supported"):
            ensure_rgb(img)

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

    def test_header_read_decompression_bomb_reraises(self, tmp_path, monkeypatch):
        """Pillow decompression bombs should stop before a full decode fallback.

        Args:
            tmp_path: Temporary directory fixture used to create a small image file.
            monkeypatch: Fixture used to replace ``io.imread`` with a function that
                would fail if the loader fell back to a full decode.

        Returns:
            None: The test passes when the expected ``DecompressionBombError`` is raised.
        """
        from PIL import Image

        from colorcast.processing import image_loader

        img_path = tmp_path / "bomb.png"
        Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8)).save(img_path)

        def fail_imread(*args, **kwargs):
            raise AssertionError("full decode should not be attempted")

        monkeypatch.setattr(image_loader.io, "imread", fail_imread)

        with pytest.raises(Image.DecompressionBombError):
            image_loader._get_image_dimensions(str(img_path), max_pixels=10)

    def test_header_read_rejects_multiframe_stack(self, monkeypatch):
        """Multi-frame TIFF-like arrays should not be treated as a single image.

        Args:
            monkeypatch: Fixture used to replace ``io.imread`` with a stub that
                returns a multi-frame array shape.

        Returns:
            None: The test passes when the loader raises ``InvalidImageFormatError``
            for the unsupported multi-frame input.
        """
        from colorcast.processing import image_loader

        monkeypatch.setattr(
            image_loader.io, "imread", lambda path: np.zeros((2, 4, 6, 3), dtype=np.uint8)
        )

        with pytest.raises(InvalidImageFormatError, match="Multi-frame"):
            image_loader._get_image_dimensions("dummy.tiff")


class TestSaveImage:
    """Tests for save_image dtype validation."""

    def test_save_float32(self, tmp_path):
        """Float32 arrays are scaled to uint8 on save."""
        img = np.random.rand(10, 10, 3).astype(np.float32)
        out = tmp_path / "out.png"

        save_image(img, str(out))

        assert out.exists()

    def test_save_uint8(self, tmp_path):
        """Uint8 arrays are saved as-is."""
        img = (np.random.rand(10, 10, 3) * 255).astype(np.uint8)
        out = tmp_path / "out.png"

        save_image(img, str(out))

        assert out.exists()

    @pytest.mark.parametrize("dtype", [np.int16, np.float16, np.uint16, bool])
    def test_save_unsupported_dtype_raises(self, tmp_path, dtype):
        """Unsupported dtypes raise instead of writing a corrupt file."""
        img = np.zeros((10, 10, 3), dtype=dtype)

        with pytest.raises(InvalidImageFormatError, match="Unsupported dtype"):
            save_image(img, str(tmp_path / "out.png"))
