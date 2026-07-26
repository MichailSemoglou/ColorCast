"""Tests for the enhanced validators in colorcast.utils.validators_enhanced."""

import numpy as np
import pytest
from skimage import io
from colorcast.processing.image_loader import normalize_to_float32
from colorcast.utils.validators_enhanced import (
    validate_image_array,
    validate_image_file,
    ALLOWED_IMAGE_EXTENSIONS,
)
from colorcast.utils.exceptions import ValidationError


class TestSpoofedFileRejection:
    """Magic-number verification via validate_image_file."""

    def test_png_renamed_to_jpg_is_rejected(self, tmp_path):
        """A valid PNG renamed to .jpg fails magic-number check."""
        img = (np.random.rand(16, 16, 3) * 255).astype(np.uint8)
        real_png = tmp_path / "real.png"
        spoofed_jpg = tmp_path / "spoofed.jpg"
        io.imsave(real_png, img)
        real_png.rename(spoofed_jpg)

        with pytest.raises((ValidationError, FileNotFoundError)):
            validate_image_file(str(spoofed_jpg), ALLOWED_IMAGE_EXTENSIONS)

    def test_valid_jpg_passes(self, tmp_path):
        """A valid JPEG passes the magic-number check."""
        img = (np.random.rand(16, 16, 3) * 255).astype(np.uint8)
        jpg_path = tmp_path / "real.jpg"
        io.imsave(jpg_path, img)

        validate_image_file(str(jpg_path), ALLOWED_IMAGE_EXTENSIONS)

    def test_valid_png_passes(self, tmp_path):
        """A valid PNG passes the magic-number check."""
        img = (np.random.rand(16, 16, 3) * 255).astype(np.uint8)
        png_path = tmp_path / "real.png"
        io.imsave(png_path, img)

        validate_image_file(str(png_path), ALLOWED_IMAGE_EXTENSIONS)


class TestNanInfRejection:
    """NaN and Inf pixel rejection via validate_image_array."""

    def test_nan_pixels_rejected(self):
        """validate_image_array raises ValidationError for NaN pixels."""
        clean = np.random.rand(8, 8, 3).astype(np.float32)
        corrupted = clean.copy()
        corrupted[2, 2, 1] = np.nan

        with pytest.raises(ValidationError, match="NaN"):
            validate_image_array(corrupted, max_pixels=10000)

    def test_inf_pixels_rejected(self):
        """validate_image_array raises ValidationError for Inf pixels."""
        clean = np.random.rand(8, 8, 3).astype(np.float32)
        corrupted = clean.copy()
        corrupted[3, 3, 0] = np.inf

        with pytest.raises(ValidationError, match="infinite"):
            validate_image_array(corrupted, max_pixels=10000)

    def test_clean_image_passes(self):
        """A valid float32 image passes NaN/Inf checks."""
        clean = np.random.rand(8, 8, 3).astype(np.float32)

        result = validate_image_array(clean, max_pixels=10000)

        np.testing.assert_array_equal(result, clean)

    def test_nan_rejected_on_uint8_as_well(self):
        """Explicit NaN in a uint8 array (cast from float) is caught."""
        arr = np.zeros((8, 8, 3), dtype=np.uint8)
        arr_float = arr.astype(np.float32)
        arr_float[0, 0, 0] = np.nan
        arr_nan_subtle = np.clip(arr_float, 0, 255).astype(np.float32)

        with pytest.raises(ValidationError, match="NaN"):
            validate_image_array(arr_nan_subtle, max_pixels=10000, dtype_range=(0.0, 255.0))


class TestMalformedShapeRejection:
    """normalize_to_float32 rejects non-(H, W, 3) inputs."""

    def test_2d_grayscale_rejected(self):
        """A 2D array raises ValueError."""
        img = np.random.rand(8, 8).astype(np.float32)

        with pytest.raises(ValueError, match="Expected image with shape"):
            normalize_to_float32(img)

    def test_4channel_rgba_rejected(self):
        """A 4-channel array raises ValueError."""
        img = np.random.rand(8, 8, 4).astype(np.float32)

        with pytest.raises(ValueError, match="Expected image with shape"):
            normalize_to_float32(img)

    def test_1d_array_rejected(self):
        """A 1D array raises ValueError."""
        arr = np.array([0, 128, 255], dtype=np.uint8)

        with pytest.raises(ValueError, match="Expected image with shape"):
            normalize_to_float32(arr)

    def test_hw3_passes(self):
        """An (H, W, 3) uint8 array normalises successfully."""
        img = np.full((4, 4, 3), 128, dtype=np.uint8)

        result = normalize_to_float32(img)

        assert result.shape == (4, 4, 3)
        assert result.dtype == np.float32
        np.testing.assert_allclose(result, 128 / 255, atol=0.01)
