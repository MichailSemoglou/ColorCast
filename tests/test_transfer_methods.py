"""Tests for color transfer methods."""

import pytest
import numpy as np
from colorcast.processing.transfer_methods import (
    match_histograms_multichannel,
    color_transfer_meanstd,
    lut_transfer_with_curve,
    selective_color_transfer,
)
from colorcast.utils.exceptions import InvalidImageFormatError


class TestHistogramMatching:
    """Tests for histogram matching."""

    def test_basic_histogram_matching(self):
        """Test basic histogram matching functionality."""
        source = np.random.rand(100, 100, 3)
        reference = np.random.rand(100, 100, 3)

        result = match_histograms_multichannel(source, reference)

        assert result.shape == source.shape
        assert result.dtype == np.float64
        assert np.all(result >= 0) and np.all(result <= 1)

    def test_histogram_matching_different_sizes(self):
        """Test that reference image is resized correctly."""
        source = np.random.rand(100, 100, 3)
        reference = np.random.rand(200, 200, 3)

        result = match_histograms_multichannel(source, reference)

        assert result.shape == source.shape

    def test_histogram_matching_invalid_input_2d(self):
        """Test that 2D image raises appropriate errors."""
        source = np.random.rand(100, 100)  # 2D image
        reference = np.random.rand(100, 100, 3)

        with pytest.raises(InvalidImageFormatError):
            match_histograms_multichannel(source, reference)

    def test_histogram_matching_invalid_input_4d(self):
        """Test that 4D image raises appropriate errors."""
        source = np.random.rand(100, 100, 3, 2)  # 4D image
        reference = np.random.rand(100, 100, 3)

        with pytest.raises(InvalidImageFormatError):
            match_histograms_multichannel(source, reference)


class TestMeanStdTransfer:
    """Tests for mean/std transfer."""

    def test_mean_std_basic(self):
        """Test basic mean/std transfer."""
        source = np.random.rand(100, 100, 3)
        reference = np.random.rand(100, 100, 3)

        result = color_transfer_meanstd(source, reference)

        assert result.shape == source.shape
        # Verify result is in valid range
        assert np.all(result >= 0) and np.all(result <= 1)

    def test_mean_std_statistics(self):
        """Test that result matches reference statistics approximately."""
        source = np.random.rand(100, 100, 3)
        reference = np.random.rand(100, 100, 3)

        result = color_transfer_meanstd(source, reference)

        # Verify result matches reference statistics (with tolerance)
        for i in range(3):
            assert abs(np.mean(result[:, :, i]) - np.mean(reference[:, :, i])) < 0.01

    def test_mean_std_zero_std(self):
        """Test handling of zero standard deviation."""
        source = np.ones((100, 100, 3)) * 0.5  # Zero std
        reference = np.random.rand(100, 100, 3)

        result = color_transfer_meanstd(source, reference)

        assert result.shape == source.shape
        assert np.all(result >= 0) and np.all(result <= 1)


class TestLutTransfer:
    """Tests for LUT transfer with curves."""

    @pytest.mark.parametrize("curve_type", ["linear", "s-curve", "contrast"])
    def test_lut_transfer_curve_types(self, curve_type):
        """Test all curve types."""
        source = np.random.rand(100, 100, 3)
        reference = np.random.rand(100, 100, 3)

        result = lut_transfer_with_curve(source, reference, curve_type)

        assert result.shape == source.shape
        assert np.all(result >= 0) and np.all(result <= 1)

    def test_lut_transfer_invalid_curve(self):
        """Test invalid curve type."""
        source = np.random.rand(100, 100, 3)
        reference = np.random.rand(100, 100, 3)

        with pytest.raises(ValueError):
            lut_transfer_with_curve(source, reference, "invalid")


class TestSelectiveTransfer:
    """Tests for selective color transfer."""

    def test_shadow_region_transfer(self):
        """Test selective transfer in shadow regions."""
        source = np.random.rand(100, 100, 3) * 0.2  # Dark image
        reference = np.random.rand(100, 100, 3)

        result = selective_color_transfer(source, reference, mode="shadows")

        assert result.shape == source.shape
        assert np.all(result >= 0) and np.all(result <= 1)

    def test_midtone_region_transfer(self):
        """Test selective transfer in midtone regions."""
        source = np.random.rand(100, 100, 3) * 0.5 + 0.25  # Mid tones
        reference = np.random.rand(100, 100, 3)

        result = selective_color_transfer(source, reference, mode="midtones")

        assert result.shape == source.shape
        assert np.all(result >= 0) and np.all(result <= 1)

    def test_highlight_region_transfer(self):
        """Test selective transfer in highlight regions."""
        source = np.random.rand(100, 100, 3) * 0.2 + 0.8  # Bright image
        reference = np.random.rand(100, 100, 3)

        result = selective_color_transfer(source, reference, mode="highlights")

        assert result.shape == source.shape
        assert np.all(result >= 0) and np.all(result <= 1)

    def test_full_mode_transfer(self):
        """Test full mode (should behave like histogram matching)."""
        source = np.random.rand(100, 100, 3)
        reference = np.random.rand(100, 100, 3)

        result = selective_color_transfer(source, reference, mode="full")

        assert result.shape == source.shape
        assert np.all(result >= 0) and np.all(result <= 1)

    @pytest.mark.parametrize("mode", ["shadows", "midtones", "highlights", "full"])
    def test_all_selective_modes(self, mode):
        """Test all selective transfer modes."""
        source = np.random.rand(100, 100, 3)
        reference = np.random.rand(100, 100, 3)

        result = selective_color_transfer(source, reference, mode=mode)

        assert result.shape == source.shape
        assert np.all(result >= 0) and np.all(result <= 1)

    def test_invalid_mode(self):
        """Test invalid mode."""
        source = np.random.rand(100, 100, 3)
        reference = np.random.rand(100, 100, 3)

        with pytest.raises(ValueError):
            selective_color_transfer(source, reference, mode="invalid")

    def test_custom_thresholds(self):
        """Test custom threshold values."""
        source = np.random.rand(100, 100, 3)
        reference = np.random.rand(100, 100, 3)

        result = selective_color_transfer(
            source, reference, mode="midtones", shadow_threshold=0.4, highlight_threshold=0.8
        )

        assert result.shape == source.shape
        assert np.all(result >= 0) and np.all(result <= 1)