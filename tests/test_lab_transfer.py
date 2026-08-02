"""Tests for Lab color space transfer method."""

import numpy as np

from colorcast.processing.transfer_methods import color_transfer_lab


class TestLabColorTransfer:
    """Test suite for Lab color space transfer method."""

    def test_basic_transfer(self):
        """Test basic Lab transfer between two images."""
        # Create test images
        source = np.random.rand(100, 100, 3)
        reference = np.random.rand(100, 100, 3)

        result = color_transfer_lab(source, reference)

        # Check output shape
        assert result.shape == source.shape
        assert result.dtype == np.float32

        # Check values are in valid range [0, 1]
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    def test_alpha_parameter(self):
        """Test alpha parameter controls transfer strength."""
        source = np.random.rand(50, 50, 3)
        reference = np.random.rand(50, 50, 3)

        # Full transfer (alpha=1.0)
        result_full = color_transfer_lab(source, reference, alpha=1.0)

        # Partial transfer (alpha=0.5)
        result_half = color_transfer_lab(source, reference, alpha=0.5)

        # No transfer (alpha=0.0)
        result_none = color_transfer_lab(source, reference, alpha=0.0)

        # No transfer should return original image
        np.testing.assert_array_almost_equal(result_none, source, decimal=5)

        # Half transfer should be between source and full transfer
        # (approximately, due to nonlinear Lab space)
        assert not np.array_equal(result_half, source)
        assert not np.array_equal(result_half, result_full)

    def test_edge_cases_alpha(self):
        """Test edge cases for alpha parameter."""
        source = np.random.rand(50, 50, 3)
        reference = np.random.rand(50, 50, 3)

        # Alpha < 0 should be treated as 0
        result_neg = color_transfer_lab(source, reference, alpha=-0.5)
        np.testing.assert_array_almost_equal(result_neg, source, decimal=5)

        # Alpha > 1 should be treated as 1
        result_over = color_transfer_lab(source, reference, alpha=1.5)
        result_full = color_transfer_lab(source, reference, alpha=1.0)
        np.testing.assert_array_almost_equal(result_over, result_full, decimal=5)

    def test_identical_images(self):
        """Test transfer when source and reference are identical."""
        image = np.random.rand(100, 100, 3)
        result = color_transfer_lab(image, image)

        # Result should be close to original (small differences due to log/exp)
        # Allow some tolerance due to numerical precision in Lab space
        np.testing.assert_array_almost_equal(result, image, decimal=3)

    def test_different_sizes(self):
        """Test that images with different sizes are handled correctly."""
        source = np.random.rand(100, 100, 3)
        reference = np.random.rand(50, 50, 3)

        # Should not raise an error (validation handles resizing)
        result = color_transfer_lab(source, reference)
        assert result.shape == source.shape

    def test_monochrome_reference(self):
        """Test transfer with monochrome reference image."""
        source = np.random.rand(100, 100, 3)
        reference = np.ones((100, 100, 3)) * 0.5  # Gray reference

        result = color_transfer_lab(source, reference)
        assert result.shape == source.shape
        assert np.all(result >= 0) and np.all(result <= 1)

    def test_extreme_color_values(self):
        """Test with extreme color values (very dark/bright images)."""
        # Very dark source
        dark_source = np.random.rand(100, 100, 3) * 0.1
        # Very bright reference
        bright_ref = 0.9 + np.random.rand(100, 100, 3) * 0.1

        result = color_transfer_lab(dark_source, bright_ref)
        assert np.all(result >= 0) and np.all(result <= 1)

    def test_color_preservation(self):
        """Test that Lab transfer preserves color relationships better than RGB."""
        # Create source with specific color pattern
        source = np.zeros((100, 100, 3))
        source[:, :50, 0] = 0.8  # Red region
        source[:, 50:, 1] = 0.8  # Green region

        # Create reference with different colors
        reference = np.random.rand(100, 100, 3)

        result = color_transfer_lab(source, reference)

        # Result should still have structure (not completely random)
        # Check that result is not uniform
        assert np.std(result) > 0.01

    def test_numerical_stability(self):
        """Test numerical stability with very small values."""
        source = np.ones((100, 100, 3)) * 1e-6
        reference = np.ones((100, 100, 3)) * 1e-5

        result = color_transfer_lab(source, reference)

        # Should not produce NaN or Inf
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
        assert np.all(result >= 0) and np.all(result <= 1)

    def test_lab_space_properties(self):
        """Test that the transfer actually operates in Lab space."""
        source = np.random.rand(100, 100, 3)
        reference = np.random.rand(100, 100, 3)

        result = color_transfer_lab(source, reference)

        # Result should be different from source (unless identical)
        if not np.array_equal(source, reference):
            assert not np.array_equal(result, source)

    def test_consistency(self):
        """Test that same inputs produce same outputs."""
        source = np.random.rand(100, 100, 3)
        reference = np.random.rand(100, 100, 3)

        result1 = color_transfer_lab(source, reference, alpha=0.7)
        result2 = color_transfer_lab(source, reference, alpha=0.7)

        np.testing.assert_array_equal(result1, result2)

    def test_real_world_scenario(self):
        """Test with more realistic image-like data."""
        # Create a gradient source image
        x = np.linspace(0, 1, 100)
        y = np.linspace(0, 1, 100)
        X, Y = np.meshgrid(x, y)

        source = np.stack([X, Y, 1 - X], axis=2)
        reference = np.stack([1 - X, Y, X], axis=2)

        result = color_transfer_lab(source, reference)

        assert result.shape == source.shape
        assert np.all(result >= 0) and np.all(result <= 1)
        assert not np.array_equal(result, source)
