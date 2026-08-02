"""Tests for image blending."""

import numpy as np

from colorcast.processing.blending import blend_images


class TestBlending:
    """Tests for image blending functionality."""

    def test_blend_full_intensity(self):
        """Test blending with full intensity (should be styled)."""
        original = np.random.rand(100, 100, 3)
        styled = np.random.rand(100, 100, 3)

        result = blend_images(original, styled, 1.0)

        assert result.shape == original.shape
        # With full intensity, result should equal styled
        np.testing.assert_array_almost_equal(result, styled)

    def test_blend_zero_intensity(self):
        """Test blending with zero intensity (should be original)."""
        original = np.random.rand(100, 100, 3)
        styled = np.random.rand(100, 100, 3)

        result = blend_images(original, styled, 0.0)

        assert result.shape == original.shape
        # With zero intensity, result should equal original
        np.testing.assert_array_almost_equal(result, original)

    def test_blend_half_intensity(self):
        """Test blending with half intensity."""
        original = np.ones((100, 100, 3)) * 0.2
        styled = np.ones((100, 100, 3)) * 0.8

        result = blend_images(original, styled, 0.5)

        assert result.shape == original.shape
        # Half intensity should give midpoint
        np.testing.assert_array_almost_equal(result, np.ones((100, 100, 3)) * 0.5)

    def test_blend_clamps_above_one(self):
        """Test that intensity > 1.0 is clamped to 1.0."""
        original = np.random.rand(100, 100, 3)
        styled = np.random.rand(100, 100, 3)

        result = blend_images(original, styled, 1.5)

        # Should be clamped to full styled
        np.testing.assert_array_almost_equal(result, styled)

    def test_blend_clamps_below_zero(self):
        """Test that intensity < 0.0 is clamped to 0.0."""
        original = np.random.rand(100, 100, 3)
        styled = np.random.rand(100, 100, 3)

        result = blend_images(original, styled, -0.5)

        # Should be clamped to original
        np.testing.assert_array_almost_equal(result, original)

    def test_blend_various_intensities(self):
        """Test blending with various intensity values."""
        original = np.ones((100, 100, 3)) * 0.0
        styled = np.ones((100, 100, 3)) * 1.0

        for intensity in [0.0, 0.25, 0.5, 0.75, 1.0]:
            result = blend_images(original, styled, intensity)
            assert result.shape == original.shape
            np.testing.assert_array_almost_equal(result, np.ones((100, 100, 3)) * intensity)
