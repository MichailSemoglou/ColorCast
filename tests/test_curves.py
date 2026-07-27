"""Tests for tone curve utilities."""

import numpy as np
import pytest
from colorcast.processing.curves import apply_curve


class TestApplyCurve:
    """Tests for the apply_curve function."""

    def test_linear_is_identity(self):
        values = np.random.rand(100, 100)
        result = apply_curve(values, "linear")
        np.testing.assert_array_equal(result, values)

    def test_s_curve_preserves_shape(self):
        values = np.random.rand(100, 100)
        result = apply_curve(values, "s-curve")
        assert result.shape == values.shape
        assert result.dtype == values.dtype

    def test_s_curve_fixed_points(self):
        values = np.array([0.0, 0.5, 1.0])
        result = apply_curve(values, "s-curve")
        np.testing.assert_array_almost_equal(result, [0.0, 0.5, 1.0])

    def test_s_curve_endpoints(self):
        assert apply_curve(np.array([0.0]), "s-curve")[0] == pytest.approx(0.0, abs=1e-10)
        assert apply_curve(np.array([0.5]), "s-curve")[0] == pytest.approx(0.5, abs=1e-10)
        assert apply_curve(np.array([1.0]), "s-curve")[0] == pytest.approx(1.0, abs=1e-10)

    def test_contrast_preserves_shape(self):
        values = np.random.rand(100, 100)
        result = apply_curve(values, "contrast")
        assert result.shape == values.shape
        assert result.dtype == values.dtype

    def test_contrast_increases_dark_values(self):
        values = np.array([0.1, 0.1, 0.1])
        result = apply_curve(values, "contrast")
        assert np.all(result > 0.1)

    def test_contrast_changes_midtones(self):
        values = np.array([0.4, 0.4, 0.4])
        result = apply_curve(values, "contrast")
        assert not np.allclose(result, values)

    def test_contrast_endpoints(self):
        result = apply_curve(np.array([0.0, 1.0]), "contrast")
        np.testing.assert_array_almost_equal(result, [0.0, 1.0])

    def test_unknown_curve_type_raises(self):
        values = np.array([0.5])
        with pytest.raises(ValueError, match="Unknown curve type"):
            apply_curve(values, "invalid")

    def test_3d_array_compatible(self):
        values = np.random.rand(50, 50, 3)
        result = apply_curve(values, "s-curve")
        assert result.shape == (50, 50, 3)

    @pytest.mark.parametrize("curve_type", ["linear", "s-curve", "contrast"])
    def test_output_range(self, curve_type):
        values = np.random.rand(100, 100)
        result = apply_curve(values, curve_type)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    @pytest.mark.parametrize("curve_type", ["linear", "s-curve", "contrast"])
    def test_deterministic(self, curve_type):
        values = np.random.rand(50, 50)
        r1 = apply_curve(values, curve_type)
        r2 = apply_curve(values, curve_type)
        np.testing.assert_array_equal(r1, r2)
