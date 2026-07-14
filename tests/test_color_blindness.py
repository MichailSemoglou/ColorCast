"""Tests for color-blindness simulation, error maps, and Daltonization."""

import numpy as np
import pytest
from skimage import color as skcolor

from colorcast.analysis.daltonization import apply_daltonization, daltonize
from colorcast.analysis.error_map import ErrorMap, get_error_map, summarize_error_map
from colorcast.processing.simulation import ColorBlindSimulator


class TestColorBlindSimulator:
    """Tests for ColorBlindSimulator."""

    @pytest.fixture
    def simulator(self):
        return ColorBlindSimulator()

    @pytest.fixture
    def rgb_image(self):
        np.random.seed(0)
        return np.random.rand(32, 32, 3).astype(np.float32)

    @pytest.mark.parametrize("deficiency", ["protanopia", "deuteranopia", "tritanopia"])
    def test_transform_returns_correct_shape_and_range(self, simulator, rgb_image, deficiency):
        result = simulator.transform_color_space(rgb_image, deficiency)

        assert result.shape == rgb_image.shape
        assert result.dtype == np.float32
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)
        assert np.all(np.isfinite(result))

    @pytest.mark.parametrize("deficiency", ["protanopia", "deuteranopia", "tritanopia"])
    def test_uint8_input(self, simulator, deficiency):
        image = (np.random.rand(16, 16, 3) * 255).astype(np.uint8)

        result = simulator.transform_color_space(image, deficiency)

        assert result.shape == image.shape
        assert result.dtype == np.float32
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_unknown_deficiency_raises(self, simulator, rgb_image):
        with pytest.raises(ValueError, match="Unknown deficiency type"):
            simulator.transform_color_space(rgb_image, "achromatopsia")

    def test_uniform_gray_is_handled(self, simulator):
        gray = np.full((16, 16, 3), 0.5, dtype=np.float32)

        result = simulator.transform_color_space(gray, "deuteranopia")

        assert result.shape == gray.shape
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_simulation_changes_colorful_image(self, simulator, rgb_image):
        simulated = simulator.transform_color_space(rgb_image, "deuteranopia")

        assert not np.allclose(simulated, rgb_image, atol=1e-3)

    @pytest.mark.parametrize("deficiency", ["protanopia", "deuteranopia", "tritanopia"])
    def test_white_and_gray_are_preserved(self, simulator, deficiency):
        """Achromatic inputs must map to themselves (white-point invariant)."""
        white = np.ones((8, 8, 3), dtype=np.float32)
        gray = np.full((8, 8, 3), 0.5, dtype=np.float32)

        for neutral in [white, gray]:
            simulated = simulator.transform_color_space(neutral, deficiency)
            np.testing.assert_allclose(
                simulated, neutral, atol=1e-5,
                err_msg=f"{deficiency} does not preserve achromatic input",
            )

    def test_uint16_matches_normalized_float(self, simulator):
        """Full-range uint16 input must produce the same simulation as its
        float32 equivalent, confirming normalize_to_float32 scales by 65535."""
        rng = np.random.default_rng(42)
        float_img = rng.random((16, 16, 3)).astype(np.float32)
        uint16_img = (float_img * 65535).astype(np.uint16)

        result_float = simulator.transform_color_space(float_img, "deuteranopia")
        result_uint16 = simulator.transform_color_space(uint16_img, "deuteranopia")

        np.testing.assert_allclose(result_uint16, result_float, atol=5e-4)


class TestErrorMap:
    """Tests for error map computation."""

    @pytest.fixture
    def rgb_image(self):
        np.random.seed(1)
        return np.random.rand(32, 32, 3).astype(np.float32)

    def test_zero_error_for_identical_images(self, rgb_image):
        error_map = get_error_map(rgb_image, rgb_image)

        assert isinstance(error_map, ErrorMap)
        np.testing.assert_array_almost_equal(error_map.signed, 0.0, decimal=5)
        np.testing.assert_array_almost_equal(error_map.absolute, 0.0, decimal=5)
        np.testing.assert_array_almost_equal(error_map.chroma_error, 0.0, decimal=4)

    def test_error_map_shapes_and_types(self, rgb_image):
        simulator = ColorBlindSimulator()
        simulated = simulator.transform_color_space(rgb_image, "deuteranopia")
        error_map = get_error_map(rgb_image, simulated)

        assert error_map.signed.shape == rgb_image.shape
        assert error_map.absolute.shape == rgb_image.shape
        assert error_map.chroma_error.shape == rgb_image.shape[:2]
        assert error_map.signed_chroma_ab.shape == (rgb_image.shape[0], rgb_image.shape[1], 2)
        assert error_map.orig_l_star.shape == rgb_image.shape[:2]

        assert error_map.signed.dtype == np.float32
        assert error_map.absolute.dtype == np.float32
        assert error_map.chroma_error.dtype == np.float32

    def test_chroma_error_non_negative(self, rgb_image):
        simulator = ColorBlindSimulator()
        simulated = simulator.transform_color_space(rgb_image, "deuteranopia")
        error_map = get_error_map(rgb_image, simulated)

        assert np.all(error_map.chroma_error >= 0.0)
        assert np.all(error_map.absolute >= 0.0)

    def test_shape_mismatch_raises(self, rgb_image):
        other = np.random.rand(16, 16, 3).astype(np.float32)

        with pytest.raises(ValueError, match="Shape mismatch"):
            get_error_map(rgb_image, other)

    def test_non_rgb_raises(self):
        gray = np.random.rand(32, 32).astype(np.float32)

        with pytest.raises(ValueError, match="Expected.*3"):
            get_error_map(gray, gray)

    def test_summary_statistics(self, rgb_image):
        simulator = ColorBlindSimulator()
        simulated = simulator.transform_color_space(rgb_image, "deuteranopia")
        error_map = get_error_map(rgb_image, simulated)
        stats = summarize_error_map(error_map)

        assert set(stats.keys()) == {
            "mean_chroma_error",
            "max_chroma_error",
            "p95_chroma_error",
            "mean_rgb_error",
            "max_rgb_error",
        }
        assert stats["max_chroma_error"] >= stats["mean_chroma_error"]
        assert stats["p95_chroma_error"] <= stats["max_chroma_error"]
        assert stats["max_rgb_error"] >= stats["mean_rgb_error"]


class TestDaltonization:
    """Tests for Daltonization."""

    @pytest.fixture
    def rgb_image(self):
        np.random.seed(2)
        return np.random.rand(32, 32, 3).astype(np.float32)

    @pytest.mark.parametrize("deficiency", ["protanopia", "deuteranopia", "tritanopia"])
    def test_daltonize_zero_intensity_returns_original(self, rgb_image, deficiency):
        result = daltonize(rgb_image, deficiency, intensity=0.0)

        assert result.shape == rgb_image.shape
        assert result.dtype == np.float32
        np.testing.assert_array_almost_equal(result, rgb_image, decimal=5)

    @pytest.mark.parametrize("deficiency", ["protanopia", "deuteranopia", "tritanopia"])
    def test_daltonize_output_range(self, rgb_image, deficiency):
        result = daltonize(rgb_image, deficiency, intensity=1.0)

        assert result.shape == rgb_image.shape
        assert result.dtype == np.float32
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)
        assert np.all(np.isfinite(result))

    def test_daltonize_unknown_deficiency_raises(self, rgb_image):
        with pytest.raises(ValueError, match="Unknown deficiency type"):
            daltonize(rgb_image, "achromatopsia")

    def test_apply_daltonization_with_error_map(self, rgb_image):
        simulator = ColorBlindSimulator()
        simulated = simulator.transform_color_space(rgb_image, "deuteranopia")
        error_map = get_error_map(rgb_image, simulated)

        result = apply_daltonization(rgb_image, error_map, "deuteranopia", intensity=0.8)

        assert result.shape == rgb_image.shape
        assert result.dtype == np.float32
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)
        assert np.all(np.isfinite(result))

    def test_apply_daltonization_changes_image(self, rgb_image):
        simulator = ColorBlindSimulator()
        simulated = simulator.transform_color_space(rgb_image, "deuteranopia")
        error_map = get_error_map(rgb_image, simulated)

        result = apply_daltonization(rgb_image, error_map, "deuteranopia", intensity=1.0)

        assert not np.allclose(result, rgb_image, atol=1e-3)

    def test_tritanopia_luminance_correction_not_erased(self):
        """Verify the Tritanopia correction signal survives luminance processing.

        The Tritanopia shift matrix encodes the blue error as equal R+G shifts
        (luminance modulation). Restoring L* after correction would erase that
        signal. This test checks that the corrected image differs from the
        original in luminance, confirming the modulation was preserved.
        """
        img = np.zeros((16, 16, 3), dtype=np.float32)
        img[:, :, 2] = 1.0  # pure blue

        corrected = daltonize(img, "tritanopia", intensity=1.0)

        corrected_lab = skcolor.rgb2lab(corrected)
        orig_lab = skcolor.rgb2lab(img)
        l_diff = np.abs(corrected_lab[:, :, 0] - orig_lab[:, :, 0]).mean()

        assert l_diff > 1.0, (
            f"Tritanopia correction was erased: mean L* change is {l_diff:.3f}, "
            "expected > 1.0 (luminance modulation must survive)"
        )
