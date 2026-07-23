"""Tests for the transfer-method plugin registry."""

import numpy as np

from colorcast.processing.registry import TransferMethod, registry


class TestRequiresReference:
    """Reference-image metadata on registered methods."""

    def test_default_is_true(self):
        """The ABC default requires a reference image."""
        assert TransferMethod.requires_reference is True

    def test_only_reference_free_methods_skip_reference(self):
        """Exactly the simulate_* and daltonize_* methods declare
        requires_reference=False."""
        reference_free = ("simulate_", "daltonize_")
        for method_id in registry.list_methods():
            method = registry.get_method(method_id)
            assert method.requires_reference is (
                not method_id.startswith(reference_free)
            ), f"unexpected requires_reference for {method_id}"

    def test_simulator_accepts_none_reference(self):
        """Simulator methods run with reference=None."""
        img = np.random.rand(8, 8, 3).astype(np.float32)

        result = registry.get_method("simulate_protanopia").transfer(img, None)

        assert result.shape == img.shape
        assert result.dtype == np.float32

    def test_transfer_method_rejects_none_reference(self):
        """Reference-requiring methods fail loudly on reference=None."""
        import pytest

        img = np.random.rand(8, 8, 3).astype(np.float32)

        with pytest.raises(ValueError, match="requires a reference image"):
            registry.get_method("histogram").transfer(img, None)


class TestDaltonizerMethods:
    """Daltonization correction methods registered alongside the simulators."""

    def test_all_daltonizers_registered(self):
        """All three daltonize_* methods are registered."""
        for deficiency in ("protanopia", "deuteranopia", "tritanopia"):
            assert registry.has_method(f"daltonize_{deficiency}")

    def test_daltonizer_accepts_none_reference(self):
        """Daltonizer methods run with reference=None."""
        img = np.random.rand(8, 8, 3).astype(np.float32)

        result = registry.get_method("daltonize_deuteranopia").transfer(img, None)

        assert result.shape == img.shape
        assert result.dtype == np.float32
        assert np.all(result >= 0) and np.all(result <= 1)


class TestSliderLabel:
    """Slider-label metadata driving the GUI intensity control."""

    def test_default_is_style_intensity(self):
        """The ABC default labels the slider as style intensity."""
        assert TransferMethod.slider_label == "Style Intensity:"

    def test_simulators_declare_severity_label(self):
        """Simulator methods label the slider as severity."""
        for method_id in registry.list_methods():
            if method_id.startswith("simulate_"):
                assert registry.get_method(method_id).slider_label == (
                    "Severity (0%=normal, 100%=full):"
                )

    def test_daltonizers_declare_correction_label(self):
        """Daltonizer methods label the slider as correction intensity."""
        for method_id in registry.list_methods():
            if method_id.startswith("daltonize_"):
                assert registry.get_method(method_id).slider_label == (
                    "Correction Intensity (0%=original, 100%=fully corrected):"
                )
