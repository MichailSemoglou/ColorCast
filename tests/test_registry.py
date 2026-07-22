"""Tests for the transfer-method plugin registry."""

import numpy as np

from colorcast.processing.registry import TransferMethod, registry


class TestRequiresReference:
    """Reference-image metadata on registered methods."""

    def test_default_is_true(self):
        """The ABC default requires a reference image."""
        assert TransferMethod.requires_reference is True

    def test_only_simulators_skip_reference(self):
        """Exactly the simulate_* methods declare requires_reference=False."""
        for method_id in registry.list_methods():
            method = registry.get_method(method_id)
            assert method.requires_reference is (
                not method_id.startswith("simulate_")
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
