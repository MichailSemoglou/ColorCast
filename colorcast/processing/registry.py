"""Plugin architecture for transfer methods."""

from abc import ABC, abstractmethod
from typing import Dict, Type, Any
import numpy as np


class TransferMethod(ABC):
    """Abstract base class for transfer methods."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Display name of the method."""
        pass

    @property
    @abstractmethod
    def id(self) -> str:
        """Unique identifier for the method."""
        pass

    @property
    def parameters(self) -> Dict[str, dict]:
        """Configurable parameters and their defaults."""
        return {}

    @abstractmethod
    def transfer(
        self,
        source: np.ndarray,
        reference: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Apply transfer method.

        Args:
            source: Source image (H, W, 3)
            reference: Reference image (H, W, 3)
            **kwargs: Method-specific parameters

        Returns:
            Transferred image (H, W, 3)
        """
        pass


class TransferMethodRegistry:
    """Registry for transfer methods."""

    def __init__(self):
        self._methods: Dict[str, Type[TransferMethod]] = {}

    def register(self, method_class: Type[TransferMethod]) -> Type[TransferMethod]:
        """
        Register a transfer method.

        Args:
            method_class: TransferMethod subclass to register

        Returns:
            The registered method class (for decorator use)
        """
        instance = method_class()
        self._methods[instance.id] = method_class
        return method_class

    def get_method(self, method_id: str) -> TransferMethod:
        """
        Get instance of registered method.

        Args:
            method_id: Unique identifier of the method

        Returns:
            Instance of the transfer method

        Raises:
            ValueError: If method_id is not registered
        """
        if method_id not in self._methods:
            raise ValueError(f"Unknown transfer method: {method_id}")
        return self._methods[method_id]()

    def list_methods(self) -> Dict[str, str]:
        """
        List all registered methods.

        Returns:
            Dictionary mapping method IDs to display names
        """
        methods = {}
        for method_id, method_class in self._methods.items():
            instance = method_class()
            methods[instance.id] = instance.name
        return methods

    def has_method(self, method_id: str) -> bool:
        """
        Check if a method is registered.

        Args:
            method_id: Method identifier to check

        Returns:
            True if method is registered, False otherwise
        """
        return method_id in self._methods


# Global registry instance for transfer methods
# This registry is automatically populated with all built-in transfer methods
# when the module is imported. Custom methods can be registered using
# the @registry.register decorator.
registry = TransferMethodRegistry()


# Register built-in methods
@registry.register
class HistogramMatchingMethod(TransferMethod):
    """Histogram matching transfer method."""

    @property
    def name(self) -> str:
        return "Histogram Matching"

    @property
    def id(self) -> str:
        return "histogram"

    def transfer(
        self,
        source: np.ndarray,
        reference: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        from colorcast.processing.transfer_methods import match_histograms_multichannel

        return match_histograms_multichannel(source, reference)


@registry.register
class MeanStdTransferMethod(TransferMethod):
    """Mean/Std transfer method."""

    @property
    def name(self) -> str:
        return "Mean/Std Transfer"

    @property
    def id(self) -> str:
        return "meanstd"

    def transfer(
        self,
        source: np.ndarray,
        reference: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        from colorcast.processing.transfer_methods import color_transfer_meanstd

        return color_transfer_meanstd(source, reference)


@registry.register
class LabTransferMethod(TransferMethod):
    """Lab color space transfer method (Reinhard)."""

    @property
    def name(self) -> str:
        return "Lab Color Transfer (Reinhard)"

    @property
    def id(self) -> str:
        return "lab_reinhard"

    @property
    def parameters(self) -> Dict[str, dict]:
        return {
            "alpha": {
                "default": 1.0,
                "min": 0.0,
                "max": 1.0,
                "type": float,
            }
        }

    def transfer(
        self,
        source: np.ndarray,
        reference: np.ndarray,
        alpha: float = 1.0,
        **kwargs: Any,
    ) -> np.ndarray:
        from colorcast.processing.transfer_methods import color_transfer_lab

        return color_transfer_lab(source, reference, alpha=alpha)


@registry.register
class LutLinearMethod(TransferMethod):
    """LUT with linear curve method."""

    @property
    def name(self) -> str:
        return "LUT + Linear Curve"

    @property
    def id(self) -> str:
        return "lut_linear"

    def transfer(
        self,
        source: np.ndarray,
        reference: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        from colorcast.processing.transfer_methods import lut_transfer_with_curve

        return lut_transfer_with_curve(source, reference, "linear")


@registry.register
class LutSCurveMethod(TransferMethod):
    """LUT with S-curve method."""

    @property
    def name(self) -> str:
        return "LUT + S-Curve"

    @property
    def id(self) -> str:
        return "lut_scurve"

    def transfer(
        self,
        source: np.ndarray,
        reference: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        from colorcast.processing.transfer_methods import lut_transfer_with_curve

        return lut_transfer_with_curve(source, reference, "s-curve")


@registry.register
class LutContrastMethod(TransferMethod):
    """LUT with contrast curve method."""

    @property
    def name(self) -> str:
        return "LUT + Contrast"

    @property
    def id(self) -> str:
        return "lut_contrast"

    def transfer(
        self,
        source: np.ndarray,
        reference: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        from colorcast.processing.transfer_methods import lut_transfer_with_curve

        return lut_transfer_with_curve(source, reference, "contrast")


@registry.register
class SelectiveShadowsMethod(TransferMethod):
    """Selective transfer for shadows method."""

    @property
    def name(self) -> str:
        return "Selective: Shadows"

    @property
    def id(self) -> str:
        return "selective_shadows"

    @property
    def parameters(self) -> Dict[str, dict]:
        return {
            "shadow_threshold": {
                "default": 0.3,
                "min": 0.0,
                "max": 1.0,
                "type": float,
            }
        }

    def transfer(
        self,
        source: np.ndarray,
        reference: np.ndarray,
        shadow_threshold: float = 0.3,
        **kwargs: Any,
    ) -> np.ndarray:
        from colorcast.processing.transfer_methods import selective_color_transfer

        return selective_color_transfer(
            source, reference, mode="shadows", shadow_threshold=shadow_threshold
        )


@registry.register
class SelectiveMidtonesMethod(TransferMethod):
    """Selective transfer for midtones method."""

    @property
    def name(self) -> str:
        return "Selective: Midtones"

    @property
    def id(self) -> str:
        return "selective_midtones"

    @property
    def parameters(self) -> Dict[str, dict]:
        return {
            "shadow_threshold": {
                "default": 0.3,
                "min": 0.0,
                "max": 1.0,
                "type": float,
            },
            "highlight_threshold": {
                "default": 0.7,
                "min": 0.0,
                "max": 1.0,
                "type": float,
            },
        }

    def transfer(
        self,
        source: np.ndarray,
        reference: np.ndarray,
        shadow_threshold: float = 0.3,
        highlight_threshold: float = 0.7,
        **kwargs: Any,
    ) -> np.ndarray:
        from colorcast.processing.transfer_methods import selective_color_transfer

        return selective_color_transfer(
            source,
            reference,
            mode="midtones",
            shadow_threshold=shadow_threshold,
            highlight_threshold=highlight_threshold,
        )


@registry.register
class SelectiveHighlightsMethod(TransferMethod):
    """Selective transfer for highlights method."""

    @property
    def name(self) -> str:
        return "Selective: Highlights"

    @property
    def id(self) -> str:
        return "selective_highlights"

    @property
    def parameters(self) -> Dict[str, dict]:
        return {
            "highlight_threshold": {
                "default": 0.7,
                "min": 0.0,
                "max": 1.0,
                "type": float,
            }
        }

    def transfer(
        self,
        source: np.ndarray,
        reference: np.ndarray,
        highlight_threshold: float = 0.7,
        **kwargs: Any,
    ) -> np.ndarray:
        from colorcast.processing.transfer_methods import selective_color_transfer

        return selective_color_transfer(
            source, reference, mode="highlights", highlight_threshold=highlight_threshold
        )


@registry.register
class DeuteranopiaSimulatorMethod(TransferMethod):
    """Deuteranopia (green-cone absent) colour-blindness simulation.

    Unlike the style-transfer methods, this method ignores the reference
    image entirely and transforms the source image alone.  The ``intensity``
    slider in the GUI acts as a *severity* control (0 % = normal vision,
    100 % = total green-blindness) via the standard blend_images call.
    """

    @property
    def name(self) -> str:
        return "Simulator: Deuteranopia"

    @property
    def id(self) -> str:
        return "simulate_deuteranopia"

    def transfer(
        self,
        source: np.ndarray,
        reference: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        from colorcast.processing.simulation import ColorBlindSimulator

        return ColorBlindSimulator().simulate_deuteranopia(source)


@registry.register
class ProtanopiaSimulatorMethod(TransferMethod):
    """Protanopia (red-cone absent) colour-blindness simulation."""

    @property
    def name(self) -> str:
        return "Simulator: Protanopia"

    @property
    def id(self) -> str:
        return "simulate_protanopia"

    def transfer(
        self,
        source: np.ndarray,
        reference: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        from colorcast.processing.simulation import ColorBlindSimulator

        return ColorBlindSimulator().simulate_protanopia(source)


@registry.register
class TritanopiaSimulatorMethod(TransferMethod):
    """Tritanopia (blue-cone absent) colour-blindness simulation."""

    @property
    def name(self) -> str:
        return "Simulator: Tritanopia"

    @property
    def id(self) -> str:
        return "simulate_tritanopia"

    def transfer(
        self,
        source: np.ndarray,
        reference: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        from colorcast.processing.simulation import ColorBlindSimulator

        return ColorBlindSimulator().simulate_tritanopia(source)