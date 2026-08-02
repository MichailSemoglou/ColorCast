"""Plugin architecture for transfer methods."""

import threading
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from colorcast.processing.cache import StyleTransferCache


class TransferMethod(ABC):
    """Abstract base class for transfer methods."""

    # --- Slider label constants for subclasses -------------------------------
    _SLIDER_STYLE = "Style Intensity:"
    _SLIDER_SEVERITY = "Severity (0%=normal, 100%=full):"
    _SLIDER_CORRECTION = "Correction Intensity (0%=original, 100%=fully corrected):"

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
    def parameters(self) -> dict[str, dict]:
        """Configurable parameters and their defaults."""
        return {}

    #: Whether the method needs a reference (style) image. Methods that
    #: transform the source image alone, such as the CVD simulators and
    #: Daltonizers, set this to False; callers may then pass
    #: ``reference=None``.
    requires_reference: bool = True

    #: Label for the GUI intensity slider. Transfer methods blend source
    #: and styled image; simulators treat the slider as severity and
    #: Daltonizers as correction strength.
    slider_label: str = _SLIDER_STYLE

    def _require_reference(self, reference: np.ndarray | None) -> np.ndarray:
        """Validate that a reference image was provided.

        Args:
            reference: Reference image passed to ``transfer``

        Returns:
            The validated reference image

        Raises:
            ValueError: If ``reference`` is None
        """
        if reference is None:
            raise ValueError(f"{self.id} requires a reference image")
        return reference

    @abstractmethod
    def transfer(
        self,
        source: np.ndarray,
        reference: np.ndarray | None,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Apply transfer method.

        Args:
            source: Source image (H, W, 3)
            reference: Reference image (H, W, 3), or None for methods with
                ``requires_reference = False``
            **kwargs: Method-specific parameters

        Returns:
            Transferred image (H, W, 3)
        """
        pass


class TransferMethodRegistry:
    """Registry for transfer methods."""

    def __init__(self):
        self._methods: dict[str, type[TransferMethod]] = {}
        self._cache: StyleTransferCache | None = None  # lazy-initialized
        self._cache_lock = threading.Lock()

    def _ensure_cache(self) -> "StyleTransferCache":
        with self._cache_lock:
            if self._cache is None:
                from colorcast.processing.cache import LRUCache

                self._cache = LRUCache(max_size=32)
            return self._cache

    def transfer_cached(
        self,
        method_id: str,
        source: np.ndarray,
        reference: np.ndarray | None,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Apply a transfer method, caching the result for repeated calls.

        Args:
            method_id: Unique method identifier.
            source: Source image (H, W, 3).
            reference: Reference image or None.
            **kwargs: Method-specific parameters forwarded to ``transfer``.

        Returns:
            Transferred image (H, W, 3).
        """
        cache = self._ensure_cache()
        cache_key = cache.generate_key(source, reference, method_id, kwargs)

        def _compute() -> np.ndarray:
            method = self.get_method(method_id)
            return method.transfer(source, reference, **kwargs)

        return cache.get_or_compute(cache_key, _compute)

    def cache_stats(self) -> dict:
        """
        Get cache hit/miss/size statistics.

        Returns:
            Dictionary with ``hits``, ``misses``, and ``size`` keys, or
            an empty dict when the cache has not been initialized.
        """
        if self._cache is None:
            return {}
        return self._cache.stats()

    def clear_cache(self) -> None:
        """Remove all cached entries."""
        if self._cache is not None:
            self._cache.clear()

    def register(self, method_class: type[TransferMethod]) -> type[TransferMethod]:
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

    def list_methods(self) -> dict[str, str]:
        """
        List all registered methods.

        Returns:
            Dictionary mapping method IDs to display names
        """
        methods = {}
        for _method_id, method_class in self._methods.items():
            instance = method_class()
            methods[instance.id] = instance.name
        return methods


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
        reference: np.ndarray | None,
        **kwargs: Any,
    ) -> np.ndarray:
        reference = self._require_reference(reference)

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
        reference: np.ndarray | None,
        **kwargs: Any,
    ) -> np.ndarray:
        reference = self._require_reference(reference)

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
    def parameters(self) -> dict[str, dict]:
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
        reference: np.ndarray | None,
        alpha: float = 1.0,
        **kwargs: Any,
    ) -> np.ndarray:
        reference = self._require_reference(reference)

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
        reference: np.ndarray | None,
        **kwargs: Any,
    ) -> np.ndarray:
        reference = self._require_reference(reference)

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
        reference: np.ndarray | None,
        **kwargs: Any,
    ) -> np.ndarray:
        reference = self._require_reference(reference)

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
        reference: np.ndarray | None,
        **kwargs: Any,
    ) -> np.ndarray:
        reference = self._require_reference(reference)

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
    def parameters(self) -> dict[str, dict]:
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
        reference: np.ndarray | None,
        shadow_threshold: float = 0.3,
        **kwargs: Any,
    ) -> np.ndarray:
        reference = self._require_reference(reference)

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
    def parameters(self) -> dict[str, dict]:
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
        reference: np.ndarray | None,
        shadow_threshold: float = 0.3,
        highlight_threshold: float = 0.7,
        **kwargs: Any,
    ) -> np.ndarray:
        reference = self._require_reference(reference)

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
    def parameters(self) -> dict[str, dict]:
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
        reference: np.ndarray | None,
        highlight_threshold: float = 0.7,
        **kwargs: Any,
    ) -> np.ndarray:
        reference = self._require_reference(reference)

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

    requires_reference = False
    slider_label = TransferMethod._SLIDER_SEVERITY

    def transfer(
        self,
        source: np.ndarray,
        reference: np.ndarray | None,
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

    requires_reference = False
    slider_label = TransferMethod._SLIDER_SEVERITY

    def transfer(
        self,
        source: np.ndarray,
        reference: np.ndarray | None,
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

    requires_reference = False
    slider_label = TransferMethod._SLIDER_SEVERITY

    def transfer(
        self,
        source: np.ndarray,
        reference: np.ndarray | None,
        **kwargs: Any,
    ) -> np.ndarray:
        from colorcast.processing.simulation import ColorBlindSimulator

        return ColorBlindSimulator().simulate_tritanopia(source)


@registry.register
class DaltonizeProtanopiaMethod(TransferMethod):
    """Daltonization correction for protanopia (red-cone absent).

    Like the simulators, this method ignores the reference image and
    transforms the source image alone.  The full correction is computed
    at ``intensity=1.0``; the GUI blends original and corrected images
    through its intensity slider.
    """

    @property
    def name(self) -> str:
        return "Daltonize: Protanopia"

    @property
    def id(self) -> str:
        return "daltonize_protanopia"

    requires_reference = False
    slider_label = TransferMethod._SLIDER_CORRECTION

    def transfer(
        self,
        source: np.ndarray,
        reference: np.ndarray | None,
        intensity: float = 1.0,
        **kwargs: Any,
    ) -> np.ndarray:
        from colorcast.analysis.daltonization import daltonize

        return daltonize(source, "protanopia", intensity=intensity)


@registry.register
class DaltonizeDeuteranopiaMethod(TransferMethod):
    """Daltonization correction for deuteranopia (green-cone absent)."""

    @property
    def name(self) -> str:
        return "Daltonize: Deuteranopia"

    @property
    def id(self) -> str:
        return "daltonize_deuteranopia"

    requires_reference = False
    slider_label = TransferMethod._SLIDER_CORRECTION

    def transfer(
        self,
        source: np.ndarray,
        reference: np.ndarray | None,
        intensity: float = 1.0,
        **kwargs: Any,
    ) -> np.ndarray:
        from colorcast.analysis.daltonization import daltonize

        return daltonize(source, "deuteranopia", intensity=intensity)


@registry.register
class DaltonizeTritanopiaMethod(TransferMethod):
    """Daltonization correction for tritanopia (blue-cone absent)."""

    @property
    def name(self) -> str:
        return "Daltonize: Tritanopia"

    @property
    def id(self) -> str:
        return "daltonize_tritanopia"

    requires_reference = False
    slider_label = TransferMethod._SLIDER_CORRECTION

    def transfer(
        self,
        source: np.ndarray,
        reference: np.ndarray | None,
        intensity: float = 1.0,
        **kwargs: Any,
    ) -> np.ndarray:
        from colorcast.analysis.daltonization import daltonize

        return daltonize(source, "tritanopia", intensity=intensity)
