"""Perceptually uniform appearance spaces for color-difference metrics.

This module provides an extensible abstraction over color appearance
spaces — CIELAB (legacy) and ICtCp (HDR-aware, ITU-R BT.2100) — so
that error-map and dashboard computations can report ΔE values that
account for viewing conditions and absolute luminance.

Implemented standards:
- CIE 1976 L*a*b* (CIELAB) with CIE76 ΔE*ab and CIEDE2000 ΔE00
- ITU-R BT.2100 ICtCp color space
- ITU-R BT.2124-0 (2018) ΔE_ITP color difference metric

References:
- CIE (2004). Colorimetry (CIE Publication 15:2004).
- ITU-R (2018). BT.2100: Image parameter values for high dynamic range
  television for use in production and international programme exchange.
- ITU-R (2018). BT.2124-0: Objective metric for the assessment of colour
  volume in high dynamic range television.
- Sharma, G., Wu, W., & Dalal, E. N. (2005). The CIEDE2000 color-difference
  formula: Implementation notes, supplementary test data, and mathematical
  observations. Color Research & Application, 30(1), 21-30.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from colorcast.utils.color_utils import srgb_to_linear

# ---------------------------------------------------------------------------
# linear sRGB → display-referred BT.2020 RGB
# ---------------------------------------------------------------------------

_SRGB_TO_BT2020 = np.array(
    [
        [0.62740389, 0.32928304, 0.04331308],
        [0.06909729, 0.91954040, 0.01136232],
        [0.01639144, 0.08801331, 0.89559525],
    ],
    dtype=np.float64,
)

# ---------------------------------------------------------------------------
# BT.2020 RGB → LMS — BT.2100 / ICtCp transform
# ---------------------------------------------------------------------------

# Column-major form: M[i,j] = coefficient of RGB channel *j* for LMS channel *i*.
# Transposed in _rgb_to_ictcp for row-vector multiplication (N,3) @ (3,3).
_BT2020_TO_LMS = np.array(
    [
        [1688.0 / 4096.0, 2146.0 / 4096.0, 262.0 / 4096.0],
        [683.0 / 4096.0, 2951.0 / 4096.0, 462.0 / 4096.0],
        [99.0 / 4096.0, 309.0 / 4096.0, 3688.0 / 4096.0],
    ],
    dtype=np.float64,
)

# ---------------------------------------------------------------------------
# LMS' → ICtCp — ITU-R BT.2100
# ---------------------------------------------------------------------------

_LMS_TO_ICTCP = np.array(
    [
        [0.5000, 0.5000, 0.0000],
        [1.6138, -3.3235, 1.7097],
        [4.3782, -4.2456, -0.1326],
    ],
    dtype=np.float64,
)

# ---------------------------------------------------------------------------
# PQ (ST 2084) transfer function constants
# ---------------------------------------------------------------------------

_PQ_M1 = 0.1593017578125  # 2610 / 16384
_PQ_M2 = 78.84375  # 2523 / 32
_PQ_C1 = 0.8359375  # 3424 / 4096
_PQ_C2 = 18.8515625  # 2413 / 128
_PQ_C3 = 18.6875  # 2392 / 128
_PQ_LUMINANCE_REF = 10000.0  # absolute luminance reference, cd/m² (BT.2100)
_PQ_NORM = 1.0 / _PQ_LUMINANCE_REF
_SRGB_PEAK_LUMINANCE = 100.0  # reference display peak white, cd/m² (sRGB)

_ICTCP_SCALE = 720.0  # BT.2124 ICtCp ΔE scaling factor


def _pq_eotf_inv(linear: np.ndarray) -> np.ndarray:
    """Apply the ST 2084 inverse EOTF (perceptual quantizer).

    Expects absolute linear luminance in cd/m² (range [0, 10000] for
    BT.2100 reference).  Returns PQ-encoded values in [0, 1].
    """
    y = np.maximum(linear * _PQ_NORM, 0.0)
    ym1 = np.power(y, _PQ_M1)
    numerator = _PQ_C1 + _PQ_C2 * ym1
    denominator = 1.0 + _PQ_C3 * ym1
    return np.power(numerator / denominator, _PQ_M2)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@dataclass
class AppearanceDelta:
    """Per-pixel color-difference result from an appearance space."""

    values: np.ndarray  # (H, W) float32, per-pixel ΔE
    space_name: str  # e.g. "ICtCp", "CIELAB"


class AppearanceSpace(ABC):
    """Abstract color space for per-pixel ΔE computation.

    Thread-safety: implementations must be safe for concurrent ``delta_E``
    calls on a shared instance. The dashboard runs three deficiency
    simulations in parallel and may share one appearance space across workers.
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def from_rgb(self, rgb: np.ndarray) -> np.ndarray:
        """Convert a float32 RGB image (H, W, 3) in [0, 1] to this space."""

    @abstractmethod
    def delta_E(self, a: np.ndarray, b: np.ndarray) -> AppearanceDelta:
        """Compute per-pixel ΔE between two images in this space."""


# ---------------------------------------------------------------------------
# CIELAB (legacy) backend
# ---------------------------------------------------------------------------


class CIELABSpace(AppearanceSpace):
    """CIE 1976 L*a*b* (D65) — legacy perceptually uniform space.

    Supports two color-difference metrics:
    - CIE76 ΔE*ab: Euclidean distance in L*a*b* space (fast, less perceptually uniform)
    - CIEDE2000 ΔE00: Improved formula with better perceptual uniformity (slower)

    The metric is selected via the ``metric`` parameter in the constructor.
    Default is CIEDE2000 for better perceptual uniformity.

    Example:
        >>> from colorcast.analysis.appearance import CIELABSpace
        >>> space_cie76 = CIELABSpace(metric="cie76")
        >>> space_ciede2000 = CIELABSpace(metric="ciede2000")
        >>> delta = space_ciede2000.delta_E(a, b)
    """

    def __init__(self, metric: str = "ciede2000") -> None:
        """Initialize the CIELAB space with a color-difference metric.

        Args:
            metric: Color-difference metric. Supported values are ``"cie76"``
                (CIE 1976 ΔE*ab, Euclidean distance) and ``"ciede2000"``
                (CIEDE2000 ΔE00, improved perceptual uniformity).
        """
        normalized = metric.lower()
        if normalized not in {"cie76", "ciede2000"}:
            raise ValueError(f"unsupported metric: {metric!r}")
        self.metric = normalized

    @property
    def name(self) -> str:
        return f"CIELAB ({self.metric})"

    def from_rgb(self, rgb: np.ndarray) -> np.ndarray:
        from skimage.color import rgb2lab

        return rgb2lab(np.asarray(rgb, dtype=np.float64)).astype(np.float32)

    def delta_E(self, a: np.ndarray, b: np.ndarray) -> AppearanceDelta:
        lab1 = self.from_rgb(a)
        lab2 = self.from_rgb(b)

        if self.metric == "cie76":
            diff = lab1 - lab2
            values = np.sqrt(np.sum(diff**2, axis=-1)).astype(np.float32)
        else:  # ciede2000
            from skimage.color import deltaE_ciede2000

            values = deltaE_ciede2000(lab1, lab2).astype(np.float32)

        return AppearanceDelta(values=values, space_name=self.name)


# ---------------------------------------------------------------------------
# ICtCp (ITU-R BT.2100, HDR-aware) backend
# ---------------------------------------------------------------------------


class ICtCpSpace(AppearanceSpace):
    """ITU-R BT.2100 ICtCp — HDR-aware perceptually uniform color space.

    This implementation follows ITU-R BT.2100 and BT.2124-0 exactly. The
    color-difference metric is ΔE_ITP as defined in BT.2124-0 (2018):

        ΔE_ITP = 720 × √(ΔI² + (0.5 × ΔCt)² + ΔCp²)

    The backend accepts normalized ``(H, W, 3)`` image values in ``[0, 1]``.
    By default these are interpreted as nonlinear sRGB and are decoded with
    IEC 61966-2-1 before being mapped to a peak luminance of 100 cd/m².
    The resulting ``(H, W, 3)`` array has dtype ``float32`` and contains the
    ICtCp coordinates.

    The appearance difference metric is reported as ``AppearanceDelta`` with
    ``values`` shaped ``(H, W)`` and dtype ``float32``.

    Example:
        >>> from colorcast.analysis.appearance import ICtCpSpace
        >>> space = ICtCpSpace()
        >>> ictcp = space.from_rgb(rgb_image)
        >>> delta = space.delta_E(a, b)
    """

    def __init__(
        self,
        *,
        transfer_function: str = "srgb",
        peak_luminance: float = _SRGB_PEAK_LUMINANCE,
    ) -> None:
        """Initialize the appearance space with explicit input encoding.

        Args:
            transfer_function: Input transfer function for normalized RGB data.
                Supported values are ``"srgb"`` for nonlinear sRGB and
                ``"linear"`` for already-linear data.
            peak_luminance: Peak display luminance in cd/m² used to scale the
                input values before BT.2100 conversion.
        """
        normalized_tf = transfer_function.lower()
        if normalized_tf not in {"srgb", "linear"}:
            raise ValueError(f"unsupported transfer function: {transfer_function!r}")
        if peak_luminance <= 0:
            raise ValueError("peak_luminance must be positive")

        self.transfer_function = normalized_tf
        self.peak_luminance = float(peak_luminance)

    def _rgb_to_ictcp(self, rgb: np.ndarray) -> np.ndarray:
        if self.transfer_function == "srgb":
            linear = srgb_to_linear(rgb)
        else:
            linear = np.asarray(rgb, dtype=np.float64)
        linear *= self.peak_luminance  # [0, 1] → absolute cd/m²
        h, w = linear.shape[:2]
        flat = linear.reshape(-1, 3)

        bt2020 = flat @ _SRGB_TO_BT2020.T
        lms = bt2020 @ _BT2020_TO_LMS.T
        lms_pq = _pq_eotf_inv(lms)
        ictcp = lms_pq @ _LMS_TO_ICTCP.T

        return ictcp.reshape(h, w, 3).astype(np.float32)

    def from_rgb(self, rgb: np.ndarray) -> np.ndarray:
        return self._rgb_to_ictcp(rgb)

    @property
    def name(self) -> str:
        return "ICtCp"

    def delta_E(self, a: np.ndarray, b: np.ndarray) -> AppearanceDelta:
        ic1 = self._rgb_to_ictcp(a)
        ic2 = self._rgb_to_ictcp(b)
        diff = ic1 - ic2
        # ITU-R BT.2124-0: ΔE_ITP = 720 × √(ΔI² + (0.5 × ΔCt)² + ΔCp²)
        diff = np.stack(
            [diff[..., 0], 0.5 * diff[..., 1], diff[..., 2]],
            axis=-1,
        )
        de = np.sqrt(np.sum(diff**2, axis=-1)) * _ICTCP_SCALE
        return AppearanceDelta(values=de.astype(np.float32), space_name=self.name)


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def make_appearance_space(name: str) -> AppearanceSpace:
    """Create an appearance space by name.

    This factory centralizes space selection so CLI, GUI, and other callers
    do not duplicate the mapping logic or rely on fragile index arithmetic.

    Args:
        name: Space identifier. Supported values are ``"cielab"`` (CIEDE2000)
            and ``"ictcp"`` (BT.2124 ΔE_ITP).

    Returns:
        An ``AppearanceSpace`` instance configured with sensible defaults.

    Raises:
        ValueError: If ``name`` is not recognized.

    Example:
        >>> from colorcast.analysis.appearance import make_appearance_space
        >>> space = make_appearance_space("ictcp")
        >>> delta = space.delta_E(a, b)
    """
    normalized = name.lower()
    if normalized == "cielab":
        return CIELABSpace()
    if normalized == "ictcp":
        return ICtCpSpace()
    raise ValueError(f"unsupported appearance space: {name!r}. " f"Supported: 'cielab', 'ictcp'.")
