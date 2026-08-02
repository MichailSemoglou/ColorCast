"""CVD accessibility dashboard — compare all three deficiencies at once.

Public API
----------
- :class:`DashboardResult` — container for original, three simulations, three
  error maps, and per-deficiency summary statistics.
- :func:`compute_dashboard` — run all three CVD simulations and error maps
  in parallel and return a :class:`DashboardResult`.
- :func:`generate_dashboard_report` — render the dashboard as a full-
  resolution Matplotlib figure and save to a file.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from colorcast.analysis.error_map import ErrorMap, get_error_map
from colorcast.processing.simulation import ColorBlindSimulator

_DEFICIENCIES: tuple[str, ...] = ("protanopia", "deuteranopia", "tritanopia")
_DEFICIENCY_LABELS: dict[str, str] = {
    "protanopia": "Protanopia (red-blind)",
    "deuteranopia": "Deuteranopia (green-blind)",
    "tritanopia": "Tritanopia (blue-blind)",
}


@dataclass
class DashboardResult:
    """Container returned by :func:`compute_dashboard`.

    Attributes
    ----------
    original : np.ndarray
        The source image, float32 (H, W, 3), values in [0, 1].
    simulated : dict[str, np.ndarray]
        Simulated images keyed by deficiency name.
    error_maps : dict[str, ErrorMap]
        :class:`ErrorMap` instances keyed by deficiency name.
    summary : dict[str, dict[str, float]]
        Per-deficiency summary stats: ``mean_error``, ``median_error``,
        ``p95_error``, ``percent_affected``.
    """

    original: np.ndarray
    simulated: dict[str, np.ndarray]
    error_maps: dict[str, ErrorMap]
    summary: dict[str, dict[str, float]]


def compute_dashboard(
    image_array: np.ndarray,
    max_workers: int | None = None,
) -> DashboardResult:
    """Run all three CVD simulations and error maps in parallel.

    Args:
        image_array: Source image, any numeric dtype, shape (H, W, 3).
        max_workers: Passed to ``ThreadPoolExecutor``.  Defaults to the
            number of deficiencies (3).

    Returns:
        :class:`DashboardResult`
    """
    from colorcast.processing.image_loader import normalize_to_float32

    image_array = normalize_to_float32(image_array)
    simulator = ColorBlindSimulator()

    def _simulate_and_map(deficiency: str) -> tuple[str, np.ndarray, ErrorMap, dict[str, float]]:
        sim = simulator.transform_color_space(image_array, deficiency)  # type: ignore[arg-type]
        em = get_error_map(image_array, sim, compute_dE00=True)
        stats = _summarize(em)
        return deficiency, sim, em, stats

    simulated: dict[str, np.ndarray] = {}
    error_maps: dict[str, ErrorMap] = {}
    summary: dict[str, dict[str, float]] = {}

    workers = max_workers if max_workers is not None else len(_DEFICIENCIES)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_simulate_and_map, d) for d in _DEFICIENCIES]
        for future in futures:
            deficiency, sim, em, stats = future.result()
            simulated[deficiency] = sim
            error_maps[deficiency] = em
            summary[deficiency] = stats

    return DashboardResult(
        original=image_array,
        simulated=simulated,
        error_maps=error_maps,
        summary=summary,
    )


def _summarize(em: ErrorMap) -> dict[str, float]:
    """Derive scalar summary statistics from an ErrorMap."""
    ce = em.chroma_error_dE00
    if ce is None:
        return {
            "mean_error": float("nan"),
            "median_error": float("nan"),
            "p95_error": float("nan"),
            "percent_affected": float("nan"),
        }
    return {
        "mean_error": float(np.mean(ce)),
        "median_error": float(np.median(ce)),
        "p95_error": float(np.percentile(ce, 95)),
        # dE00 ≈ 1 is the classic just-noticeable-difference threshold.
        "percent_affected": float(np.count_nonzero(ce > 1.0) / ce.size * 100),
    }


def format_summary_table(result: DashboardResult) -> str:
    """
    Format the deficiency summary as a monospace-aligned table string.

    Args:
        result: A pre-computed :class:`DashboardResult` whose ``summary``
            dict holds per-deficiency scalar statistics.

    Returns:
        Monospace-aligned plain-text table with one row per deficiency
        (mean ΔE, median ΔE, p95 ΔE, and affected-area percentage).
    """
    header = (
        f"{'Deficiency':<16}  {'Mean ΔE':>10}  "
        f"{'Median ΔE':>12}  {'p95 ΔE':>10}  {'Affected %':>12}"
    )
    lines = [header, "-" * 68]
    for deficiency in _DEFICIENCIES:
        s = result.summary.get(deficiency, {})
        label = deficiency.capitalize()[:13]
        lines.append(
            f"{label:<16}  "
            f"{s.get('mean_error', 0):>10.3f}  "
            f"{s.get('median_error', 0):>12.3f}  "
            f"{s.get('p95_error', 0):>10.3f}  "
            f"{s.get('percent_affected', 0):>11.1f}%"
        )
    return "\n".join(lines)


def generate_dashboard_report(
    result: DashboardResult,
    output_path: str,
    title: str = "CVD Accessibility Dashboard",
) -> str:
    """Render the dashboard as a full-resolution Matplotlib PNG.

    Layout::
        Row 0 — [           Original (centred)           ]
        Row 1 — [Protanopia]   [Deuteranopia]  [Tritanopia]
        Row 2 — [Chroma Loss (P)] [Chroma Loss (D)] [Chroma Loss (T)]
        Row 3 — (empty — reserved for summary table and caption)

    A summary table and a short caption explaining how to read the
    chroma-loss heatmaps are placed below the grid.

    Args:
        result: Result from :func:`compute_dashboard`.
        output_path: File path for the output PNG.
        title: Overall figure title.

    Returns:
        The ``output_path`` that was written.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    orig = np.clip(result.original, 0, 1)

    fig = Figure(figsize=(18, 16))
    FigureCanvasAgg(fig)
    axes = fig.subplots(4, 3)
    fig.suptitle(title, x=0.02, y=0.98, fontsize=15, fontfamily="monospace")

    # Row 0: Original — centred in the middle column
    axes[0, 0].axis("off")
    axes[0, 2].axis("off")
    axes[0, 1].imshow(orig)
    axes[0, 1].set_title("Original", fontfamily="monospace", fontsize=10)
    axes[0, 1].axis("off")

    # Row 1: three simulations
    for col, deficiency in enumerate(_DEFICIENCIES):
        sim = np.clip(result.simulated[deficiency], 0, 1)
        axes[1, col].imshow(sim)
        label = _DEFICIENCY_LABELS.get(deficiency, deficiency.capitalize())
        axes[1, col].set_title(label, fontfamily="monospace", fontsize=10)
        axes[1, col].axis("off")

    # Row 2: three chroma-loss heatmaps
    for col, deficiency in enumerate(_DEFICIENCIES):
        em = result.error_maps.get(deficiency)
        if em is not None:
            _show_heatmap(axes[2, col], em.chroma_error)
            axes[2, col].set_title(f"Chroma Loss ({deficiency[0].upper()})", fontsize=11)

    # Row 3: empty — reserved for the summary table and caption below
    for col in range(3):
        axes[3, col].axis("off")

    # Summary table — built with explicit column widths so data aligns
    summary_text = format_summary_table(result)
    fig.text(0.05, 0.1, summary_text, fontfamily="monospace", fontsize=10)

    # Caption explaining how to read the chroma-loss images
    caption = (
        "Chroma-loss heatmaps show, per pixel, how much chromatic information\n"
        "the simulation removed — red / bright regions lost the most colour\n"
        "contrast; dark regions were preserved.  The hot colormap is\n"
        "normalised so the brightest pixel in each heatmap represents the\n"
        "maximum ΔE loss for that deficiency."
    )
    fig.text(0.05, 0.025, caption, fontfamily="monospace", fontsize=10)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return output_path


def _show_heatmap(ax, chroma_error: np.ndarray) -> None:
    """Render a chroma-loss heatmap on a Matplotlib axis."""
    vmax = float(chroma_error.max())
    if vmax < 1e-6:
        ax.imshow(np.zeros_like(chroma_error), cmap="hot", vmin=0, vmax=1)
    else:
        ax.imshow(chroma_error / vmax, cmap="hot", vmin=0, vmax=1)
    ax.axis("off")
