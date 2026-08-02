#!/usr/bin/env python3
"""Measure Daltonization efficacy: before/after mean chroma-error ratio.

For each deficiency type the script:

1. Simulates the deficiency on the original image.
2. Measures the baseline chroma error (original vs simulated).
3. Daltonizes the original and re-simulates the corrected output.
4. Measures the post-correction chroma error.
5. Reports the ratio ``after_mean / before_mean`` — values below 1.0
   indicate improvement (less error after correction).

The script seeds the RNG for deterministic output and targets the samples
that the CI review used.

Usage::

    .venv/bin/python scripts/daltonization_efficacy.py
"""

from __future__ import annotations

import numpy as np

from colorcast.analysis.daltonization import daltonize
from colorcast.analysis.error_map import get_error_map
from colorcast.processing.simulation import ColorBlindSimulator

rng = np.random.default_rng(seed=2026_07_31)

DEFICIENCIES: tuple[str, ...] = ("protanopia", "deuteranopia", "tritanopia")
NUM_SAMPLES: int = 10
IMAGE_SIZE: int = 128


def _generate_sample() -> np.ndarray:
    """Generate a random RGB image in [0, 1] for testing."""
    return rng.random((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)


def _mean_chroma_error(original: np.ndarray, simulated: np.ndarray) -> float:
    """Return the mean chroma error between *original* and *simulated*."""
    em = get_error_map(original, simulated)
    return float(em.chroma_error.mean())


def main() -> None:
    simulator = ColorBlindSimulator()
    print(f"{'Deficiency':<16}  {'Before':>10}  {'After':>10}  {'Ratio (A/B)':>12}")
    print("-" * 56)

    for deficiency in DEFICIENCIES:
        before_means: list[float] = []
        after_means: list[float] = []

        for _ in range(NUM_SAMPLES):
            img = _generate_sample()
            simulated = simulator.transform_color_space(img, deficiency)
            before = _mean_chroma_error(img, simulated)

            corrected = daltonize(img, deficiency, intensity=1.0)
            re_simulated = simulator.transform_color_space(corrected, deficiency)
            after = _mean_chroma_error(corrected, re_simulated)

            before_means.append(before)
            after_means.append(after)

        avg_before = float(np.mean(before_means))
        avg_after = float(np.mean(after_means))
        ratio = avg_after / avg_before if avg_before > 1e-9 else float("inf")

        print(
            f"  {deficiency:<14}  "
            f"{avg_before:>10.4f}  "
            f"{avg_after:>10.4f}  "
            f"{ratio:>12.4f}"
        )


if __name__ == "__main__":
    main()
