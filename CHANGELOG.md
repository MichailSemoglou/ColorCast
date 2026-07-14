# Changelog

## [2.1.1] – 2026-07-14

### Fixed

- `ColorBlindSimulator` white-point bug: `[1, 1, 1]` no longer collapses to `[0, 1, 1]` under protanopia. Protanopia and Deuteranopia now use the Viénot, Brettel, & Mollon (1999) linear RGB matrices and Tritanopia uses the Brettel, Viénot, & Mollon (1997) two-half-plane method, all pre-computed from DaltonLens. Each matrix row sums to 1, so achromatic whites and grays are preserved.

### Changed

- `colorcast/processing/simulation.py` replaces the explicit RGB → LMS → projection → LMS → RGB pipeline with direct linear RGB matrices; the LMS conversion matrices have been removed.
- `docs/wiki/Color-Vision-Background.md` updated to describe the new pre-computed matrix pipeline and the Brettel 1997 tritanopia implementation.

### Added

- Regression tests in `tests/test_color_blindness.py` verifying that white and gray inputs are preserved for all three deficiencies.

v2.1.1 · released July 2026 · MIT

## [2.1.0] – unreleased

### Added

- Color-blindness simulation in `colorcast/processing/simulation.py` via `ColorBlindSimulator`, supporting deuteranopia, protanopia, and tritanopia using the Smith-Pokorny / Viénot model.
- Error-map analysis in `colorcast/analysis/error_map.py` (`ErrorMap`, `get_error_map`, `plot_error_heatmap`, `summarize_error_map`) for measuring chromatic information lost in simulation.
- Daltonization in `colorcast/analysis/daltonization.py` (`apply_daltonization`, `daltonize`) for re-encoding lost chromatic information into perceptible channels.
- Three simulator methods registered in `colorcast/processing/registry.py` so the GUI and CLI can select them.
- PyQt5 graphical interface in `colorcast/gui.py` extracted from the legacy root script.
- Distinct entry points: `colorcast` for the CLI and `colorcast-gui` for the GUI; `python -m colorcast` launches the GUI to match the README.
- Root `colorcast.py` reduced to a deprecated compatibility shim that delegates to the package; a `DeprecationWarning` is emitted when it is run directly.
- Smoke tests in `tests/test_entry_points.py` for both console scripts and module execution.
- Tests in `tests/test_color_blindness.py` covering simulation, error maps, and Daltonization.

### Fixed

- `ColorBlindSimulator` now gamma-decodes nonlinear sRGB input before the linear RGB → LMS transform and gamma-encodes the result, correcting the color-space mismatch in the simulation pipeline.
- `colorcast/__main__.py` now imports `ALLOWED_IMAGE_EXTENSIONS` from the correct module and no longer passes invalid `min`/`max` arguments to `argparse`.
- `pyproject.toml` switches to `[tool.setuptools.packages.find]` with `exclude = ["tests*"]` for package discovery. `scikit-image` minimum raised to `>=0.19.0`.
- Documentation and packaging metadata synchronized: `README.md`, `docs/index.rst`, and generated `dist/`/`colorcast.egg-info/` artifacts reported version 2.1.0 and Python 3.10+ at the time of the 2.1.0 milestone.

## [2.0.0] – 2026-02-09

### Added

- Nine color transfer methods: histogram matching, mean/standard-deviation transfer, Lab transfer after Reinhard, LUT with linear/S-curve/contrast curves, and selective transfer for shadows, midtones, highlights, and full image.
- GPU-accelerated transfer methods in `colorcast/processing/gpu_transfer.py` with CPU fallback when CuPy is unavailable.
- Input validation for image dimensions, pixel values, color space, and histogram data.
- LRU cache in `colorcast/processing/cache.py` for repeated transfer operations.
- Batch processing in `colorcast/processing/batch.py` with `ThreadPoolExecutor` and per-file failure handling.
- Analysis tools in `colorcast/analysis/` for side-by-side method comparison and visualization.
- Command-line interface in `colorcast/__main__.py`.
- Sphinx documentation configuration in `docs/`.
- Citation metadata in `CITATION.cff` and Zenodo metadata in `.zenodo.json`.

### Changed

- Renamed public transfer functions: `histogram_matching` to `match_histograms_multichannel`, `mean_std_transfer` to `color_transfer_meanstd`, and `lab_transfer` to `color_transfer_lab`.

v2.0.0 · released February 2026 · MIT
