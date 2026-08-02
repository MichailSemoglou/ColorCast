# Changelog

## [2.5.0] – 2026-08-02

### Added

- Headless GUI smoke tests in `tests/test_gui.py` (6 tests) covering method-switch toggles the style button and apply-with-no-images warning branches, runnable with `QT_QPA_PLATFORM=offscreen`.
- Path-containment regression tests in `tests/test_validators_enhanced.py` (5 tests) guarding against sibling-prefix bypass, path traversal, and multiple-base-directory selection in `validate_file_path`.
- `--verbose` flag on the CLI parser and each subcommand so tracebacks on errors are opt-in and default to a one-line message; `test_cli_error_shows_traceback_with_verbose` in `tests/test_entry_points.py`.
- GitHub Actions CI pipeline (`.github/workflows/ci.yml`): lint (black, isort, ruff), mypy type-check, and test matrix across Python 3.10–3.13 with xvfb for headless PyQt5.
- `[tool.ruff]` configuration in `pyproject.toml` (line-length 100, target py310+, rules E/F/I/W/UP/B/SIM/C4) with per-file ignores for test files with unavoidable long hypothesis decorator lines.
- CVD accessibility dashboard in `colorcast/analysis/dashboard.py` (`compute_dashboard`, `DashboardResult`, `generate_dashboard_report`) for comparing all three deficiencies at once, plus `_DEFICIENCIES` and `_DEFICIENCY_LABELS` constants.
- PyPI publish GitHub Actions workflow (`.github/workflows/publish.yml`) triggered on release publish.
- `.github/dependabot.yml` for automated dependency update PRs.
- `.github/PULL_REQUEST_TEMPLATE.md`, `.github/ISSUE_TEMPLATE/bug_report.md`, `.github/ISSUE_TEMPLATE/feature_request.md` — GitHub community templates.
- `CODE_OF_CONDUCT.md`, `CONTRIBUTING.md`, `SECURITY.md` — community health files.
- Citing section and CI status badge in `README.md`.

### Changed

- `cmd_transfer` now builds kwargs from `method.parameters` instead of unconditionally forwarding `shadow_threshold` and `highlight_threshold` to every method, preventing signature clashes if a reference-free method tightens its `transfer()` contract.
- `docs/index.rst` restructured: removed `quickstart`, `user_guide`, `advanced`, and `contributing` placeholder pages; source install command simplified to `pip install -e ".[dev,analysis]"`; CVD and Daltonization added to the feature list.
- `docs/conf.py` updated with current author, copyright range, and dynamic version import.
- `pyproject.toml` dev dependencies: dropped `pylint` and `pytest-mock`; added `imageio`.
- `README.md` lint command updated to `ruff check`.
- `.gitignore` excludes image files (`*.png`, `*.jpg`, etc.) except `imgs/`; `MANIFEST.in` drops `requirements.txt`/`requirements-dev.txt`, includes `SECURITY.md`, and grafts `imgs/`.
- `main()` classifies `ValueError`, `FileNotFoundError`, `ValidationError`, and `ImageProcessingError` as exit code 2; unexpected exceptions exit with code 3. Tracebacks go to stderr and only when `--verbose` is set.
- `--intensity` in the CLI now rejects values outside `[0, 1]` with a clear argparse error, using `validate_float_parameter` behind a custom type validator.
- `process_pairs` and `process_single` in `batch.py` now catch `FileNotFoundError` and `OSError` alongside `ImageLoadError`, so missing files are classified as image load errors rather than "unexpected error."
- `apply_daltonization` extracted `_compute_chromaticity_weight` and `_restore_luminance` helpers, cutting the function from ~135 to ~45 lines.
- `colorcast/__init__.py` re-exports `ColorBlindSimulator`, `daltonize`, `get_error_map`, `ErrorMap`, and `MethodComparison`; module docstring deletes banned vocabulary ("advanced", "sophisticated") and adds mentions of colour-blindness simulation and Daltonization.
- Version string centralized in `colorcast/_version.py`; `__init__.py`, `docs/conf.py`, and `__main__.py` import from it. `pyproject.toml` declares `dynamic = ["version"]` and reads the value from `_version.py` via `[tool.setuptools.dynamic]`, eliminating the duplicated version string.
- `_get_image_dimensions()` reads image dimensions from PIL headers before `imread` allocates the full array, rejecting oversized images before decoding; a 200 MB byte-level file-size cap guards against decompression bombs.
- `StyleTransferCache.get_or_compute` stores a copy of computed values so cached results are immutable snapshots; callers can no longer corrupt the cache by mutating a returned array in place.
- `gpu_transfer.py` module docstring explains the `gpu_` prefix is historical; `gpu_histogram_matching` collapsed to its CPU body with the dead GPU→CPU round-trip removed; `gpu_lab_transfer` GPU branch documented as a placeholder.
- CuPy import warning removed from module level; `is_gpu_available()` is the public entry point for checking accelerator presence.
- `BatchProcessor` docstring warns against concurrent reuse of a single instance.
- `StyleTransferApp.__init__` accepts an optional `ColorCastConfig` parameter; window size, preview size, default intensity, slider debounce, and default method now derive from config instead of hardcoded module-level constants. `gui.py` imports `ALLOWED_IMAGE_EXTENSIONS` from `validators_enhanced` and derives file-dialog filters from it.
- `_EPSILON = 1e-8`, `_LAB_L_BOUNDS = (0, 100)`, and `_LAB_AB_BOUNDS = (-128, 127)` extracted as module-level constants in `transfer_methods.py` and imported by `gpu_transfer.py`, eliminating 6 copies each of epsilon and Lab channel bounds.
- `TransferMethod` base class now carries `_SLIDER_SEVERITY` and `_SLIDER_CORRECTION` class constants; the 6 simulator and Daltonizer subclasses in `registry.py` reference them instead of repeating string literals.
- All CLI functions in `__main__.py` now have return type annotations (`-> None` or `-> argparse.Namespace`); error `print()` calls route to `stderr`.
- Remaining broad `except Exception` blocks (`gui.py:235`, `gui.py:353`, `gui.py:401`, `__main__.py:362`, `comparison.py:174`) now carry `# noqa: BLE001` with a one-line justification, matching the convention in `batch.py`.
- `validate_and_resize_images` return annotation tightened from bare `tuple` to `tuple[np.ndarray, np.ndarray]` (`colorcast/processing/transfer_methods.py:47`).
- README test metrics updated to match current suite output: 75% coverage, 359 passed, 1 skipped.

### Fixed

- `save_image` re-raises `ImageProcessingError` with `from e` so the original cause and traceback chain are preserved.
- `show_image` in `gui.py` calls `QImage.copy()` before converting to `QPixmap` to prevent garbage-collection crashes when the backing numpy buffer goes out of scope.
- `validate_image_file` extract `_check_extension_matches` helper and shared `_FORMAT_TO_EXT` / `_PIL_FORMAT_TO_EXT` maps eliminate the duplicate imghdr/PIL extension-check paths.
- `MethodComparison.compare_methods` logs per-method exceptions via `logger.error(exc_info=True)` so failures are surfaced in the log even though the NaN `_error` field allows ranking to continue.
- `ColorCastConfig.load()` now validates known configuration values against their declared types and raises `TypeError` for mismatches instead of silently accepting incompatible values.
- `tests/test_config.py` adds 10 tests for `ColorCastConfig` default values in the 2.5.0 release notes.
- Adjusted test expectations to match the new CLI error exit codes (`ValueError`, `FileNotFoundError`, `ValidationError`, and `ImageProcessingError` now exit with code 2).
- `_get_image_dimensions` in `image_loader.py` no longer catches all exceptions silently: `ImportError` (PIL absent) and `OSError`/`ValueError` (header failure) fall through to full decode, and unexpected exceptions are logged at warning level.
- `B904` violations fixed: `image_loader.py` and `validators_enhanced.py` re-raises in `except` blocks now use `raise ... from e` to preserve the cause chain.
- `tests/test_visualization.py` adds `plt.close(fig)` after every figure creation to prevent Matplotlib handle accumulation between tests.
- `tests/test_property_based.py` fixes dtype declarations (uses `np.float64` instead of bare `float`) to satisfy Hypothesis strict type checking.
- `daltonize()` with `intensity=0` returned the simulated image instead of the original because `apply_daltonization` set its working base to the simulated image before the zero-intensity early-return path. Added an early return in `daltonize()` that skips simulation and error-map computation when `intensity < 1e-6`, returning the original image unchanged (`colorcast/analysis/daltonization.py:239-244`).

### Removed

- Dead GPU branch in `gpu_histogram_matching`: the CuPy-available path copied data GPU→CPU per channel, matched histograms with scikit-image, then copied CPU→GPU — strictly slower than the CPU fallback.
- `ColorCastConfig.cache_size`: unused field that accepted values silently and never affected any runtime behavior.
- Module-level `warnings.warn` in `gpu_transfer.py` that fired on every import even for users with no GPU.
- `requirements.txt` and `requirements-dev.txt` — all dependencies are now declared in `pyproject.toml` `[project.optional-dependencies]`.
- `colorcast.py` and `colorcast/utils/validators.py` — legacy standalone module and validation shim.

## [2.4.2] – 2026-07-27

### Added

- 148 new tests across 6 test files, raising coverage from 51% to 69%:
  - `tests/test_curves.py` -- 13 tests for `apply_curve` (all curve types, edge cases)
  - `tests/test_gpu_transfer.py` -- 21 tests for GPU transfer CPU fallback paths across `gpu_histogram_matching`, `gpu_mean_std_transfer`, `gpu_lab_transfer`, and `gpu_histogram_matching_multichannel`
  - `tests/test_config.py` -- 21 tests covering persistence and filtering for `ColorCastConfig`
  - `tests/test_comparison.py` -- 46 tests for `MethodComparison` metrics (PSNR, SSIM, color distance, histogram distance), method comparison orchestration, ranking with auto-inferred directions, report generation, `find_best_method`, and metric direction constants
  - `tests/test_visualization.py` -- 21 tests for all 4 figure-creation functions across `show_histograms` and `show_difference` parameter combinations
  - `tests/test_cli.py` -- 21 tests for `parse_args`, `cmd_transfer` (simulator, style-required, intensity blend), `cmd_batch`, `cmd_list_methods`, `cmd_info`, and `main` entry point
- Autouse `close_figures` fixture in `tests/test_visualization.py` closes all Matplotlib figures after every test, including parameterized cases, preventing figure handle accumulation between runs.
- Regression test `test_get_or_compute_concurrent_same_key` in `tests/test_cache.py` covers the double-check path in `get_or_compute`: two threads both miss the first lock check; the second caller returns the value stored by the first and the cache holds exactly one entry.

### Fixed

- `gpu_mean_std_transfer` divided by 255 and multiplied by 255 in the GPU branch despite the [0, 1] float input contract documented for the function and followed by the CPU fallback, producing silently wrong output when CuPy was available. The GPU branch now matches the [0, 1] contract.
- `selective_color_transfer` documented continuous blending masks ("0.0 to 1.0 for smooth blending at boundaries") but produced hard binary masks via `.astype(float)` on boolean arrays, leaving visible seams at shadow and highlight thresholds. Replaced with smoothstep feathered masks over a `\u00b1 0.05` luminance band.
- `StyleTransferCache` was not thread-safe (plain int counters and compound `OrderedDict` read-modify-write operations), though documented for use with the threaded batch path. Added a `threading.Lock` around all cache mutations.
- `validate_file_path` used `str.startswith` to check base-directory containment, admitting `/data-leaks` when the base was `/data`. Replaced with `Path.relative_to` in a try/except block.
- `_compute_hash` hashed the full image bytes on every cache get and set, making the LRU path slower than the transfer it was meant to cache for large images. Now hashes shape, dtype, strides, and a downsampled fingerprint instead.
- `enable_parallel` was a dead configuration field; it is now wired into `BatchProcessor` (caps `max_workers` to 1 when false).
- `MethodComparison.compute_histogram_distance` divided by `hist.sum()` without guarding zero, producing NaN metrics on empty or degenerate channels. Both histograms now guard the divisor.
- `visualize_method_comparison` in `colorcast/analysis/visualization.py` raised `UnboundLocalError` when called with a custom `figsize` because the `n_rows` variable was assigned only inside the `if figsize is None:` branch. `n_rows` is now computed before that branch.
- Added a missing import and updated docstrings in GPU functions.

## [2.4.0] - 2026-07-26

### Added

- `tests/test_validators_enhanced.py` with 11 tests covering extension-spoofed file rejection, NaN and Inf pixel detection, and malformed shape rejection via `normalize_to_float32`.

### Changed

- `normalize_to_float32` (`colorcast/processing/image_loader.py`) now validates that the input has shape `(H, W, 3)` before normalization, raising `ValueError` with the actual shape if not. Callers that pass 2D grayscale or 4-channel RGBA arrays must route through `ensure_rgb` first.
- Image loading now validates file magic numbers via `validate_image_file` to reject extension-spoofed files, and validates loaded arrays via `validate_image_array` to reject NaN/Inf pixel values. Imports in `colorcast/processing/image_loader.py` and `colorcast/utils/__init__.py` route through `colorcast.utils.validators_enhanced`, which is now the single authoritative validation module.
- `colorcast.utils.validators` reduced to a re-export shim that emits a `DeprecationWarning`; all logic moved to `colorcast.utils.validators_enhanced`.
- `transfer_methods.validate_and_resize_images` now normalizes both inputs to float32 in [0, 1] via `normalize_to_float32`, enforcing the documented value-range contract for all five transfer functions.
- `CITATION.cff` and `.zenodo.json` abstracts and keywords updated to lead with the accessibility pipeline and Daltonization.
- Removed `matplotlib.use('Agg')` from `visualization.py` (it followed `import pyplot` and had no effect).

### Fixed

- `MethodComparison.find_best_method` returned the worst method for distance metrics (`color_distance`, `histogram_distance`) because `rank_methods` defaulted to `ascending=False`. Metric direction is now inferred automatically from a module-level mapping; the `ascending` parameter defaults to `None` (infer from the metric).
- `MethodComparison.compute_histogram_distance` documented Earth Mover's Distance but computed L1 distance between normalized histograms, saturating at 2.0 for any non-overlapping pair. Replaced with 1-D EMD (`|cumsum(hist1) - cumsum(hist2)|`).
- `MethodComparison.compare_methods` computed baseline `color_distance` as `d(source, reference)` instead of zero, and the baseline row could appear in rankings and `find_best_method`. Baseline `color_distance` is now zero; baseline is excluded from ranking.
- `MethodComparison.compare_methods` had no per-method exception handling; one failing method aborted the whole comparison. Failures are now recorded with NaN metrics and an `_error` field, and excluded from ranking.
- `MethodComparison.generate_comparison_report` header and rule widths mismatched (75 vs 60). Both now use a single `rule_width` value.
- `ColorBlindSimulator.transform_color_space` had a redundant `(H, W, 3)` shape guard after `normalize_to_float32`, which already validates the shape.
- `plot_error_heatmap` could not display integer input because `np.clip` to [0, 1] preceded the uint8 rescale test. Replaced with `normalize_to_float32`.
- `plot_error_heatmap` colorbar label read `ΔE (Lab*)`, which falsely implies L* is included; corrected to `chroma error (a*, b\*)`.
- `comparison.py` migrated from `typing.Dict/List/Tuple` to `from __future__ import annotations` with builtin generics, matching the rest of the codebase.

v2.4.0 · released July 2026 · MIT

## [2.3.0] – 2026-07-23

### Added

- Three Daltonizer methods (`daltonize_protanopia`, `daltonize_deuteranopia`, `daltonize_tritanopia`) registered in `colorcast/processing/registry.py`, wrapping `colorcast.analysis.daltonization.daltonize`. All 15 GUI modes are now registered, and the Daltonizers are selectable through the CLI and API as well as the GUI.
- `TransferMethod.slider_label` declares the GUI intensity-slider label for each method. Simulator and Daltonizer methods override the default.

### Changed

- The GUI dispatches every method through `registry.get_method()`; the 14-branch `if`-`elif` chain in `apply_style_transfer` and the method-id prefix checks in `on_method_changed` were removed.
- The GUI slider label and style-image controls derive from method metadata (`requires_reference`, `slider_label`) instead of hardcoded category sets.

v2.3.0 · released July 2026 · MIT

## [2.2.0] – 2026-07-22

### Added

- `SUPPORTED_DEFICIENCIES` in `colorcast/processing/simulation.py`: the runtime tuple of deficiency names, derived from the `DeficiencyType` alias so validation and static typing share one source.
- `TransferMethod.requires_reference` indicates whether a method needs a reference (style) image. The three CVD simulator methods set it to False, and the CLI skips loading the style image for them; the `style` argument of `colorcast transfer` is now optional for `simulate_*` methods.
- `StyleTransferCache` accepts `style=None`, producing style-independent cache keys for reference-free methods.
- Reference-requiring registry methods raise `ValueError` when called with `reference=None`.
- `lab_reinhard` (Lab color transfer, Reinhard) is now selectable in the GUI method dropdown; it was previously registered but reachable only through the CLI and API.

### Changed

- `ensure_rgb` composites RGBA input onto a white background instead of discarding the alpha channel. Signed-integer and 64-bit-integer RGBA input raises `InvalidImageFormatError`.
- `save_image` accepts only float32, float64, and uint8 input and raises `InvalidImageFormatError` for anything else.
- `ColorBlindSimulator.transform_color_space` raises `ValueError` for input that is not `(H, W, 3)`.
- The CLI dispatches all transfer methods through `registry.get_method()`; the duplicate `get_transfer_function()` map in `colorcast/__main__.py` was removed.
- Tritanopia is dispatched through the same strategy registry as the other deficiencies, and its two-plane selection now computes both projections and merges them with `np.where`.
- `StyleTransferCache` LRU bookkeeping uses `collections.OrderedDict` instead of a hand-ordered access list.
- `BatchProcessor.process_directory` collects per-worker outcomes and merges them after `executor.map()`; worker threads no longer mutate shared failure state.
- `colorcast/gui.py` no longer calls `logging.basicConfig()` at import time; logging is configured in `main()`.

### Fixed

- CLI transfer methods missing from the old hardcoded dispatch map (`lab_reinhard`, the simulators) no longer fall back silently to histogram matching.
- `StyleTransferCache.set()` no longer evicts an unrelated entry when overwriting an existing key in a full cache.

v2.2.0 · released July 2026 · MIT

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
