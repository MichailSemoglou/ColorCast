# Commit Plan — ColorCast v2.6.0 working tree

> Generated 2026-08-05. Commits ordered for atomicity, buildability, and review flow.

## 1. Summary

- **Changed files:** 21 tracked, 5 untracked, 1 deleted, 1 DO NOT COMMIT
- **Proposed commits:** 7
- **Split files requiring `git add -p`:** 2 (`colorcast/gui.py`, `pyproject.toml`)

## 2. Excluded files

| File             | Reason                                                        |
| ---------------- | ------------------------------------------------------------- |
| `imgs/.DS_Store` | macOS system junk — already gitignored by `**/.DS_Store` rule |

Untracked files that SHOULD be committed (all accounted for in the clusters below):
`colorcast/analysis/appearance.py`, `colorcast/utils/color_utils.py`, `colorcast/assets/`, `imgs/dashboard_report_ICtCp.png`, `tests/test_appearance.py`.

## 3. Commit sequence

### Commit 1 — `feat: add perceptually uniform appearance spaces (ICtCp and CIELAB)`

**Why:** All new infrastructure for ICtCp and CIELAB color-difference metrics. This is the foundation that every subsequent commit depends on. It ships with a shared `srgb_to_linear` utility extracted from the duplicated simulation helper, plus a full test suite.

**Files:**

```
colorcast/analysis/appearance.py         (new — ICtCp/CIELAB ABC, backends, factory)
colorcast/utils/color_utils.py           (new — shared srgb_to_linear, extracted from simulation)
colorcast/processing/simulation.py       (modified — imports shared utility, removes local copy)
tests/test_appearance.py                 (new — 12 public-API + private-helper tests)
```

**Staging commands:**

```bash
git add colorcast/analysis/appearance.py
git add colorcast/utils/color_utils.py
git add colorcast/processing/simulation.py
git add tests/test_appearance.py
```

**Commit message:**

```
feat: add perceptually uniform appearance spaces (ICtCp and CIELAB)

Introduce an `AppearanceSpace` ABC with two backends:
- CIELAB (CIE 1976, CIE76 ΔE*ab or CIEDE2000 ΔE00 via skimage)
- ICtCp (ITU-R BT.2100, BT.2124-0 ΔE_ITP, HDR-aware)

The `make_appearance_space(name)` factory centralizes creation.
Extract the shared `srgb_to_linear` into `colorcast/utils/color_utils.py`
so both simulation and appearance can import it without duplication.

12 tests cover identical-image zero-ΔE, monotonicity, PQ/linear
helpers, metric names, and input validation.
```

**Completeness:** ✅ Standalone. `simulation.py` passes existing tests; `test_appearance.py` passes all 12.

---

### Commit 2 — `feat: wire appearance metrics into error maps and dashboard`

**Why:** Threads the appearance-space abstraction from Commit 1 into `ErrorMap`, `get_error_map`, `compute_dashboard`, and the CLI/GUI. Depends on Commit 1.

**Files (tracked, modified):**

```
colorcast/analysis/error_map.py          (appearance_delta fields, preferred_metric(), get_error_map parameter)
colorcast/analysis/dashboard.py          (DashboardResult.metric_label, _summarize refactor, appearance param)
colorcast/analysis/__init__.py           (new exports: AppearanceSpace, make_appearance_space, etc.)
colorcast/__main__.py                    (new cmd_dashboard with --appearance flag)
```

**Files (`git add -p` from `colorcast/gui.py`):**

```
hunk: import make_appearance_space
hunk: DashboardDialog.__init__ (self._appearance_combo, self._error, self._current_request_id)
hunk: _start_computation (request_id, appearance selection, local-import removal)
hunk: _restart_computation
hunk: _export_report (metric-derived filename and title)
hunk: _sanitize_filename helper
```

**Files (`git add -p` from `tests/test_gui.py`):**

```
hunk: test_heatmap_title_uses_selected_metric_label
hunk: test_stale_completion_does_not_mutate_current_result (fix)
```

**Staging commands:**

```bash
# Tracked files — stage whole
git add colorcast/analysis/error_map.py
git add colorcast/analysis/dashboard.py
git add colorcast/analysis/__init__.py
git add colorcast/__main__.py

# Split files — stage hunk by hunk
git add -p colorcast/gui.py
#   accept: import make_appearance_space
#   accept: DashboardDialog.__init__ additions
#   accept: _start_computation changes
#   accept: _restart_computation
#   accept: _export_report changes
#   accept: _sanitize_filename helper
#   skip:   _APP_STYLESHEET constant (→ Commit 6)
#   skip:   main() stylesheet block (→ Commit 6)
#   skip:   QDialog/QMessageBox/QFileDialog style rules (→ Commit 6)

git add -p tests/test_gui.py
#   accept: test_heatmap_title_uses_selected_metric_label
#   accept: test_stale_completion_does_not_mutate_current_result fix

# Images — all deferred to Commit 7 (docs)
```

**Commit message:**

```
feat: wire appearance metrics into error maps and dashboard

Add appearance_delta / appearance_delta_name fields to ErrorMap
and a preferred_metric() method that centralizes the priority rule
(appearance > CIEDE2000 > chroma).  get_error_map() accepts an
appearance= keyword; compute_dashboard() passes it through.

New CLI subcommand `colorcast dashboard IMAGE --appearance ictcp`.
GUI Dashboard dialog gains a ΔE-metric dropdown that re-triggers
computation.  Report filenames and titles auto-include the metric
label (e.g. dashboard_report_ictcp.png, "CVD Accessibility Dashboard
– ICtCp").

Fix a GUI test that did not account for the initial _start_computation
call from DashboardDialog.__init__, causing it to see 3 pending
requests instead of 2.
```

**Completeness:** ⚠️ Depends on Commit 1 (`colorcast.analysis.appearance`). Builds and all tests pass only after Commit 1 is applied.

---

### Commit 3 — `fix: improve image loader decompression bomb defence and multi-frame rejection`

**Why:** Hardens `_get_image_dimensions` against Pillow `DecompressionBombError` (re-raises instead of falling through to a full decode) and rejects multi-frame TIFF/stacked arrays. Standalone bug-fix / hardening with its own tests.

**Files:**

```
colorcast/processing/image_loader.py    (DecompressionBombError guard, _read_image_array, multi-frame rejection)
tests/test_image_loading.py             (2 new tests: bomb guard, multi-frame rejection)
```

**Staging commands:**

```bash
git add colorcast/processing/image_loader.py
git add tests/test_image_loading.py
```

**Commit message:**

```
fix: improve image loader decompression bomb defence and multi-frame rejection

- _get_image_dimensions: catch and re-raise Pillow DecompressionBombError
  so oversized images stop at header read instead of triggering a full
  decode fallback.  Restore Pillow.MAX_IMAGE_PIXELS to its original value
  after the header check.
- Extract _read_image_array for single-frame validation; reject
  multi-frame stacks (ndim > 3 or ambiguous channel count).
- Add tests for DecompressionBombError propagation and multi-frame
  detection via skimage/imread monkeypatching.
```

**Completeness:** ✅ Standalone. Tests pass independently.

---

### Commit 4 — `fix: correct config boolean validation against int subclass`

**Why:** `colorcast.utils.config.ColorCastConfig.load()` used `isinstance(value, int)` to validate integer fields, but Python's `bool` is a subclass of `int`. Boolean JSON values (`true`/`false`) were silently accepted where an integer was expected. The fix explicitly rejects `bool` for `int`-typed fields unless the field declares `bool`.

Also reverts the `mypy` target from `3.12` back to `3.10` in `pyproject.toml` to match the project's stated support floor.

**Files:**

```
colorcast/utils/config.py               (bool/int guard, import cleanup)
```

**Files (`git add -p` from `pyproject.toml`):**

```
hunk: [tool.mypy] python_version = "3.10" (the one-line change)
```

**Staging commands:**

```bash
git add colorcast/utils/config.py
git add -p pyproject.toml
#   accept: python_version = "3.10"
#   skip:   package-data (→ Commit 6)
```

**Commit message:**

```
fix: correct config boolean validation against int subclass

ColorCastConfig.load() now rejects JSON booleans for integer-typed
fields (bool is a subclass of int, so isinstance alone was not enough).
Floats are still accepted for integer fields because JSON cannot
distinguish them.

Also revert mypy python_version to 3.10 to match the project's
minimum supported Python.
```

**Completeness:** ✅ Standalone. Config tests pass.

---

### Commit 5 — `chore: simplify gpu_transfer module docstring`

**Why:** Cosmetic-only cleanup of the `gpu_transfer.py` module docstring. Removes historical `gpu_`-prefix justification and tightens prose. No behavioral change.

**Files:**

```
colorcast/processing/gpu_transfer.py    (docstring-only)
```

**Staging commands:**

```bash
git add colorcast/processing/gpu_transfer.py
```

**Commit message:**

```
chore: simplify gpu_transfer module docstring

Remove historical `gpu_`-prefix justification prose.  Tighten the
description of the fallback contract without changing any behaviour.
```

**Completeness:** ✅ Standalone. No tests affected.

---

### Commit 6 — `feat: redesign GUI with minimal dark theme`

**Why:** Replaces the QDarkStyle dependency with a custom QSS stylesheet following Ollama-inspired minimal design: dark canvas (#171717), white pill buttons, hairline borders, zero drop shadows. Adds a combo-box chevron asset. Screenshots (`imgs/interface*.png`) reflect the new theme.

Depends on Commit 2 for the `make_appearance_space` import and the `DashboardDialog` changes that this commit's `gui.py` stylesheet block sits alongside.

**Files (`git add -p` from `colorcast/gui.py` — the remaining hunks):**

```
hunk: _APP_STYLESHEET constant (the entire QSS block)
hunk: main() stylesheet assignment
hunk: QDialog/QMessageBox/QFileDialog style rules
```

**New assets:**

```
colorcast/assets/                         (new directory)
colorcast/assets/chevron-down.png         (new — combo box dropdown arrow)
```

**Config files:**

```
.gitignore                               (assets exception: !/colorcast/assets/)
MANIFEST.in                              (assets inclusion: recursive-include)

```

**Files (`git add -p` from `pyproject.toml`):**

```
hunk: [tool.setuptools.package-data] colorcast = ["assets/*.png"]
```

**Staging commands:**

```bash
# Remaining gui.py hunks
git add -p colorcast/gui.py
#   accept: _APP_STYLESHEET constant
#   accept: main() stylesheet
#   accept: QDialog/QMessageBox/QFileDialog rules

# Assets
git add colorcast/assets/chevron-down.png

# Config
git add .gitignore MANIFEST.in
git add -p pyproject.toml
#   accept: [tool.setuptools.package-data] (only if not already staged from Commit 4)

# Screenshots — deferred to Commit 7 (docs)
```

**Commit message:**

```
feat: redesign GUI with minimal dark theme

Replace the QDarkStyle dependency with a custom QSS stylesheet:
- Dark canvas (#171717), white pill buttons, hairline borders (#3d3d3d)
- Zero drop shadows, flat card surfaces, system sans-serif fonts
- Square-cornered sliders, pill-shaped dropdowns with custom chevron
- Combo box dropdowns use a bundled PNG chevron asset

.gitignore, MANIFEST.in, and pyproject.toml updated to ship the
assets directory.  Interface screenshots updated to reflect the new
theme.
```

**Completeness:** ⚠️ Depends on Commit 2. The `gui.py` stylesheet module imports `make_appearance_space`. All GUI tests pass.

---

### Commit 7 — `docs: update README and CHANGELOG for v2.6.0`

**Why:** Documentation reflecting all the above: CHANGELOG [2.6.0] entries, README test counts (351→378), new features list, CLI example, and Python API example for appearance spaces. This is the last commit; it documents everything that came before.

**Files:**

```
README.md                               (test counts, feature list, CLI/python examples)
CHANGELOG.md                            ([2.6.0] Added/Changed sections)
```

**Screenshots (all imgs/ changes):**

```
imgs/CVD-Accessibility-Dashboard.png     (modified — updated dashboard layout)
imgs/Compare-Transfer-Methods.png        (modified)
imgs/dashboard_report.png                (deleted — replaced by metric-labelled version)
imgs/dashboard_report_ICtCp.png          (new — metric-labelled sample)
imgs/interface.png                       (modified — new theme)
imgs/interface_2.png                     (modified — new theme)
```

**Staging commands:**

```bash
git add README.md CHANGELOG.md
git add imgs/CVD-Accessibility-Dashboard.png imgs/Compare-Transfer-Methods.png
git add imgs/dashboard_report_ICtCp.png
git rm imgs/dashboard_report.png
git add imgs/interface.png imgs/interface_2.png
```

**Commit message:**

```
docs: update README and CHANGELOG for v2.6.0

- README test metrics: 378 passing, 1 skipped (up from 361)
- README new features: appearance spaces, dashboard CLI, GUI theme
- README CLI and Python API examples for appearance + dashboard
- CHANGELOG [2.6.0] entries covering all 6 prior commits
```

**Completeness:** ✅ Standalone documentation. No code dependency.

---

## 4. Open questions

1. **Q: Should the `imgs/.DS_Store` file be explicitly git-removed, or is the existing `.gitignore` rule sufficient?** It appears to be caught by an existing `**/.DS_Store` pattern. If it somehow escaped, add `imgs/.DS_Store` to `.gitignore` explicitly in Commit 6.
2. **Q: Is the `mypy python_version` change in Commit 4 intentional?** ✅ Confirmed — intentional. Stays in Commit 4.
