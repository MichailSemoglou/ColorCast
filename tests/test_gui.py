"""Headless GUI smoke tests for ColorCast.

Run with: QT_QPA_PLATFORM=offscreen pytest tests/test_gui.py
"""

import os

# Force offscreen rendering before any Qt import
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import numpy as np
import pytest
from PyQt5 import QtCore


@pytest.fixture(scope="module")
def qt_app():
    """Create a QApplication for headless tests."""
    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def _select_combo_method(combo, method_id):
    """Select a method in the combo box by its registered ID."""
    for row in range(combo.count()):
        item = combo.model().item(row)
        if item.data(QtCore.Qt.UserRole) == method_id:
            combo.setCurrentIndex(row)
            return
    raise ValueError(f"Method {method_id!r} not found in combo")


class TestMethodSwitchTogglesStyleButton:
    """Method-change toggles the style button and label correctly."""

    def test_reference_method_enables_style_button(self, qt_app):
        from colorcast.gui import StyleTransferApp

        window = StyleTransferApp()
        assert window.load_style_button.isEnabled()

    def test_simulator_disables_style_button(self, qt_app):
        from colorcast.gui import StyleTransferApp

        window = StyleTransferApp()
        _select_combo_method(window.method_combo, "simulate_protanopia")

        assert not window.load_style_button.isEnabled()

    def test_simulator_sets_placeholder_text(self, qt_app):
        from colorcast.gui import StyleTransferApp

        window = StyleTransferApp()
        _select_combo_method(window.method_combo, "simulate_deuteranopia")

        assert "Not needed for this mode" in window.style_label.text()

    def test_switch_back_to_reference_enables_button(self, qt_app):
        from colorcast.gui import StyleTransferApp

        window = StyleTransferApp()

        _select_combo_method(window.method_combo, "simulate_protanopia")
        assert not window.load_style_button.isEnabled()

        _select_combo_method(window.method_combo, "histogram")
        assert window.load_style_button.isEnabled()


class TestApplyWithNoImages:
    """apply_style_transfer with missing images shows warning."""

    def test_apply_without_content_shows_warning(self, qt_app, monkeypatch):
        from PyQt5.QtWidgets import QMessageBox

        from colorcast.gui import StyleTransferApp

        window = StyleTransferApp()
        warning_calls = []

        def fake_warning(parent, title, msg):
            warning_calls.append((title, msg))

        monkeypatch.setattr(QMessageBox, "warning", fake_warning)

        window.apply_style_transfer()

        assert len(warning_calls) == 1
        assert warning_calls[0][0] == "Missing Images"

    def test_apply_without_style_on_reference_method_shows_warning(self, qt_app, monkeypatch):
        from PyQt5.QtWidgets import QMessageBox

        from colorcast.gui import StyleTransferApp

        window = StyleTransferApp()
        window.content_image = np.random.rand(16, 16, 3).astype(np.float32)
        window.style_image = None

        warning_calls = []

        def fake_warning(parent, title, msg):
            warning_calls.append((title, msg))

        monkeypatch.setattr(QMessageBox, "warning", fake_warning)

        window.apply_style_transfer()

        assert len(warning_calls) == 1
        assert "style image" in warning_calls[0][1].lower()


class TestShowDashboard:
    """show_dashboard guards against missing content image."""

    def test_no_content_image_shows_warning(self, qt_app, monkeypatch):
        from PyQt5.QtWidgets import QMessageBox

        from colorcast.gui import StyleTransferApp

        window = StyleTransferApp()
        warning_calls = []

        def fake_warning(parent, title, msg):
            warning_calls.append((title, msg))

        monkeypatch.setattr(QMessageBox, "warning", fake_warning)

        window.show_dashboard()

        assert len(warning_calls) == 1
        assert "content image" in warning_calls[0][1].lower()


def _make_dashboard_result(h: int = 8, w: int = 8):
    """Build a minimal DashboardResult for unit tests."""
    from colorcast.analysis.dashboard import DashboardResult
    from colorcast.analysis.error_map import ErrorMap

    rng = np.random.default_rng(0)
    original = rng.random((h, w, 3)).astype(np.float32)
    sim = rng.random((h, w, 3)).astype(np.float32)
    chroma_err = (rng.random((h, w)) * 10).astype(np.float32)
    em = ErrorMap(
        signed=np.zeros((h, w, 3), np.float32),
        absolute=np.zeros((h, w, 3), np.float32),
        chroma_error=chroma_err,
        chroma_error_dE00=chroma_err.copy(),
        signed_chroma_ab=np.zeros((h, w, 2), np.float32),
        orig_l_star=np.zeros((h, w), np.float32),
    )
    deficiencies = ("protanopia", "deuteranopia", "tritanopia")
    return original, DashboardResult(
        original=original,
        simulated=dict.fromkeys(deficiencies, sim),
        error_maps=dict.fromkeys(deficiencies, em),
        summary={
            d: {
                "mean_error": 1.0,
                "median_error": 0.5,
                "p95_error": 2.0,
                "percent_affected": 30.0,
            }
            for d in deficiencies
        },
    )


def test_heatmap_title_uses_selected_metric_label():
    from colorcast.analysis.dashboard import _heatmap_title

    assert _heatmap_title("protanopia", "ICtCp") == "ICtCp (P)"
    assert _heatmap_title("deuteranopia", "CIEDE2000") == "CIEDE2000 (D)"


class TestDashboardDialogComputationRequests:
    """DashboardDialog request lifecycle keeps the report button state in sync."""

    def test_start_computation_disables_report_button(self, qt_app, monkeypatch):
        """Verify that dashboard work disables the report button.

        The test uses the qt_app fixture to provide a Qt application context and
        the monkeypatch fixture to replace the worker launcher and dashboard
        computation with no-op functions. The test returns None because it only
        asserts on the widget state.
        """
        from colorcast.gui import DashboardDialog

        monkeypatch.setattr("colorcast.gui._run_in_thread", lambda target, *, on_done=None: None)

        dialog = DashboardDialog(np.random.rand(8, 8, 3).astype(np.float32))
        dialog._report_button.setEnabled(True)

        dialog._start_computation()

        assert dialog._report_button is not None
        assert not dialog._report_button.isEnabled()


class TestDashboardDialogPopulateResults:
    """DashboardDialog._populate_results with a precomputed DashboardResult."""

    def test_summary_label_populated_and_report_button_enabled(self, qt_app, monkeypatch):
        from colorcast.gui import DashboardDialog

        monkeypatch.setattr(DashboardDialog, "_start_computation", lambda self: None)

        original, result = _make_dashboard_result()
        dialog = DashboardDialog(original)
        dialog._result = result
        dialog._populate_results()

        assert dialog._summary_label is not None
        text = dialog._summary_label.text()
        assert "Protanopia" in text
        assert dialog._report_button is not None
        assert dialog._report_button.isEnabled()

    def test_stale_completion_does_not_mutate_current_result(self, qt_app, monkeypatch):
        import colorcast.gui as gui_module
        from colorcast.gui import DashboardDialog

        pending = []

        def fake_run(target, *, on_done=None):
            pending.append((target, on_done))

        monkeypatch.setattr(gui_module, "_run_in_thread", fake_run)
        monkeypatch.setattr(
            gui_module, "compute_dashboard", lambda image, appearance=None: object()
        )

        # Suppress the initial _start_computation call from __init__ so the
        # test controls exactly how many requests are queued.
        _original_start = DashboardDialog._start_computation
        DashboardDialog._start_computation = lambda self: None
        try:
            dialog = DashboardDialog(np.random.rand(8, 8, 3).astype(np.float32))
        finally:
            DashboardDialog._start_computation = _original_start

        monkeypatch.setattr(dialog, "_populate_results", lambda: None)

        dialog._start_computation()
        dialog._start_computation()
        assert len(pending) == 2

        pending[0][0]()
        pending[0][1]()

        assert dialog._result is None
        assert dialog._error is None

        pending[1][0]()
        pending[1][1]()

        assert dialog._result is not None
        assert dialog._error is None
