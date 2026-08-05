"""PyQt5 graphical interface for ColorCast."""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from pathlib import Path
from threading import Thread

import numpy as np
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPalette, QPixmap, QStandardItem, QStandardItemModel
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from colorcast import blend_images, registry, save_image
from colorcast.analysis.appearance import make_appearance_space
from colorcast.analysis.dashboard import (
    _DEFICIENCIES,
    _DEFICIENCY_LABELS,
    DashboardResult,
    _heatmap_title,
    compute_dashboard,
    format_summary_table,
    generate_dashboard_report,
)
from colorcast.processing.image_loader import ImageMeta, load_image_with_meta
from colorcast.utils.config import ColorCastConfig
from colorcast.utils.exceptions import ImageProcessingError
from colorcast.utils.validators_enhanced import ALLOWED_IMAGE_EXTENSIONS

logger = logging.getLogger(__name__)


def _sanitize_filename(name: str) -> str:
    """Turn an appearance-space label into a safe filename fragment."""
    return name.lower().replace(" ", "_").replace("(", "").replace(")", "")


# Single source of truth for the application theme. Deep neutral canvas,
# card surfaces one step lighter, hairline borders, and a monochrome accent
# (white primary actions). Widget variants are driven by dynamic properties:
# caption, wordmark, value, imageCard, cellTitle, status, summaryCard on
# QLabel, and secondary on QPushButton.
_ASSETS_DIR = Path(__file__).resolve().parent / "assets"
_CHEVRON_PATH = (_ASSETS_DIR / "chevron-down.png").as_posix()

_STYLESHEET_TEMPLATE = """
/* Base canvas */
QWidget {
    background-color: #0E0E10;
    color: #F2F2F2;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    font-size: 13px;
}
QDialog, QMessageBox, QFileDialog {
    background-color: #0E0E10;
}

/* Labels */
QLabel {
    background-color: transparent;
}
QLabel[wordmark="true"] {
    font-size: 19px;
    font-weight: 700;
}
QLabel[caption="true"] {
    color: #8B8B93;
    font-size: 11px;
    font-weight: 600;
}
QLabel[value="true"] {
    font-weight: 600;
}
QLabel[imageCard="true"] {
    background-color: #131316;
    border: 1px solid #26262B;
    border-radius: 12px;
    color: #5C5C64;
}
QLabel[cellTitle="true"] {
    color: #B9B9C0;
    font-size: 12px;
}
QLabel[status="true"] {
    color: #8B8B93;
}
QLabel[summaryCard="true"] {
    background-color: #131316;
    border: 1px solid #26262B;
    border-radius: 8px;
    padding: 10px;
    font-family: "Menlo", "Consolas", "DejaVu Sans Mono", "Courier New", monospace;
    font-size: 12px;
}

/* Buttons: default is the primary action */
QPushButton {
    background-color: #F2F2F2;
    color: #0E0E10;
    border: 1px solid #F2F2F2;
    border-radius: 8px;
    padding: 9px 18px;
    font-weight: 600;
    min-height: 18px;
}
QPushButton:hover {
    background-color: #FFFFFF;
    border-color: #FFFFFF;
}
QPushButton:pressed {
    background-color: #D9D9DE;
    border-color: #D9D9DE;
}
QPushButton:focus {
    border-color: #FFFFFF;
}
QPushButton:disabled {
    background-color: #232327;
    border-color: #232327;
    color: #55555C;
}
QPushButton[secondary="true"] {
    background-color: transparent;
    color: #C9C9CF;
    border: 1px solid #343439;
    font-weight: 500;
}
QPushButton[secondary="true"]:hover {
    color: #FFFFFF;
    border-color: #4A4A52;
    background-color: #19191D;
}
QPushButton[secondary="true"]:pressed {
    background-color: #222227;
}
QPushButton[secondary="true"]:focus {
    border-color: #56565E;
}
QPushButton[secondary="true"]:disabled {
    color: #4A4A51;
    border-color: #26262B;
    background-color: transparent;
}

/* Slider */
QSlider::groove:horizontal {
    height: 4px;
    background: #2A2A2E;
    border-radius: 2px;
}
QSlider::sub-page:horizontal {
    background: #F2F2F2;
    border-radius: 2px;
}
QSlider::add-page:horizontal {
    background: #2A2A2E;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #FFFFFF;
    width: 16px;
    height: 16px;
    margin: -7px 0;
    border-radius: 4px;
}
QSlider::handle:horizontal:hover {
    background: #E4E4E8;
}

/* Combo box */
QComboBox {
    background-color: #1E1E22;
    border: 1px solid #2F2F35;
    border-radius: 8px;
    padding: 8px 14px;
    min-height: 18px;
}
QComboBox:hover {
    border-color: #44444C;
}
QComboBox:focus {
    border-color: #56565E;
}
QComboBox::drop-down {
    border: none;
    width: 26px;
}
QComboBox::down-arrow {
    image: url("@CHEVRON_DOWN@");
    width: 12px;
    height: 12px;
}
/* The popup container is a top-level window; paint it with the list
   surface color so no unpainted frame shows around the rounded view. */
QComboBoxPrivateContainer {
    background-color: #17171A;
}
QComboBox QAbstractItemView {
    background-color: #17171A;
    border: 1px solid #2F2F35;
    border-radius: 10px;
    padding: 6px;
    outline: none;
    selection-background-color: #2C2C33;
    selection-color: #FFFFFF;
}
QComboBox QAbstractItemView::item {
    padding: 7px 10px;
    border-radius: 6px;
    color: #E8E8EC;
}
QComboBox QAbstractItemView::item:hover {
    background-color: #222228;
}
QComboBox QAbstractItemView::item:disabled {
    color: #55555C;
    background-color: transparent;
}

/* Tooltips */
QToolTip {
    background-color: #1E1E22;
    color: #F2F2F2;
    border: 1px solid #2F2F35;
    padding: 6px 8px;
    border-radius: 6px;
}
"""

_APP_STYLESHEET = _STYLESHEET_TEMPLATE.replace("@CHEVRON_DOWN@", f'url("{_CHEVRON_PATH}")')


def _caption_label(text: str, *, align_right: bool = False) -> QLabel:
    """Build a small section caption with letter spacing."""
    label = QLabel(text)
    label.setProperty("caption", True)
    font = label.font()
    font.setLetterSpacing(QtGui.QFont.PercentageSpacing, 112)
    label.setFont(font)
    if align_right:
        label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
    return label


def _dark_palette() -> QPalette:
    """Return a palette matching the stylesheet for chrome QSS cannot reach.

    Popup frames, native dialogs, and fallback rendering (for example the
    combo box drop-down arrow area) read their colors from the application
    palette, so it must agree with the dark stylesheet.
    """
    palette = QPalette()
    palette.setColor(QPalette.Window, QtGui.QColor("#0E0E10"))
    palette.setColor(QPalette.WindowText, QtGui.QColor("#F2F2F2"))
    palette.setColor(QPalette.Base, QtGui.QColor("#131316"))
    palette.setColor(QPalette.AlternateBase, QtGui.QColor("#17171A"))
    palette.setColor(QPalette.Text, QtGui.QColor("#F2F2F2"))
    palette.setColor(QPalette.Button, QtGui.QColor("#1E1E22"))
    palette.setColor(QPalette.ButtonText, QtGui.QColor("#F2F2F2"))
    palette.setColor(QPalette.ToolTipBase, QtGui.QColor("#1E1E22"))
    palette.setColor(QPalette.ToolTipText, QtGui.QColor("#F2F2F2"))
    palette.setColor(QPalette.Highlight, QtGui.QColor("#2C2C33"))
    palette.setColor(QPalette.HighlightedText, QtGui.QColor("#FFFFFF"))
    palette.setColor(QPalette.PlaceholderText, QtGui.QColor("#8B8B93"))
    palette.setColor(QPalette.Disabled, QPalette.Text, QtGui.QColor("#55555C"))
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, QtGui.QColor("#55555C"))
    palette.setColor(QPalette.Disabled, QPalette.WindowText, QtGui.QColor("#55555C"))
    return palette


def _apply_dark_window_chrome() -> None:
    """Give macOS window title bars the dark appearance.

    Qt 5 does not expose NSAppearance, so a dark stylesheet and palette
    still leave the native title bar light. Set the application appearance
    through the Objective-C runtime instead. Best effort: on any failure
    the default chrome stays, which only affects the title bar tint.
    """
    if sys.platform != "darwin":
        return
    try:
        import ctypes
        import ctypes.util

        void_p = ctypes.c_void_p
        objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))
        appkit = ctypes.cdll.LoadLibrary(ctypes.util.find_library("AppKit"))

        objc.objc_getClass.restype = void_p
        objc.objc_getClass.argtypes = [ctypes.c_char_p]
        objc.sel_registerName.restype = void_p
        objc.sel_registerName.argtypes = [ctypes.c_char_p]

        ns_app = objc.objc_getClass(b"NSApplication")
        objc.objc_msgSend.restype = void_p
        objc.objc_msgSend.argtypes = [void_p, void_p]
        app = objc.objc_msgSend(ns_app, objc.sel_registerName(b"sharedApplication"))

        ns_appearance = objc.objc_getClass(b"NSAppearance")
        dark_name = void_p.in_dll(appkit, "NSAppearanceNameDarkAqua")
        objc.objc_msgSend.argtypes = [void_p, void_p, void_p]
        dark = objc.objc_msgSend(
            ns_appearance, objc.sel_registerName(b"appearanceNamed:"), dark_name
        )
        objc.objc_msgSend(app, objc.sel_registerName(b"setAppearance:"), dark)
    except Exception:  # noqa: BLE001 - cosmetic best effort
        logger.debug("Could not apply dark window chrome", exc_info=True)


def _array_to_pixmap(
    array: np.ndarray,
    cell_size: int,
    normalize_heatmap: bool = False,
) -> QPixmap:
    """Convert a numpy image array to a QPixmap scaled to ``cell_size``."""
    if normalize_heatmap:
        vmax = float(array.max())
        array = array / vmax if vmax > 1e-6 else np.zeros_like(array)
    else:
        array = np.clip(array, 0, 1)
    img = (array * 255).astype(np.uint8)
    img = np.ascontiguousarray(img)
    h, w = img.shape[:2]
    if img.ndim == 2 or img.shape[2] == 1:
        qt_image = QImage(img.tobytes(), w, h, w, QImage.Format_Grayscale8)
    else:
        qt_image = QImage(img.tobytes(), w, h, img.shape[2] * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qt_image).scaled(
        cell_size, cell_size, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
    )


class _WorkerSignals(QtCore.QObject):
    """Queued signal bridge that marshals completion from a worker thread back to the GUI thread."""

    finished = QtCore.pyqtSignal()


# Qt can collect an unreferenced QObject before a queued signal is
# delivered, so in-flight bridges are held here until their slot has run.
# Both add and discard happen on the GUI thread, so no locking is needed.
_LIVE_WORKER_SIGNALS: set[_WorkerSignals] = set()


def _run_in_thread(
    target: Callable[[], None], *, on_done: Callable[[], None] | None = None
) -> None:
    """Run ``target`` off the GUI thread, calling ``on_done`` via a queued signal."""
    signals = _WorkerSignals()
    _LIVE_WORKER_SIGNALS.add(signals)

    def _on_done() -> None:
        try:
            if on_done is not None:
                on_done()
        finally:
            _LIVE_WORKER_SIGNALS.discard(signals)

    signals.finished.connect(_on_done)

    def _worker() -> None:
        try:
            target()
        finally:
            signals.finished.emit()

    Thread(target=_worker, daemon=True).start()


class StyleTransferApp(QWidget):
    """Main application window for ColorCast style transfer."""

    def __init__(self, config: ColorCastConfig | None = None) -> None:
        """Initialize the ColorCast application.

        Args:
            config: Optional configuration.  When omitted, defaults are
                taken from ``ColorCastConfig()``.
        """
        super().__init__()
        cfg = config or ColorCastConfig()
        self._window_width: int = cfg.window_width
        self._window_height: int = cfg.window_height
        self._preview_size: int = cfg.preview_size
        self._intensity: float = cfg.default_intensity
        self._slider_debounce_ms: int = cfg.slider_debounce_ms
        self._current_request_id: int = 0

        self.content_image: np.ndarray | None = None
        self.style_image: np.ndarray | None = None
        self.result_image: np.ndarray | None = None
        self.styled_image: np.ndarray | None = None
        self.intensity: float = self._intensity
        self.transfer_method: str = cfg.default_method
        self.update_timer: QTimer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self.apply_intensity_blend)
        self.resize(self._window_width, self._window_height)
        self.setMinimumSize(3 * self._preview_size + 80, 660)
        self.initUI()
        self.setStyleSheet(_APP_STYLESHEET)

    def initUI(self) -> None:
        """Initialize the user interface."""
        self.setWindowTitle("ColorCast")

        self.content_label: QLabel = QLabel(self)
        self.style_label: QLabel = QLabel(self)
        self.result_label: QLabel = QLabel(self)

        for label in [self.content_label, self.style_label, self.result_label]:
            label.setMinimumSize(self._preview_size, self._preview_size)
            label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            label.setProperty("imageCard", True)
            label.setAlignment(QtCore.Qt.AlignCenter)
            label.setText("No image loaded")

        self.intensity_slider: QSlider = QSlider(QtCore.Qt.Horizontal, self)
        self.intensity_slider.setMinimum(0)
        self.intensity_slider.setMaximum(100)
        self.intensity_slider.setValue(int(self._intensity * 100))
        self.intensity_slider.valueChanged.connect(self.update_intensity)

        self.intensity_label: QLabel = QLabel(f"{int(self._intensity * 100)}%", self)
        self.intensity_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.intensity_label.setFixedWidth(48)
        self.intensity_label.setProperty("value", True)

        self.method_combo: QComboBox = QComboBox(self)

        # Build a categorized model with non-selectable section headers
        _combo_model = QStandardItemModel(self.method_combo)

        def _header(text: str) -> QStandardItem:
            """Create a bold, non-selectable section-header row."""
            item = QStandardItem(text)
            item.setEnabled(False)
            item.setSelectable(False)
            font = item.font()
            font.setBold(True)
            item.setFont(font)
            item.setForeground(QtGui.QColor("#8B8B93"))
            return item

        def _entry(label: str, data: str) -> QStandardItem:
            """Create an indented, selectable method item."""
            item = QStandardItem("    " + label)
            item.setData(data, QtCore.Qt.UserRole)
            return item

        _combo_model.appendRow(_header("  STYLIZE"))
        _combo_model.appendRow(_entry("Histogram Match", "histogram"))
        _combo_model.appendRow(_entry("Reinhard (Mean/Std)", "meanstd"))
        _combo_model.appendRow(_entry("Lab (Reinhard)", "lab_reinhard"))
        _combo_model.appendRow(_entry("LUT: Linear", "lut_linear"))
        _combo_model.appendRow(_entry("LUT: S-Curve", "lut_scurve"))
        _combo_model.appendRow(_entry("LUT: Contrast", "lut_contrast"))
        _combo_model.appendRow(_entry("Selective: Shadows", "selective_shadows"))
        _combo_model.appendRow(_entry("Selective: Midtones", "selective_midtones"))
        _combo_model.appendRow(_entry("Selective: Highlights", "selective_highlights"))
        _combo_model.appendRow(_header("  SIMULATE"))
        _combo_model.appendRow(_entry("Protanopia", "simulate_protanopia"))
        _combo_model.appendRow(_entry("Deuteranopia", "simulate_deuteranopia"))
        _combo_model.appendRow(_entry("Tritanopia", "simulate_tritanopia"))
        _combo_model.appendRow(_header("  CORRECT"))
        _combo_model.appendRow(_entry("Daltonize (P)", "daltonize_protanopia"))
        _combo_model.appendRow(_entry("Daltonize (D)", "daltonize_deuteranopia"))
        _combo_model.appendRow(_entry("Daltonize (T)", "daltonize_tritanopia"))

        self.method_combo.setModel(_combo_model)
        self.method_combo.setCurrentIndex(1)  # start on "Histogram Match"
        self.method_combo.setMinimumWidth(230)
        self.method_combo.currentIndexChanged.connect(self.on_method_changed)

        self.load_content_button: QPushButton = QPushButton("Load Content Image", self)
        self.load_style_button: QPushButton = QPushButton("Load Style Image", self)
        self.apply_button: QPushButton = QPushButton("Apply Style Transfer", self)
        self.clear_button: QPushButton = QPushButton("Clear Images", self)
        self.save_button: QPushButton = QPushButton("Save Result", self)
        self.dashboard_button: QPushButton = QPushButton("Dashboard", self)
        self.compare_button: QPushButton = QPushButton("Compare Methods", self)

        for button in [
            self.load_content_button,
            self.load_style_button,
            self.clear_button,
            self.dashboard_button,
            self.compare_button,
        ]:
            button.setProperty("secondary", True)

        self.load_content_button.clicked.connect(self.load_content_image)
        self.load_style_button.clicked.connect(self.load_style_image)
        self.apply_button.clicked.connect(self.apply_style_transfer)
        self.clear_button.clicked.connect(self.clear_images)
        self.save_button.clicked.connect(self.save_result)
        self.dashboard_button.clicked.connect(self.show_dashboard)
        self.compare_button.clicked.connect(self.show_comparison)

        # Header: brand on the left, method picker on the right.
        wordmark = QLabel("ColorCast", self)
        wordmark.setProperty("wordmark", True)
        brand_column = QVBoxLayout()
        brand_column.setSpacing(2)
        brand_column.setContentsMargins(0, 0, 0, 0)
        brand_column.addWidget(wordmark)
        brand_column.addWidget(_caption_label("COLOR AND STYLE TRANSFER"))

        method_column = QVBoxLayout()
        method_column.setSpacing(6)
        method_column.setContentsMargins(0, 0, 0, 0)
        method_column.addWidget(_caption_label("METHOD", align_right=True))
        method_column.addWidget(self.method_combo)

        header_layout = QHBoxLayout()
        header_layout.addLayout(brand_column)
        header_layout.addStretch()
        header_layout.addLayout(method_column)

        # Image stage: three equal card columns.
        stage_layout = QHBoxLayout()
        stage_layout.setSpacing(16)
        stage_layout.addLayout(
            self._stage_column("CONTENT", self.content_label, self.load_content_button), 1
        )
        stage_layout.addLayout(
            self._stage_column("STYLE", self.style_label, self.load_style_button), 1
        )
        stage_layout.addLayout(
            self._stage_column("RESULT", self.result_label, self.apply_button), 1
        )

        # Intensity: heading and readout on one row, slider full width below.
        self.intensity_heading_label: QLabel = QLabel("Style Intensity:", self)
        self.intensity_heading_label.setProperty("caption", True)
        intensity_row = QHBoxLayout()
        intensity_row.addWidget(self.intensity_heading_label)
        intensity_row.addStretch()
        intensity_row.addWidget(self.intensity_label)

        intensity_column = QVBoxLayout()
        intensity_column.setSpacing(8)
        intensity_column.addLayout(intensity_row)
        intensity_column.addWidget(self.intensity_slider)

        # Footer: secondary actions left, primary save right.
        footer_layout = QHBoxLayout()
        footer_layout.setSpacing(10)
        footer_layout.addWidget(self.clear_button)
        footer_layout.addWidget(self.dashboard_button)
        footer_layout.addWidget(self.compare_button)
        footer_layout.addStretch()
        footer_layout.addWidget(self.save_button)

        main_layout: QVBoxLayout = QVBoxLayout()
        main_layout.setContentsMargins(24, 20, 24, 20)
        main_layout.setSpacing(18)
        main_layout.addLayout(header_layout)
        main_layout.addLayout(stage_layout, 1)
        main_layout.addSpacing(4)
        main_layout.addLayout(intensity_column)
        main_layout.addLayout(footer_layout)
        self.setLayout(main_layout)

    def _stage_column(self, caption: str, label: QLabel, button: QPushButton) -> QVBoxLayout:
        """Assemble one image-stage column: caption, card, action button."""
        column = QVBoxLayout()
        column.setSpacing(10)
        column.addWidget(_caption_label(caption))
        column.addWidget(label, 1, QtCore.Qt.AlignCenter)
        column.addWidget(button)
        return column

    def _load_image_file(
        self,
        path: str,
        image_type: str,
        preview_label: QLabel,
    ) -> None:
        """Load an image file, reset stale outputs, and update the UI.

        Args:
            path: Path to the image file.
            image_type: Human-readable role ('content' or 'style').
            preview_label: Label widget to display the loaded image.
        """
        try:
            img, meta = load_image_with_meta(path)
            self.update_timer.stop()
            self.styled_image = None
            self.result_image = None
            self.result_label.clear()
            self.result_label.setText("No image loaded")
            if image_type == "content":
                self.content_image = img
            else:
                self.style_image = img
            self.show_image(img, preview_label)
            self.show_conversion_info(image_type, meta)
        except (FileNotFoundError, ImageProcessingError) as e:
            QMessageBox.critical(self, "Error", f"Failed to load {image_type} image: {e!s}")
            logger.error("Failed to load %s image: %s", image_type, e)
        except Exception:  # noqa: BLE001 — user-facing catch; QMessageBox reports the error
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to load {image_type} image: an unexpected error occurred.",
            )
            logger.exception("Unexpected error loading %s image", image_type)

    def load_content_image(self) -> None:
        """Load content image from file dialog."""
        content_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Content Image",
            "",
            f"Image Files ({' '.join('*' + ext for ext in ALLOWED_IMAGE_EXTENSIONS)})",
        )
        if content_path:
            self._load_image_file(content_path, "content", self.content_label)

    def load_style_image(self) -> None:
        """Load style image from file dialog."""
        style_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Style Image",
            "",
            f"Image Files ({' '.join('*' + ext for ext in ALLOWED_IMAGE_EXTENSIONS)})",
        )
        if style_path:
            self._load_image_file(style_path, "style", self.style_label)

    def show_conversion_info(self, image_type: str, meta: ImageMeta) -> None:
        """Show information about any image conversions that occurred.

        Args:
            image_type: Type of image ('content' or 'style')
            meta: ImageMeta with original_ndim and original_channels

        Returns:
            None
        """
        if meta.original_ndim == 2:
            QMessageBox.information(
                self,
                "Image Conversion",
                f"Grayscale {image_type} image automatically " f"converted to RGB for processing.",
            )
        elif meta.original_ndim == 3 and meta.original_channels == 4:
            QMessageBox.information(
                self,
                "Image Conversion",
                f"{image_type.title()} image with transparency "
                f"(alpha channel) detected.\nAlpha channel removed for processing.",
            )

    def update_intensity(self, value: int) -> None:
        """Update intensity slider value and trigger debounced update.

        Args:
            value: Slider value (0-100)
        """
        self.intensity = value / 100.0
        self.intensity_label.setText(f"{value}%")
        if self.content_image is not None and self.styled_image is not None:
            self.update_timer.start(self._slider_debounce_ms)

    def apply_intensity_blend(self) -> None:
        """Apply intensity blending to result image."""
        if self.content_image is not None and self.styled_image is not None:
            self.result_image = blend_images(self.content_image, self.styled_image, self.intensity)
            self.show_image(self.result_image, self.result_label)

    def on_method_changed(self, index: int) -> None:
        """Handle transfer method selection change.

        Args:
            index: Index of selected method in combo box
        """
        data = self.method_combo.itemData(index)
        if not data:
            # Header row clicked — ignore
            return
        self.transfer_method = data
        method = registry.get_method(self.transfer_method)

        # The slider label comes from method metadata: it reads as
        # intensity, severity, or correction strength depending on the method.
        self.intensity_heading_label.setText(method.slider_label)

        # Methods with requires_reference=False do not use a style image,
        # so the load button is disabled and a note is shown.
        style_needed = method.requires_reference
        self.load_style_button.setEnabled(style_needed)
        if style_needed:
            self.load_style_button.setToolTip("")
            if self.style_image is None:
                self.style_label.setText("No image loaded")
        else:
            self.load_style_button.setToolTip(
                "Style image is not used in simulation or correction modes."
            )
            if self.style_image is None:
                self.style_label.setText("Not needed for this mode")

        # Reference-free methods only need the content image; others need both.
        if self.content_image is not None and (
            not method.requires_reference or self.style_image is not None
        ):
            self.apply_style_transfer()

    def apply_style_transfer(self) -> None:
        """Apply selected style transfer method to images."""
        method = registry.get_method(self.transfer_method)
        has_content = self.content_image is not None
        has_style = self.style_image is not None

        if has_content and (has_style or not method.requires_reference):
            try:
                # For simulators and Daltonizers the style image is unused,
                # so style_image may be None here.
                self.styled_image = registry.transfer_cached(
                    self.transfer_method, self.content_image, self.style_image
                )
                self.result_image = blend_images(
                    self.content_image, self.styled_image, self.intensity
                )
                self.show_image(self.result_image, self.result_label)
            except (
                Exception
            ) as e:  # noqa: BLE001 — user-facing catch; QMessageBox reports the error
                QMessageBox.critical(self, "Error", f"Failed to apply style transfer: {str(e)}")
                logger.error(f"Style transfer failed: {e}")
        else:
            if not method.requires_reference:
                msg = "Please load a content image before applying."
            elif not has_content and not has_style:
                msg = (
                    "Please load both a content and a style image "
                    "before applying style transfer."
                )
            elif not has_content:
                msg = "Please load a content image before applying style transfer."
            else:
                msg = "Please load a style image before applying style transfer."
            QMessageBox.warning(self, "Missing Images", msg)

    def clear_images(self) -> None:
        """Clear all loaded and processed images."""
        self.content_image = None
        self.style_image = None
        self.result_image = None
        self.styled_image = None

        for label in [self.content_label, self.style_label, self.result_label]:
            label.clear()
            label.setText("No image loaded")

    def save_result(self) -> None:
        """Save the result image to file."""
        if self.result_image is not None:
            options = QFileDialog.Options()
            file_types = (
                "PNG Files (*.png);;JPEG Files (*.jpg);;"
                "TIFF Files (*.tiff);;BMP Files (*.bmp);;All Files (*.*)"
            )
            save_path, _ = QFileDialog.getSaveFileName(
                self, "Save Result Image", "Untitled", file_types, options=options
            )
            if save_path:
                try:
                    save_image(self.result_image, save_path)
                    QMessageBox.information(
                        self,
                        "Save Image",
                        f"Image successfully saved to: {save_path}",
                    )
                    logger.info(f"Image saved to: {save_path}")
                except (
                    Exception
                ) as e:  # noqa: BLE001 — user-facing catch; QMessageBox reports the error
                    QMessageBox.critical(self, "Error", f"Failed to save image: {str(e)}")
                    logger.error(f"Failed to save image: {e}")
        else:
            QMessageBox.warning(
                self,
                "No Image",
                "No result image to save. Please apply style transfer first.",
            )

    def show_comparison(self) -> None:
        """Launch the method comparison dialog for the current content and style images."""
        if self.content_image is None:
            QMessageBox.warning(
                self,
                "No Content Image",
                "Please load a content image before comparing methods.",
            )
            return
        if self.style_image is None:
            QMessageBox.warning(
                self,
                "No Style Image",
                "Please load a style image before comparing methods.",
            )
            return

        dialog = CompareMethodsDialog(self.content_image, self.style_image, self)
        dialog.exec_()

    def show_dashboard(self) -> None:
        """Launch the CVD accessibility dashboard for the current content image."""
        if self.content_image is None:
            QMessageBox.warning(
                self,
                "No Content Image",
                "Please load a content image before opening the Dashboard.",
            )
            return

        dialog = DashboardDialog(self.content_image, self)
        dialog.exec_()

    def show_image(self, img_array: np.ndarray, label: QLabel) -> None:
        """Display image array in QLabel."""
        label.setPixmap(_array_to_pixmap(img_array, self._preview_size))


class DashboardDialog(QDialog):
    """Modal dialog displaying the CVD accessibility dashboard."""

    _CELL_SIZE: int = 220

    def __init__(self, content_image: np.ndarray, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._content_image = content_image
        self._result: DashboardResult | None = None
        self._label_widgets: dict[str, QLabel] = {}
        self._title_widgets: dict[str, QLabel] = {}
        self._summary_label: QLabel | None = None
        self._report_button: QPushButton | None = None
        self._appearance_combo: QComboBox | None = None
        self._error: str | None = None
        self._current_request_id: int = 0

        self.setWindowTitle("CVD Accessibility Dashboard")

        # Match the parent window's size if available, otherwise use a
        # sensible default that fits the 2×3 grid comfortably.
        if parent is not None:
            self.resize(parent.width(), parent.height() + 100)
        else:
            self.resize(1000, 800)
        self._build_ui()
        self._start_computation()

    def _build_ui(self) -> None:
        self.setStyleSheet(_APP_STYLESHEET)
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(24, 20, 24, 20)
        main_layout.setSpacing(14)

        self._status_label = QLabel("Computing simulations. Please wait...", self)
        self._status_label.setAlignment(QtCore.Qt.AlignCenter)
        self._status_label.setProperty("status", True)
        main_layout.addWidget(self._status_label)

        grid = QGridLayout()
        grid.setSpacing(12)
        main_layout.addLayout(grid)

        # Row 0: [Original] centered, alone
        # Row 1: [Protanopia]         [Deuteranopia]         [Tritanopia]
        # Row 2: [Error Metric (P)]  [Error Metric (D)]  [Error Metric (T)]
        positions: list[tuple[int, int, str, str]] = [
            (0, 1, "original", "Original"),
            (1, 0, "protanopia", _DEFICIENCY_LABELS["protanopia"]),
            (1, 1, "deuteranopia", _DEFICIENCY_LABELS["deuteranopia"]),
            (1, 2, "tritanopia", _DEFICIENCY_LABELS["tritanopia"]),
            (2, 0, "heatmap_protanopia", "Error Metric (P)"),
            (2, 1, "heatmap_deuteranopia", "Error Metric (D)"),
            (2, 2, "heatmap_tritanopia", "Error Metric (T)"),
        ]
        for row, col, key, label_text in positions:
            label = QLabel(self)
            label.setFixedSize(self._CELL_SIZE, self._CELL_SIZE)
            label.setAlignment(QtCore.Qt.AlignCenter)
            label.setProperty("imageCard", True)
            self._label_widgets[key] = label
            container = QVBoxLayout()
            container.setSpacing(6)
            container.addWidget(label, alignment=QtCore.Qt.AlignCenter)
            title = QLabel(label_text, self)
            self._title_widgets[key] = title
            title.setAlignment(QtCore.Qt.AlignCenter)
            title.setWordWrap(True)
            title.setProperty("cellTitle", True)
            container.addWidget(title)
            grid.addLayout(container, row, col)

        # Summary panel: the monospace font comes from the summaryCard
        # stylesheet rule, since QSS font properties override setFont().
        self._summary_label = QLabel("", self)
        self._summary_label.setProperty("summaryCard", True)
        main_layout.addWidget(self._summary_label)

        # Bottom buttons
        button_layout = QHBoxLayout()
        metric_caption = _caption_label("ΔE METRIC")
        button_layout.addWidget(metric_caption)
        self._appearance_combo = QComboBox(self)
        self._appearance_combo.addItem("CIELAB (legacy)", "cielab")
        self._appearance_combo.addItem("ICtCp (HDR, BT.2100)", "ictcp")
        self._appearance_combo.setCurrentIndex(0)
        self._appearance_combo.currentIndexChanged.connect(self._restart_computation)
        button_layout.addWidget(self._appearance_combo)
        button_layout.addStretch()
        self._report_button = QPushButton("Generate Report", self)
        self._report_button.clicked.connect(self._export_report)
        self._report_button.setEnabled(False)
        close_button = QPushButton("Close", self)
        close_button.setProperty("secondary", True)
        close_button.clicked.connect(self.accept)

        button_layout.addWidget(close_button)
        button_layout.addWidget(self._report_button)
        main_layout.addLayout(button_layout)

    def _start_computation(self) -> None:
        self._error = None
        self._current_request_id += 1
        # Increment request ID so stale completions from previous computations
        # are ignored if the user switches the appearance metric mid-computation.
        request_id = self._current_request_id

        if self._report_button is not None:
            self._report_button.setEnabled(False)

        # Read appearance space id from combo box item data.  Falls back
        # to "cielab" when no combo box is present (should not happen in
        # practice, but is safe).
        space_name: str = "cielab"
        if self._appearance_combo is not None:
            data = self._appearance_combo.currentData()
            if data is not None:
                space_name = str(data)
        appearance = make_appearance_space(space_name)

        result: DashboardResult | None = None
        error: str | None = None

        def _compute() -> None:
            nonlocal result, error
            try:
                result = compute_dashboard(self._content_image, appearance=appearance)
            except Exception as e:  # noqa: BLE001 — surfaced in the dialog
                logger.exception("Dashboard computation failed")
                error = str(e)
                result = None

        def _on_done() -> None:
            nonlocal result, error
            if request_id != self._current_request_id:
                return
            self._result = result
            self._error = error
            if self._error is not None:
                self._status_label.setText(f"Computation failed: {self._error}")
            else:
                self._status_label.setText("Done.")
                self._populate_results()

        _run_in_thread(_compute, on_done=_on_done)

    def _restart_computation(self) -> None:
        self._status_label.setText("Computing simulations. Please wait...")
        self._start_computation()

    def _populate_results(self) -> None:
        if self._result is None:
            return

        result = self._result

        def _show(array: np.ndarray, key: str) -> None:
            widget = self._label_widgets.get(key)
            if widget is None:
                return
            normalize = key.startswith("heatmap_")
            widget.setPixmap(_array_to_pixmap(array, self._CELL_SIZE, normalize_heatmap=normalize))

        _show(result.original, "original")
        for deficiency in _DEFICIENCIES:
            _show(result.simulated[deficiency], deficiency)

        # Error-metric heatmaps — renders preferred_metric() as a grayscale
        # hot image
        for key, deficiency in [
            ("heatmap_protanopia", "protanopia"),
            ("heatmap_deuteranopia", "deuteranopia"),
            ("heatmap_tritanopia", "tritanopia"),
        ]:
            em = result.error_maps.get(deficiency)
            if em is not None:
                _show(em.preferred_metric(), key)
                title_widget = self._title_widgets.get(key)
                if title_widget is not None:
                    title_widget.setText(_heatmap_title(deficiency, result.metric_label))

        self._update_summary(result)
        if self._report_button is not None:
            self._report_button.setEnabled(True)

    def _update_summary(self, result: DashboardResult) -> None:
        if self._summary_label is not None:
            self._summary_label.setText(format_summary_table(result))

    def _export_report(self) -> None:
        if self._result is None:
            return

        metric_name = self._result.metric_label
        filename = f"dashboard_report_{_sanitize_filename(metric_name)}.png"
        title = f"CVD Accessibility Dashboard – {metric_name}"

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Dashboard Report",
            filename,
            "PNG Files (*.png);;All Files (*.*)",
        )
        if not save_path:
            return

        try:
            generate_dashboard_report(self._result, save_path, title=title)
        except Exception as e:  # noqa: BLE001 — user-facing catch
            logger.exception("Failed to generate dashboard report")
            QMessageBox.critical(self, "Error", f"Failed to save report: {e}")
            return
        QMessageBox.information(self, "Report Saved", f"Dashboard report saved to:\n{save_path}")


class CompareMethodsDialog(QDialog):
    """Dialog that runs every style-transfer method and shows a 3×3 grid comparison."""

    _CELL_SIZE: int = 280

    def __init__(
        self,
        content_image: np.ndarray,
        style_image: np.ndarray,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._content_image = content_image
        self._style_image = style_image
        self._label_widgets: dict[str, QLabel] = {}
        self._results: dict[str, np.ndarray] = {}

        self.setWindowTitle("Compare Transfer Methods")
        if parent is not None:
            self.resize(parent.width(), parent.height())
        else:
            self.resize(1000, 800)

        self._build_ui()
        self._start_computation()

    def _build_ui(self) -> None:
        self.setStyleSheet(_APP_STYLESHEET)
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(24, 20, 24, 20)
        main_layout.setSpacing(14)

        self._status_label = QLabel("Computing transfers. Please wait...", self)
        self._status_label.setAlignment(QtCore.Qt.AlignCenter)
        self._status_label.setProperty("status", True)
        main_layout.addWidget(self._status_label)

        grid = QGridLayout()
        grid.setSpacing(12)
        main_layout.addLayout(grid)

        method_ids = [
            m for m in registry.list_methods() if registry.get_method(m).requires_reference
        ]
        for idx, method_id in enumerate(method_ids):
            row, col = divmod(idx, 3)
            label = QLabel(self)
            label.setFixedSize(self._CELL_SIZE, self._CELL_SIZE)
            label.setAlignment(QtCore.Qt.AlignCenter)
            label.setProperty("imageCard", True)
            self._label_widgets[method_id] = label
            container = QVBoxLayout()
            container.setSpacing(6)
            container.addWidget(label, alignment=QtCore.Qt.AlignCenter)
            title = QLabel(method_id, self)
            title.setAlignment(QtCore.Qt.AlignCenter)
            title.setWordWrap(True)
            title.setProperty("cellTitle", True)
            container.addWidget(title)
            grid.addLayout(container, row, col)

        button_layout = QHBoxLayout()
        close_button = QPushButton("Close", self)
        close_button.setProperty("secondary", True)
        close_button.clicked.connect(self.accept)
        button_layout.addStretch()
        button_layout.addWidget(close_button)
        main_layout.addLayout(button_layout)

    def _start_computation(self) -> None:
        def _run() -> None:
            for method_id in self._label_widgets:
                try:
                    instance = registry.get_method(method_id)
                    self._results[method_id] = instance.transfer(
                        self._content_image, self._style_image
                    )
                except Exception:  # noqa: BLE001 — skip unsupported methods
                    logger.debug("Skipped method %s in comparison", method_id)

        _run_in_thread(_run, on_done=self._populate_results)

    def _populate_results(self) -> None:
        for method_id in self._label_widgets:
            self._show_one(method_id)
        self._on_computation_done()

    def _show_one(self, method_id: str) -> None:
        widget = self._label_widgets.get(method_id)
        result = self._results.get(method_id)
        if widget is None or result is None:
            return
        widget.setPixmap(_array_to_pixmap(result, self._CELL_SIZE))
        widget.repaint()

    def _on_computation_done(self) -> None:
        done = sum(1 for v in self._results.values() if v is not None)
        total = len(self._label_widgets)
        self._status_label.setText(f"Done: {done} of {total} methods completed.")


def main() -> None:
    """Run the ColorCast graphical application."""
    logging.basicConfig(level=logging.INFO)
    app = QApplication([])
    # Fusion keeps chrome (arrows, focus rings, native dialogs) consistent
    # with the stylesheet across platforms.
    app.setStyle("Fusion")
    app.setPalette(_dark_palette())
    app.setStyleSheet(_APP_STYLESHEET)
    _apply_dark_window_chrome()
    window = StyleTransferApp()
    window.show()
    app.exec()
