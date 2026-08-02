"""PyQt5 graphical interface for ColorCast."""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap, QStandardItem, QStandardItemModel
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
    QSlider,
    QVBoxLayout,
    QWidget,
)

from colorcast import blend_images, registry, save_image
from colorcast.analysis.dashboard import (
    _DEFICIENCIES,
    _DEFICIENCY_LABELS,
    DashboardResult,
    format_summary_table,
)
from colorcast.processing.image_loader import ImageMeta, load_image_with_meta
from colorcast.utils.config import ColorCastConfig
from colorcast.utils.exceptions import ImageProcessingError
from colorcast.utils.validators_enhanced import ALLOWED_IMAGE_EXTENSIONS

logger = logging.getLogger(__name__)


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


def _run_in_thread(
    target: Callable[[], None], *, on_done: Callable[[], None] | None = None
) -> None:
    """Run ``target`` off the GUI thread, calling ``on_done`` via a queued signal."""
    from threading import Thread

    signals = _WorkerSignals()
    _ref: list[_WorkerSignals] = [signals]  # keep alive until signal is delivered

    if on_done is not None:

        def _on_done() -> None:
            on_done()
            _ref.clear()

        signals.finished.connect(_on_done)
    else:
        signals.finished.connect(_ref.clear)

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

        self.content_image: np.ndarray | None = None
        self.style_image: np.ndarray | None = None
        self.result_image: np.ndarray | None = None
        self.styled_image: np.ndarray | None = None
        self.intensity: float = self._intensity
        self.transfer_method: str = cfg.default_method
        self.update_timer: QTimer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self.apply_intensity_blend)
        self.setFixedSize(self._window_width, self._window_height)
        self.initUI()

    def initUI(self) -> None:
        """Initialize the user interface."""
        self.setWindowTitle("ColorCast - Style Transfer")

        self.content_label: QLabel = QLabel(self)
        self.style_label: QLabel = QLabel(self)
        self.result_label: QLabel = QLabel(self)

        for label in [self.content_label, self.style_label, self.result_label]:
            label.setMinimumSize(self._preview_size, self._preview_size)
            label.setStyleSheet("border: 1px solid gray;")
            label.setAlignment(QtCore.Qt.AlignCenter)
            label.setText("No image loaded")

        self.intensity_slider: QSlider = QSlider(QtCore.Qt.Horizontal, self)
        self.intensity_slider.setMinimum(0)
        self.intensity_slider.setMaximum(100)
        self.intensity_slider.setValue(int(self._intensity * 100))
        self.intensity_slider.setTickPosition(QSlider.TicksBelow)
        self.intensity_slider.setTickInterval(10)
        self.intensity_slider.valueChanged.connect(self.update_intensity)
        self.intensity_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 9px;
                background: #E0E0E0;
                margin: 0;
            }
            QSlider::handle:horizontal {
                background: #505050;
                width: 18px;
                margin: 0;
            }
            QSlider::handle:horizontal:hover {
                background: #606060;
            }
        """)

        self.intensity_label: QLabel = QLabel(f"{int(self._intensity * 100)}%", self)
        self.intensity_label.setAlignment(QtCore.Qt.AlignCenter)

        self.method_combo: QComboBox = QComboBox(self)

        # Build a categorised model with non-selectable section headers
        _combo_model = QStandardItemModel(self.method_combo)

        def _header(text: str) -> QStandardItem:
            """Create a bold, non-selectable section-header row."""
            item = QStandardItem(text)
            item.setEnabled(False)
            item.setSelectable(False)
            font = item.font()
            font.setBold(True)
            item.setFont(font)
            item.setForeground(QtGui.QColor("#888888"))
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
        self.method_combo.currentIndexChanged.connect(self.on_method_changed)

        self.load_content_button: QPushButton = QPushButton("Load Content Image", self)
        self.load_style_button: QPushButton = QPushButton("Load Style Image", self)
        self.apply_button: QPushButton = QPushButton("Apply Style Transfer", self)
        self.clear_button: QPushButton = QPushButton("Clear Images", self)
        self.save_button: QPushButton = QPushButton("Save Result", self)
        self.dashboard_button: QPushButton = QPushButton("Dashboard", self)
        self.compare_button: QPushButton = QPushButton("Compare Methods", self)

        self.load_content_button.clicked.connect(self.load_content_image)
        self.load_style_button.clicked.connect(self.load_style_image)
        self.apply_button.clicked.connect(self.apply_style_transfer)
        self.clear_button.clicked.connect(self.clear_images)
        self.save_button.clicked.connect(self.save_result)
        self.dashboard_button.clicked.connect(self.show_dashboard)
        self.compare_button.clicked.connect(self.show_comparison)

        main_layout: QVBoxLayout = QVBoxLayout()
        method_layout: QHBoxLayout = QHBoxLayout()
        grid_layout: QGridLayout = QGridLayout()
        intensity_container: QVBoxLayout = QVBoxLayout()
        intensity_slider_layout: QHBoxLayout = QHBoxLayout()
        intensity_labels_layout: QHBoxLayout = QHBoxLayout()
        button_layout: QHBoxLayout = QHBoxLayout()

        method_layout.addWidget(QLabel("Transfer Method:"))
        method_layout.addWidget(self.method_combo)
        method_layout.addStretch()

        grid_layout.setSpacing(20)
        grid_layout.setContentsMargins(10, 10, 10, 10)

        grid_layout.addWidget(self.content_label, 0, 0)
        grid_layout.addWidget(self.style_label, 0, 1)
        grid_layout.addWidget(self.result_label, 0, 2)

        grid_layout.addWidget(self.load_content_button, 1, 0)
        grid_layout.addWidget(self.load_style_button, 1, 1)
        grid_layout.addWidget(self.apply_button, 1, 2)

        intensity_slider_layout.addWidget(self.intensity_slider)

        self.intensity_heading_label: QLabel = QLabel("Style Intensity:")
        intensity_labels_layout.addWidget(self.intensity_heading_label)
        intensity_labels_layout.addStretch()
        intensity_labels_layout.addWidget(self.intensity_label)

        intensity_container.setSpacing(0)
        intensity_container.addLayout(intensity_slider_layout)
        intensity_container.addLayout(intensity_labels_layout)
        intensity_container.setContentsMargins(0, 0, 0, 10)

        button_layout.setSpacing(20)
        button_layout.addWidget(self.clear_button)
        button_layout.addWidget(self.dashboard_button)
        button_layout.addWidget(self.compare_button)
        button_layout.addStretch()
        button_layout.addWidget(self.save_button)

        main_layout.addLayout(method_layout)
        main_layout.addLayout(grid_layout)
        main_layout.addStretch()
        main_layout.addLayout(intensity_container)
        main_layout.addLayout(button_layout)
        self.setLayout(main_layout)

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
                "TIFF Files(*.tiff);;BMP Files (*.bmp);;All Files (*.*)"
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
        self._summary_label: QLabel | None = None
        self._report_button: QPushButton | None = None

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
        main_layout = QVBoxLayout(self)

        self._status_label = QLabel("Computing simulations — please wait…", self)
        self._status_label.setAlignment(QtCore.Qt.AlignCenter)
        main_layout.addWidget(self._status_label)

        grid = QGridLayout()
        grid.setSpacing(12)
        main_layout.addLayout(grid)

        # Row 0: [Original] — centred, alone
        # Row 1: [Protanopia]         [Deuteranopia]         [Tritanopia]
        # Row 2: [Chroma Loss (P)]    [Chroma Loss (D)]      [Chroma Loss (T)]
        positions: list[tuple[int, int, str, str]] = [
            (0, 1, "original", "Original"),
            (1, 0, "protanopia", _DEFICIENCY_LABELS["protanopia"]),
            (1, 1, "deuteranopia", _DEFICIENCY_LABELS["deuteranopia"]),
            (1, 2, "tritanopia", _DEFICIENCY_LABELS["tritanopia"]),
            (2, 0, "heatmap_protanopia", "Chroma Loss (P)"),
            (2, 1, "heatmap_deuteranopia", "Chroma Loss (D)"),
            (2, 2, "heatmap_tritanopia", "Chroma Loss (T)"),
        ]
        for row, col, key, label_text in positions:
            label = QLabel(self)
            label.setFixedSize(self._CELL_SIZE, self._CELL_SIZE)
            label.setAlignment(QtCore.Qt.AlignCenter)
            label.setStyleSheet("border: 1px solid gray;")
            self._label_widgets[key] = label
            container = QVBoxLayout()
            container.addWidget(label, alignment=QtCore.Qt.AlignCenter)
            title = QLabel(label_text, self)
            title.setAlignment(QtCore.Qt.AlignCenter)
            title.setWordWrap(True)
            container.addWidget(title)
            grid.addLayout(container, row, col)

        # Summary panel
        self._summary_label = QLabel("", self)
        font = QtGui.QFont("Courier New", 11)
        font.setStyleHint(QtGui.QFont.Monospace)
        self._summary_label.setFont(font)
        self._summary_label.setStyleSheet("margin: 8px 4px; padding: 6px;")
        main_layout.addWidget(self._summary_label)

        # Bottom buttons
        button_layout = QHBoxLayout()
        self._report_button = QPushButton("Generate Report", self)
        self._report_button.clicked.connect(self._export_report)
        self._report_button.setEnabled(False)
        close_button = QPushButton("Close", self)
        close_button.clicked.connect(self.accept)

        button_layout.addStretch()
        button_layout.addWidget(close_button)
        button_layout.addWidget(self._report_button)
        main_layout.addLayout(button_layout)

    def _start_computation(self) -> None:
        self._error: str | None = None

        def _compute() -> None:
            from colorcast.analysis.dashboard import compute_dashboard

            try:
                self._result = compute_dashboard(self._content_image)
            except Exception as e:  # noqa: BLE001 — surfaced in the dialog
                logger.exception("Dashboard computation failed")
                self._error = str(e)
                self._result = None

        def _on_done() -> None:
            if self._error is not None:
                self._status_label.setText(f"Computation failed: {self._error}")
            else:
                self._status_label.setText("Done.")
                self._populate_results()

        _run_in_thread(_compute, on_done=_on_done)

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

        # Chroma-loss heatmaps — use the chroma_error field rendered as
        # a grayscale hot image
        for key, deficiency in [
            ("heatmap_protanopia", "protanopia"),
            ("heatmap_deuteranopia", "deuteranopia"),
            ("heatmap_tritanopia", "tritanopia"),
        ]:
            em = result.error_maps.get(deficiency)
            if em is not None:
                _show(em.chroma_error, key)

        self._update_summary(result)
        if self._report_button is not None:
            self._report_button.setEnabled(True)

    def _update_summary(self, result: DashboardResult) -> None:
        if self._summary_label is not None:
            self._summary_label.setText(format_summary_table(result))

    def _export_report(self) -> None:
        if self._result is None:
            return
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Dashboard Report",
            "dashboard_report.png",
            "PNG Files (*.png);;All Files (*.*)",
        )
        if not save_path:
            return
        from colorcast.analysis.dashboard import generate_dashboard_report

        try:
            generate_dashboard_report(self._result, save_path)
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
        main_layout = QVBoxLayout(self)

        self._status_label = QLabel("Computing transfers — please wait…", self)
        self._status_label.setAlignment(QtCore.Qt.AlignCenter)
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
            label.setStyleSheet("border: 1px solid gray;")
            self._label_widgets[method_id] = label
            container = QVBoxLayout()
            container.addWidget(label, alignment=QtCore.Qt.AlignCenter)
            title = QLabel(method_id, self)
            title.setAlignment(QtCore.Qt.AlignCenter)
            title.setWordWrap(True)
            container.addWidget(title)
            grid.addLayout(container, row, col)

        button_layout = QHBoxLayout()
        close_button = QPushButton("Close", self)
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
        self._status_label.setText(f"Done — {done} of {total} methods completed.")


def main() -> None:
    """Run the ColorCast graphical application."""
    logging.basicConfig(level=logging.INFO)
    app = QApplication([])
    window = StyleTransferApp()
    window.show()
    app.exec()
