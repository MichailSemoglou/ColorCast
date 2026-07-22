"""PyQt5 graphical interface for ColorCast."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap, QStandardItem, QStandardItemModel
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
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
from colorcast import (
    blend_images,
    color_transfer_lab,
    color_transfer_meanstd,
    lut_transfer_with_curve,
    match_histograms_multichannel,
    save_image,
    selective_color_transfer,
)
from colorcast.analysis.daltonization import daltonize as daltonize_image
from colorcast.processing.image_loader import ImageMeta, load_image_with_meta
from colorcast.utils.exceptions import ImageProcessingError
from colorcast.processing.simulation import ColorBlindSimulator

logger = logging.getLogger(__name__)

# Module-level constants
DEFAULT_WINDOW_WIDTH: int = 1000
DEFAULT_WINDOW_HEIGHT: int = 700
PREVIEW_SIZE: int = 300
DEFAULT_INTENSITY: float = 0.85
SLIDER_DEBOUNCE_MS: int = 50
ALLOWED_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


class StyleTransferApp(QWidget):
    """Main application window for ColorCast style transfer."""

    def __init__(self) -> None:
        """Initialize the ColorCast application."""
        super().__init__()
        self.content_image: Optional[np.ndarray] = None
        self.style_image: Optional[np.ndarray] = None
        self.result_image: Optional[np.ndarray] = None
        self.styled_image: Optional[np.ndarray] = None
        self.intensity: float = DEFAULT_INTENSITY
        self.transfer_method: str = "histogram"
        self.update_timer: QTimer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self.apply_intensity_blend)
        self.setFixedSize(DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT)
        self.initUI()

    def initUI(self) -> None:
        """Initialize the user interface."""
        self.setWindowTitle("ColorCast - Style Transfer")

        self.content_label: QLabel = QLabel(self)
        self.style_label: QLabel = QLabel(self)
        self.result_label: QLabel = QLabel(self)

        for label in [self.content_label, self.style_label, self.result_label]:
            label.setMinimumSize(PREVIEW_SIZE, PREVIEW_SIZE)
            label.setStyleSheet("border: 1px solid gray;")
            label.setAlignment(QtCore.Qt.AlignCenter)
            label.setText("No image loaded")

        self.intensity_slider: QSlider = QSlider(QtCore.Qt.Horizontal, self)
        self.intensity_slider.setMinimum(0)
        self.intensity_slider.setMaximum(100)
        self.intensity_slider.setValue(int(DEFAULT_INTENSITY * 100))
        self.intensity_slider.setTickPosition(QSlider.TicksBelow)
        self.intensity_slider.setTickInterval(10)
        self.intensity_slider.valueChanged.connect(self.update_intensity)
        self.intensity_slider.setStyleSheet(
            """
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
        """
        )

        self.intensity_label: QLabel = QLabel(f"{int(DEFAULT_INTENSITY * 100)}%", self)
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

        self.load_content_button: QPushButton = QPushButton(
            "Load Content Image", self
        )
        self.load_style_button: QPushButton = QPushButton("Load Style Image", self)
        self.apply_button: QPushButton = QPushButton("Apply Style Transfer", self)
        self.clear_button: QPushButton = QPushButton("Clear Images", self)
        self.save_button: QPushButton = QPushButton("Save Result", self)

        self.load_content_button.clicked.connect(self.load_content_image)
        self.load_style_button.clicked.connect(self.load_style_image)
        self.apply_button.clicked.connect(self.apply_style_transfer)
        self.clear_button.clicked.connect(self.clear_images)
        self.save_button.clicked.connect(self.save_result)

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
        image_attr: str,
        preview_label: QLabel,
        image_type: str,
    ) -> None:
        """Load an image file, reset stale outputs, and update the UI.

        Args:
            path: Path to the image file.
            image_attr: Name of the instance attribute to assign the array.
            preview_label: Label widget to display the loaded image.
            image_type: Human-readable role ('content' or 'style').
        """
        try:
            img, meta = load_image_with_meta(path)
            self.update_timer.stop()
            self.styled_image = None
            self.result_image = None
            self.result_label.clear()
            self.result_label.setText("No image loaded")
            setattr(self, image_attr, img)
            self.show_image(img, preview_label)
            self.show_conversion_info(image_type, meta)
        except (FileNotFoundError, ImageProcessingError) as e:
            QMessageBox.critical(
                self, "Error", f"Failed to load {image_type} image: {e!s}"
            )
            logger.error("Failed to load %s image: %s", image_type, e)
        except Exception:  # noqa: BLE001
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
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)",
        )
        if content_path:
            self._load_image_file(
                content_path, "content_image", self.content_label, "content"
            )

    def load_style_image(self) -> None:
        """Load style image from file dialog."""
        style_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Style Image",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)",
        )
        if style_path:
            self._load_image_file(
                style_path, "style_image", self.style_label, "style"
            )

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
                f"Grayscale {image_type} image automatically "
                f"converted to RGB for processing.",
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
            self.update_timer.start(SLIDER_DEBOUNCE_MS)

    def apply_intensity_blend(self) -> None:
        """Apply intensity blending to result image."""
        if (
            self.content_image is not None
            and self.styled_image is not None
        ):
            self.result_image = blend_images(
                self.content_image, self.styled_image, self.intensity
            )
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
        # Rename the slider label depending on mode
        _simulators = {
            "simulate_deuteranopia",
            "simulate_protanopia",
            "simulate_tritanopia",
        }
        _daltonizers = {
            "daltonize_deuteranopia",
            "daltonize_protanopia",
            "daltonize_tritanopia",
        }
        if self.transfer_method in _simulators:
            self.intensity_heading_label.setText("Severity (0%=normal, 100%=full):")
        elif self.transfer_method in _daltonizers:
            self.intensity_heading_label.setText(
                "Correction Intensity (0%=original, 100%=fully corrected):"
            )
        else:
            self.intensity_heading_label.setText("Style Intensity:")

        # Update style-image controls: simulators and daltonizers do not use a
        # style image, so the load button is disabled and a note is shown.
        style_needed = self.transfer_method not in _simulators and self.transfer_method not in _daltonizers
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

        # Simulators and Daltonizers only need the content image; others need both
        if self.transfer_method in _simulators or self.transfer_method in _daltonizers:
            if self.content_image is not None:
                self.apply_style_transfer()
        elif self.content_image is not None and self.style_image is not None:
            self.apply_style_transfer()

    def apply_style_transfer(self) -> None:
        """Apply selected style transfer method to images."""
        _simulators = {
            "simulate_deuteranopia",
            "simulate_protanopia",
            "simulate_tritanopia",
        }
        _daltonizers = {
            "daltonize_deuteranopia": "deuteranopia",
            "daltonize_protanopia": "protanopia",
            "daltonize_tritanopia": "tritanopia",
        }
        is_simulator = self.transfer_method in _simulators
        is_daltonizer = self.transfer_method in _daltonizers
        has_content = self.content_image is not None
        has_style = self.style_image is not None

        if has_content and (has_style or is_simulator or is_daltonizer):
            try:
                if is_simulator:
                    self.styled_image = ColorBlindSimulator().transform_color_space(
                        self.content_image,
                        self.transfer_method.replace("simulate_", ""),
                    )
                elif is_daltonizer:
                    # Phase 3: full pipeline (simulate -> error map -> correct).
                    # Intensity=1.0 here; the slider blends original <-> fully
                    # corrected via apply_intensity_blend / blend_images.
                    self.styled_image = daltonize_image(
                        self.content_image,
                        _daltonizers[self.transfer_method],
                        intensity=1.0,
                    )
                elif self.transfer_method == "histogram":
                    self.styled_image = match_histograms_multichannel(
                        self.content_image, self.style_image
                    )
                elif self.transfer_method == "meanstd":
                    self.styled_image = color_transfer_meanstd(
                        self.content_image, self.style_image
                    )
                elif self.transfer_method == "lab_reinhard":
                    self.styled_image = color_transfer_lab(
                        self.content_image, self.style_image
                    )
                elif self.transfer_method == "lut_linear":
                    self.styled_image = lut_transfer_with_curve(
                        self.content_image, self.style_image, "linear"
                    )
                elif self.transfer_method == "lut_scurve":
                    self.styled_image = lut_transfer_with_curve(
                        self.content_image, self.style_image, "s-curve"
                    )
                elif self.transfer_method == "lut_contrast":
                    self.styled_image = lut_transfer_with_curve(
                        self.content_image, self.style_image, "contrast"
                    )
                elif self.transfer_method == "selective_shadows":
                    self.styled_image = selective_color_transfer(
                        self.content_image, self.style_image, "shadows"
                    )
                elif self.transfer_method == "selective_midtones":
                    self.styled_image = selective_color_transfer(
                        self.content_image, self.style_image, "midtones"
                    )
                elif self.transfer_method == "selective_highlights":
                    self.styled_image = selective_color_transfer(
                        self.content_image, self.style_image, "highlights"
                    )
                else:
                    self.styled_image = match_histograms_multichannel(
                        self.content_image, self.style_image
                    )

                self.result_image = blend_images(
                    self.content_image, self.styled_image, self.intensity
                )
                self.show_image(self.result_image, self.result_label)
            except Exception as e:
                QMessageBox.critical(
                    self, "Error", f"Failed to apply style transfer: {str(e)}"
                )
                logger.error(f"Style transfer failed: {e}")
        else:
            if is_simulator or is_daltonizer:
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
                except Exception as e:
                    QMessageBox.critical(
                        self, "Error", f"Failed to save image: {str(e)}"
                    )
                    logger.error(f"Failed to save image: {e}")
        else:
            QMessageBox.warning(
                self,
                "No Image",
                "No result image to save. Please apply style transfer first.",
            )

    def show_image(self, img_array: np.ndarray, label: QLabel) -> None:
        """Display image array in QLabel.

        Args:
            img_array: Image array to display (float [0,1] or uint8 [0,255])
            label: QLabel widget to display image in

        Raises:
            ValueError: If image format is unsupported for display
        """
        img_array = np.clip(img_array, 0, 1)
        img_array = (img_array * 255).astype(np.uint8)

        if img_array.ndim == 2:
            h, w = img_array.shape
            bytes_per_line = w
            qt_image = QImage(
                img_array.data, w, h, bytes_per_line, QImage.Format_Grayscale8
            )
        elif img_array.ndim == 3 and img_array.shape[2] == 3:
            h, w, ch = img_array.shape
            bytes_per_line = ch * w
            qt_image = QImage(
                img_array.data, w, h, bytes_per_line, QImage.Format_RGB888
            )
        else:
            raise ValueError(f"Unsupported image format for display: {img_array.shape}")

        label.setPixmap(
            QPixmap.fromImage(qt_image).scaled(
                PREVIEW_SIZE, PREVIEW_SIZE, QtCore.Qt.KeepAspectRatio
            )
        )


def main() -> None:
    """Run the ColorCast graphical application."""
    logging.basicConfig(level=logging.INFO)
    app = QApplication([])
    window = StyleTransferApp()
    window.show()
    app.exec()
