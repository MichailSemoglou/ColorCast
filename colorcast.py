"""
ColorCast - Advanced Color Transfer Suite

A sophisticated PyQt5 GUI application for advanced color and style transfer between images.

Features:
- 9 sophisticated transfer algorithms (histogram, statistical, LUT curves, selective regional)
- Real-time intensity control with smooth slider (0-100%)
- Selective regional color transfer (shadows/midtones/highlights)
- LUT-based transfer with multiple curve options (linear, s-curve, contrast)
- Supports RGB, grayscale, and RGBA images (automatically converts to RGB for processing)
- Optimized performance with smart caching and debounced slider updates
- Interactive GUI with image preview and save functionality

Constants:
    DEFAULT_WINDOW_WIDTH (int): Default GUI window width in pixels (1000)
    DEFAULT_WINDOW_HEIGHT (int): Default GUI window height in pixels (700)
    PREVIEW_SIZE (int): Size of image preview labels in pixels (300)
    DEFAULT_INTENSITY (float): Default style intensity (0.0-1.0, default 0.85)
    SLIDER_DEBOUNCE_MS (int): Slider update debounce time in milliseconds (50)
    MAX_IMAGE_PIXELS (int): Maximum allowed image pixels (50,000,000)

Dependencies: numpy, scikit-image, scipy, PyQt5
"""

import logging
from pathlib import Path
from typing import Literal, Optional

import numpy as np
from skimage import io, img_as_float, exposure, transform
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import (
    QFileDialog,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QWidget,
    QApplication,
    QSlider,
    QComboBox,
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Module-level constants
DEFAULT_WINDOW_WIDTH: int = 1000
DEFAULT_WINDOW_HEIGHT: int = 700
PREVIEW_SIZE: int = 300
DEFAULT_INTENSITY: float = 0.85
SLIDER_DEBOUNCE_MS: int = 50
MAX_IMAGE_PIXELS: int = 50_000_000
ALLOWED_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _validate_image_path(path: str) -> None:
    """
    Validate image path exists and has valid extension.

    Args:
        path: Path to image file

    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If file extension is not allowed
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    if path_obj.suffix.lower() not in ALLOWED_IMAGE_EXTENSIONS:
        raise ValueError(
            f"Invalid file extension. "
            f"Allowed: {', '.join(ALLOWED_IMAGE_EXTENSIONS)}"
        )


def _validate_image_size(img_array: np.ndarray, max_pixels: int = MAX_IMAGE_PIXELS) -> None:
    """
    Validate image size is within reasonable limits.

    Args:
        img_array: Image array to validate
        max_pixels: Maximum allowed pixels (default: 50MP)

    Raises:
        ValueError: If image is too large
    """
    if img_array.ndim < 2:
        raise ValueError("Image must have at least 2 dimensions")
    total_pixels = img_array.shape[0] * img_array.shape[1]
    if total_pixels > max_pixels:
        raise ValueError(
            f"Image too large: {total_pixels:,} pixels "
            f"(max: {max_pixels:,})"
        )


def load_image(path: str) -> np.ndarray:
    """
    Load image from file path.

    Args:
        path: Path to image file

    Returns:
        RGB image array in float format [0, 1]

    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If file format is invalid or too large
    """
    _validate_image_path(path)
    try:
        img = img_as_float(io.imread(path))
        _validate_image_size(img)
        return ensure_rgb(img)
    except (IOError, ValueError) as e:
        raise ValueError(f"Could not load image from {path}: {str(e)}")
    except Exception as e:
        raise ValueError(f"Unexpected error loading image: {str(e)}")


def ensure_rgb(img: np.ndarray) -> np.ndarray:
    """
    Convert image to RGB format, handling grayscale and RGBA images.

    Args:
        img: Input image array (2D, 3D with 1, 3, or 4 channels)

    Returns:
        RGB image array (H, W, 3)

    Raises:
        ValueError: If image has unsupported dimensions or channels
    """
    if img.ndim == 2:
        return np.stack([img, img, img], axis=2)
    elif img.ndim == 3:
        if img.shape[2] == 1:
            return np.concatenate([img, img, img], axis=2)
        elif img.shape[2] == 3:
            return img
        elif img.shape[2] == 4:
            return img[:, :, :3]
        else:
            raise ValueError(f"Unsupported number of channels: {img.shape[2]}")
    else:
        raise ValueError(f"Unsupported image dimensions: {img.ndim}")


def _ensure_compatible_shapes(
    source: np.ndarray, reference: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Ensure source and reference images have same dimensions.

    Args:
        source: Source image array (H, W, C)
        reference: Reference image array (H, W, C)

    Returns:
        Tuple of (source, reference) with matching shapes
    """
    if source.shape != reference.shape:
        logger.debug(
            f"Resizing style image from {reference.shape} to {source.shape}"
        )
        reference = transform.resize(
            reference, source.shape, anti_aliasing=True, preserve_range=True
        )
    return source, reference


def match_histograms_multichannel(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Match histograms between images per channel.

    Args:
        source: Source image array (H, W, 3)
        reference: Reference image array (H, W, 3)

    Returns:
        Matched image array (H, W, 3)

    Raises:
        ValueError: If images have incorrect dimensions
    """
    source, reference = _ensure_compatible_shapes(source, reference)

    if source.ndim != 3 or source.shape[2] != 3:
        raise ValueError("Source image is not 3-channel RGB after preprocessing.")
    if reference.ndim != 3 or reference.shape[2] != 3:
        raise ValueError("Reference image is not 3-channel RGB after preprocessing.")

    matched = np.empty_like(source)
    for i in range(3):
        matched[:, :, i] = exposure.match_histograms(
            source[:, :, i], reference[:, :, i]
        )
    return matched


def color_transfer_meanstd(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Transfer color using mean and standard deviation matching per channel.

    Args:
        source: Source image array (H, W, 3)
        reference: Reference image array (H, W, 3)

    Returns:
        Color-transferred image array (H, W, 3) in range [0, 1]

    Raises:
        ValueError: If images have incorrect dimensions
    """
    source, reference = _ensure_compatible_shapes(source, reference)

    if source.ndim != 3 or source.shape[2] != 3:
        raise ValueError("Source image is not 3-channel RGB after preprocessing.")
    if reference.ndim != 3 or reference.shape[2] != 3:
        raise ValueError("Reference image is not 3-channel RGB after preprocessing.")

    result = np.empty_like(source)
    for i in range(3):
        source_mean = np.mean(source[:, :, i])
        source_std = np.std(source[:, :, i])
        ref_mean = np.mean(reference[:, :, i])
        ref_std = np.std(reference[:, :, i])

        result[:, :, i] = (
            (source[:, :, i] - source_mean)
            * (ref_std / (source_std + 1e-8))
        ) + ref_mean

    return np.clip(result, 0, 1)


def apply_curve(
    values: np.ndarray,
    curve_type: Literal["linear", "s-curve", "contrast"] = "linear",
) -> np.ndarray:
    """
    Apply tone curve to values.

    Args:
        values: Input values (typically in range [0, 1])
        curve_type: Type of curve to apply ('linear', 's-curve', 'contrast')

    Returns:
        Values with curve applied

    Raises:
        ValueError: If curve_type is invalid
    """
    if curve_type == "linear":
        return values
    elif curve_type == "s-curve":
        return 0.5 + 0.5 * np.sin(np.pi * (values - 0.5))
    elif curve_type == "contrast":
        return np.power(values, 0.8)
    else:
        raise ValueError(f"Unknown curve type: {curve_type}")


def lut_transfer_with_curve(
    source: np.ndarray,
    reference: np.ndarray,
    curve_type: Literal["linear", "s-curve", "contrast"] = "linear",
) -> np.ndarray:
    """
    LUT-based transfer with tone curve adjustment.

    Args:
        source: Source image array (H, W, 3)
        reference: Reference image array (H, W, 3)
        curve_type: Type of tone curve to apply ('linear', 's-curve', 'contrast')

    Returns:
        Transferred image array (H, W, 3) in range [0, 1]

    Raises:
        ValueError: If images have incorrect dimensions or curve_type is invalid
    """
    source, reference = _ensure_compatible_shapes(source, reference)

    if source.ndim != 3 or source.shape[2] != 3:
        raise ValueError("Source image is not 3-channel RGB after preprocessing.")
    if reference.ndim != 3 or reference.shape[2] != 3:
        raise ValueError("Reference image is not 3-channel RGB after preprocessing.")

    matched = np.empty_like(source)
    for i in range(3):
        matched[:, :, i] = exposure.match_histograms(
            source[:, :, i], reference[:, :, i]
        )
        matched[:, :, i] = apply_curve(matched[:, :, i], curve_type)

    return np.clip(matched, 0, 1)


def selective_color_transfer(
    source: np.ndarray,
    reference: np.ndarray,
    mode: Literal["full", "shadows", "midtones", "highlights"] = "full",
    shadow_threshold: float = 0.3,
    highlight_threshold: float = 0.7,
) -> np.ndarray:
    """
    Transfer colors selectively based on luminance regions.

    Args:
        source: Source image array (H, W, 3) in range [0, 1]
        reference: Reference image array (H, W, 3) in range [0, 1]
        mode: Transfer mode ('full', 'shadows', 'midtones', 'highlights')
        shadow_threshold: Luminance threshold for shadow region (0-1)
        highlight_threshold: Luminance threshold for highlight region (0-1)

    Returns:
        Color-transferred image array (H, W, 3) in range [0, 1]

    Raises:
        ValueError: If mode is invalid or images have incorrect dimensions
    """
    source, reference = _ensure_compatible_shapes(source, reference)

    if source.ndim != 3 or source.shape[2] != 3:
        raise ValueError("Source image is not 3-channel RGB after preprocessing.")
    if reference.ndim != 3 or reference.shape[2] != 3:
        raise ValueError("Reference image is not 3-channel RGB after preprocessing.")

    # Calculate luminance using ITU-R BT.601 coefficients
    source_lum = (
        0.299 * source[:, :, 0]
        + 0.587 * source[:, :, 1]
        + 0.114 * source[:, :, 2]
    )

    # Create mask based on selected tonal region
    if mode == "full":
        mask = np.ones_like(source_lum)
    elif mode == "shadows":
        mask = (source_lum < shadow_threshold).astype(float)
    elif mode == "midtones":
        mask = (
            (source_lum >= shadow_threshold)
            & (source_lum <= highlight_threshold)
        ).astype(float)
    elif mode == "highlights":
        mask = (source_lum > highlight_threshold).astype(float)
    else:
        raise ValueError(f"Invalid mode: {mode}")

    # Expand mask to 3 channels for RGB processing
    mask = np.stack([mask, mask, mask], axis=2)

    # Apply histogram matching to all channels
    matched = np.empty_like(source)
    for i in range(3):
        matched[:, :, i] = exposure.match_histograms(
            source[:, :, i], reference[:, :, i]
        )

    # Blend original and matched images using the mask
    result = source * (1 - mask) + matched * mask
    return np.clip(result, 0, 1)


def blend_images(
    original: np.ndarray, styled: np.ndarray, intensity: float
) -> np.ndarray:
    """
    Blend original and styled images based on intensity.

    Args:
        original: Original image array (H, W, 3)
        styled: Styled image array (H, W, 3)
        intensity: Blending intensity (0.0 to 1.0), where 0.0 = original,
                  1.0 = fully styled

    Returns:
        Blended image array (H, W, 3) in range [0, 1]
    """
    intensity = np.clip(intensity, 0.0, 1.0)
    return original * (1 - intensity) + styled * intensity


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
        self.method_combo.addItem("Histogram Matching", "histogram")
        self.method_combo.addItem("Mean/Std Transfer", "meanstd")
        self.method_combo.addItem("LUT + Linear Curve", "lut_linear")
        self.method_combo.addItem("LUT + S-Curve", "lut_scurve")
        self.method_combo.addItem("LUT + Contrast", "lut_contrast")
        self.method_combo.addItem("Selective: Shadows", "selective_shadows")
        self.method_combo.addItem("Selective: Midtones", "selective_midtones")
        self.method_combo.addItem(
            "Selective: Highlights", "selective_highlights"
        )
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

        intensity_labels_layout.addWidget(QLabel("Style Intensity:"))
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

    def load_content_image(self) -> None:
        """Load content image from file dialog."""
        content_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Content Image",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All Files (*)",
        )
        if content_path:
            try:
                self.content_image = load_image(content_path)
                self.show_image(self.content_image, self.content_label)
                self.show_conversion_info("content", content_path)
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self, "Error", f"Failed to load content image: {str(e)}"
                )
                logger.error(f"Failed to load content image: {e}")

    def load_style_image(self) -> None:
        """Load style image from file dialog."""
        style_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Style Image",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All Files (*)",
        )
        if style_path:
            try:
                self.style_image = load_image(style_path)
                self.show_image(self.style_image, self.style_label)
                self.show_conversion_info("style", style_path)
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self, "Error", f"Failed to load style image: {str(e)}"
                )
                logger.error(f"Failed to load style image: {e}")

    def show_conversion_info(self, image_type: str, path: str) -> None:
        """
        Show information about any image conversions that occurred.

        Args:
            image_type: Type of image ('content' or 'style')
            path: Path to image file
        """
        try:
            original_img = img_as_float(io.imread(path))
            if original_img.ndim == 2:
                QtWidgets.QMessageBox.information(
                    self,
                    "Image Conversion",
                    f"Grayscale {image_type} image automatically "
                    f"converted to RGB for processing.",
                )
            elif original_img.ndim == 3 and original_img.shape[2] == 4:
                QtWidgets.QMessageBox.information(
                    self,
                    "Image Conversion",
                    f"{image_type.title()} image with transparency "
                    f"(alpha channel) detected.\nAlpha channel removed for processing.",
                )
        except (IOError, ValueError) as e:
            logger.warning(f"Failed to show conversion info: {e}")

    def update_intensity(self, value: int) -> None:
        """
        Update intensity slider value and trigger debounced update.

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
        """
        Handle transfer method selection change.

        Args:
            index: Index of selected method in combo box
        """
        self.transfer_method = self.method_combo.itemData(index)
        if self.content_image is not None and self.style_image is not None:
            self.apply_style_transfer()

    def apply_style_transfer(self) -> None:
        """Apply selected style transfer method to images."""
        if self.content_image is not None and self.style_image is not None:
            try:
                if self.transfer_method == "histogram":
                    self.styled_image = match_histograms_multichannel(
                        self.content_image, self.style_image
                    )
                elif self.transfer_method == "meanstd":
                    self.styled_image = color_transfer_meanstd(
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
                QtWidgets.QMessageBox.critical(
                    self, "Error", f"Failed to apply style transfer: {str(e)}"
                )
                logger.error(f"Style transfer failed: {e}")
        else:
            QtWidgets.QMessageBox.warning(
                self,
                "Missing Images",
                "Please load both content and style images "
                "before applying style transfer.",
            )

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
                    save_image = (
                        np.clip(self.result_image, 0, 1) * 255
                    ).astype(np.uint8)
                    io.imsave(save_path, save_image)
                    QtWidgets.QMessageBox.information(
                        self,
                        "Save Image",
                        f"Image successfully saved to: {save_path}",
                    )
                    logger.info(f"Image saved to: {save_path}")
                except Exception as e:
                    QtWidgets.QMessageBox.critical(
                        self, "Error", f"Failed to save image: {str(e)}"
                    )
                    logger.error(f"Failed to save image: {e}")
        else:
            QtWidgets.QMessageBox.warning(
                self,
                "No Image",
                "No result image to save. Please apply style transfer first.",
            )

    def show_image(self, img_array: np.ndarray, label: QLabel) -> None:
        """
        Display image array in QLabel.

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
    """Run the ColorCast application."""
    app = QApplication([])
    ex = StyleTransferApp()
    ex.show()
    app.exec()


if __name__ == "__main__":
    main()