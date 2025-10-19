"""
ColorCast - Advanced Color Transfer Suite

A sophisticated PyQt5 GUI application for advanced color and style transfer between images.
ColorCast offers 8 transfer algorithms including histogram matching, mean/std transfer, 
LUT-based curves, and selective regional color transfer.

Features:
- 8 sophisticated transfer algorithms (histogram, statistical, LUT curves, selective regional)
- Real-time intensity control with smooth slider (0-100%)
- Selective regional color transfer (shadows/midtones/highlights)
- LUT-based transfer with multiple curve options (linear, s-curve, contrast)
- Supports RGB, grayscale, and RGBA images (automatically converts to RGB for processing)
- Optimized performance with smart caching and debounced slider updates
- Interactive GUI with image preview and save functionality

Dependencies: numpy, scikit-image, scipy, PyQt5
"""

import sys
import numpy as np
from skimage import io, img_as_float, exposure, transform
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QFileDialog, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QGridLayout, QWidget, QApplication, QSlider, QComboBox
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer

def load_image(path):
    try:
        img = img_as_float(io.imread(path))
        return ensure_rgb(img)
    except Exception as e:
        raise ValueError(f"Could not load image from {path}: {str(e)}")

def ensure_rgb(img):
    """Convert image to RGB format, handling grayscale and RGBA images."""
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

def match_histograms_multichannel(source, reference):
    if source.shape != reference.shape:
        print(f"Resizing style image from {reference.shape} to {source.shape}")
        reference = transform.resize(reference, source.shape, anti_aliasing=True, preserve_range=True)
    
    if source.ndim != 3 or source.shape[2] != 3:
        raise ValueError("Source image is not 3-channel RGB after preprocessing.")
    if reference.ndim != 3 or reference.shape[2] != 3:
        raise ValueError("Reference image is not 3-channel RGB after preprocessing.")
    
    matched = np.empty_like(source)
    for i in range(3):
        matched[:,:,i] = exposure.match_histograms(source[:,:,i], reference[:,:,i])
    return matched

def color_transfer_meanstd(source, reference):
    """Transfer color using mean and standard deviation matching per channel."""
    if source.shape != reference.shape:
        print(f"Resizing style image from {reference.shape} to {source.shape}")
        reference = transform.resize(reference, source.shape, anti_aliasing=True, preserve_range=True)
    
    if source.ndim != 3 or source.shape[2] != 3:
        raise ValueError("Source image is not 3-channel RGB after preprocessing.")
    if reference.ndim != 3 or reference.shape[2] != 3:
        raise ValueError("Reference image is not 3-channel RGB after preprocessing.")
    
    result = np.empty_like(source)
    for i in range(3):
        source_mean = np.mean(source[:,:,i])
        source_std = np.std(source[:,:,i])
        ref_mean = np.mean(reference[:,:,i])
        ref_std = np.std(reference[:,:,i])
        
        result[:,:,i] = ((source[:,:,i] - source_mean) * (ref_std / (source_std + 1e-8))) + ref_mean
    
    return np.clip(result, 0, 1)

def apply_curve(values, curve_type='linear'):
    """Apply tone curve to values."""
    if curve_type == 'linear':
        return values
    elif curve_type == 's-curve':
        return 0.5 + 0.5 * np.sin(np.pi * (values - 0.5))
    elif curve_type == 'contrast':
        return np.power(values, 0.8)
    return values

def lut_transfer_with_curve(source, reference, curve_type='linear'):
    """LUT-based transfer with curve adjustment."""
    if source.shape != reference.shape:
        print(f"Resizing style image from {reference.shape} to {source.shape}")
        reference = transform.resize(reference, source.shape, anti_aliasing=True, preserve_range=True)
    
    if source.ndim != 3 or source.shape[2] != 3:
        raise ValueError("Source image is not 3-channel RGB after preprocessing.")
    if reference.ndim != 3 or reference.shape[2] != 3:
        raise ValueError("Reference image is not 3-channel RGB after preprocessing.")
    
    matched = np.empty_like(source)
    for i in range(3):
        matched[:,:,i] = exposure.match_histograms(source[:,:,i], reference[:,:,i])
        matched[:,:,i] = apply_curve(matched[:,:,i], curve_type)
    
    return np.clip(matched, 0, 1)

def selective_color_transfer(source, reference, mode='full', shadow_threshold=0.3, highlight_threshold=0.7):
    """Transfer colors selectively based on luminance regions."""
    if source.shape != reference.shape:
        print(f"Resizing style image from {reference.shape} to {source.shape}")
        reference = transform.resize(reference, source.shape, anti_aliasing=True, preserve_range=True)
    
    if source.ndim != 3 or source.shape[2] != 3:
        raise ValueError("Source image is not 3-channel RGB after preprocessing.")
    if reference.ndim != 3 or reference.shape[2] != 3:
        raise ValueError("Reference image is not 3-channel RGB after preprocessing.")
    
    source_lum = 0.299 * source[:,:,0] + 0.587 * source[:,:,1] + 0.114 * source[:,:,2]
    
    if mode == 'full':
        mask = np.ones_like(source_lum)
    elif mode == 'shadows':
        mask = (source_lum < shadow_threshold).astype(float)
    elif mode == 'midtones':
        mask = ((source_lum >= shadow_threshold) & (source_lum <= highlight_threshold)).astype(float)
    elif mode == 'highlights':
        mask = (source_lum > highlight_threshold).astype(float)
    else:
        mask = np.ones_like(source_lum)
    
    mask = np.stack([mask, mask, mask], axis=2)
    
    matched = np.empty_like(source)
    for i in range(3):
        matched[:,:,i] = exposure.match_histograms(source[:,:,i], reference[:,:,i])
    
    result = source * (1 - mask) + matched * mask
    return np.clip(result, 0, 1)

def apply_lut_transfer(content, style, intensity=1.0):
    matched = match_histograms_multichannel(content, style)
    return blend_images(content, matched, intensity)

def blend_images(original, styled, intensity):
    """Blend original and styled images based on intensity (0.0 to 1.0)."""
    intensity = np.clip(intensity, 0.0, 1.0)
    return original * (1 - intensity) + styled * intensity

class StyleTransferApp(QWidget):
    def __init__(self):
        super().__init__()
        self.content_image = None
        self.style_image = None
        self.result_image = None
        self.styled_image = None
        self.intensity = 1.0
        self.transfer_method = "histogram"
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self.apply_intensity_blend)
        self.setFixedSize(1000, 700)
        self.initUI()

    def initUI(self):
        self.setWindowTitle("ColorCast - Style Transfer")

        self.content_label = QLabel(self)
        self.style_label = QLabel(self)
        self.result_label = QLabel(self)
        
        for label in [self.content_label, self.style_label, self.result_label]:
            label.setMinimumSize(300, 300)
            label.setStyleSheet("border: 1px solid gray;")
            label.setAlignment(QtCore.Qt.AlignCenter)
            label.setText("No image loaded")

        self.intensity_slider = QSlider(QtCore.Qt.Horizontal, self)
        self.intensity_slider.setMinimum(0)
        self.intensity_slider.setMaximum(100)
        self.intensity_slider.setValue(85)
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
        
        self.intensity_label = QLabel('85%', self)
        self.intensity_label.setAlignment(QtCore.Qt.AlignCenter)
        
        self.method_combo = QComboBox(self)
        self.method_combo.addItem("Histogram Matching", "histogram")
        self.method_combo.addItem("Mean/Std Transfer", "meanstd")
        self.method_combo.addItem("LUT + Linear Curve", "lut_linear")
        self.method_combo.addItem("LUT + S-Curve", "lut_scurve")
        self.method_combo.addItem("LUT + Contrast", "lut_contrast")
        self.method_combo.addItem("Selective: Shadows", "selective_shadows")
        self.method_combo.addItem("Selective: Midtones", "selective_midtones")
        self.method_combo.addItem("Selective: Highlights", "selective_highlights")
        self.method_combo.currentIndexChanged.connect(self.on_method_changed)
        
        self.load_content_button = QPushButton('Load Content Image', self)
        self.load_style_button = QPushButton('Load Style Image', self)
        self.apply_button = QPushButton('Apply Style Transfer', self)
        self.clear_button = QPushButton('Clear Images', self)
        self.save_button = QPushButton('Save Result', self)

        self.load_content_button.clicked.connect(self.load_content_image)
        self.load_style_button.clicked.connect(self.load_style_image)
        self.apply_button.clicked.connect(self.apply_style_transfer)
        self.clear_button.clicked.connect(self.clear_images)
        self.save_button.clicked.connect(self.save_result)

        main_layout = QVBoxLayout()
        method_layout = QHBoxLayout()
        grid_layout = QGridLayout()
        intensity_container = QVBoxLayout()
        intensity_slider_layout = QHBoxLayout()
        intensity_labels_layout = QHBoxLayout()
        button_layout = QHBoxLayout()

        method_layout.addWidget(QLabel('Transfer Method:'))
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
        
        intensity_labels_layout.addWidget(QLabel('Style Intensity:'))
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

    def load_content_image(self):
        content_path, _ = QFileDialog.getOpenFileName(self, 'Open Content Image', '', 'Image Files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All Files (*)')
        if content_path:
            try:
                self.content_image = load_image(content_path)
                self.show_image(self.content_image, self.content_label)
                self.show_conversion_info("content", content_path)
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load content image: {str(e)}")

    def load_style_image(self):
        style_path, _ = QFileDialog.getOpenFileName(self, 'Open Style Image', '', 'Image Files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All Files (*)')
        if style_path:
            try:
                self.style_image = load_image(style_path)
                self.show_image(self.style_image, self.style_label)
                self.show_conversion_info("style", style_path)
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load style image: {str(e)}")

    def show_conversion_info(self, image_type, path):
        """Show information about any image conversions that occurred."""
        try:
            original_img = img_as_float(io.imread(path))
            if original_img.ndim == 2:
                QtWidgets.QMessageBox.information(self, "Image Conversion", 
                    f"Grayscale {image_type} image automatically converted to RGB for processing.")
            elif original_img.ndim == 3 and original_img.shape[2] == 4:
                QtWidgets.QMessageBox.information(self, "Image Conversion", 
                    f"{image_type.title()} image with transparency (alpha channel) detected.\nAlpha channel removed for processing.")
        except:
            pass

    def update_intensity(self, value):
        self.intensity = value / 100.0
        self.intensity_label.setText(f'{value}%')
        if self.content_image is not None and self.styled_image is not None:
            self.update_timer.start(50)
    
    def apply_intensity_blend(self):
        if self.content_image is not None and self.styled_image is not None:
            self.result_image = blend_images(self.content_image, self.styled_image, self.intensity)
            self.show_image(self.result_image, self.result_label)
    
    def on_method_changed(self, index):
        self.transfer_method = self.method_combo.itemData(index)
        if self.content_image is not None and self.style_image is not None:
            self.apply_style_transfer()
    
    def apply_style_transfer(self):
        if self.content_image is not None and self.style_image is not None:
            try:
                if self.transfer_method == "histogram":
                    self.styled_image = match_histograms_multichannel(self.content_image, self.style_image)
                elif self.transfer_method == "meanstd":
                    self.styled_image = color_transfer_meanstd(self.content_image, self.style_image)
                elif self.transfer_method == "lut_linear":
                    self.styled_image = lut_transfer_with_curve(self.content_image, self.style_image, 'linear')
                elif self.transfer_method == "lut_scurve":
                    self.styled_image = lut_transfer_with_curve(self.content_image, self.style_image, 's-curve')
                elif self.transfer_method == "lut_contrast":
                    self.styled_image = lut_transfer_with_curve(self.content_image, self.style_image, 'contrast')
                elif self.transfer_method == "selective_shadows":
                    self.styled_image = selective_color_transfer(self.content_image, self.style_image, 'shadows')
                elif self.transfer_method == "selective_midtones":
                    self.styled_image = selective_color_transfer(self.content_image, self.style_image, 'midtones')
                elif self.transfer_method == "selective_highlights":
                    self.styled_image = selective_color_transfer(self.content_image, self.style_image, 'highlights')
                else:
                    self.styled_image = match_histograms_multichannel(self.content_image, self.style_image)
                
                self.result_image = blend_images(self.content_image, self.styled_image, self.intensity)
                self.show_image(self.result_image, self.result_label)
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to apply style transfer: {str(e)}")
        else:
            QtWidgets.QMessageBox.warning(self, "Missing Images", "Please load both content and style images before applying style transfer.")

    def clear_images(self):
        self.content_image = None
        self.style_image = None
        self.result_image = None
        self.styled_image = None
        
        for label in [self.content_label, self.style_label, self.result_label]:
            label.clear()
            label.setText("No image loaded")

    def save_result(self):
        if self.result_image is not None:
            options = QFileDialog.Options()
            file_types = "PNG Files (*.png);;JPEG Files (*.jpg);;TIFF Files(*.tiff);;BMP Files (*.bmp);;All Files (*.*)"
            save_path, _ = QFileDialog.getSaveFileName(self, 'Save Result Image', 'Untitled', file_types, options=options)
            if save_path:
                save_image = (np.clip(self.result_image, 0, 1) * 255).astype(np.uint8)
                io.imsave(save_path, save_image)
                QtWidgets.QMessageBox.information(self, "Save Image", f"Image successfully saved to: {save_path}")
        else:
            QtWidgets.QMessageBox.warning(self, "No Image", "No result image to save. Please apply the style transfer first.")

    def show_image(self, img_array, label):
        img_array = np.clip(img_array, 0, 1)
        img_array = (img_array * 255).astype(np.uint8)
        
        if img_array.ndim == 2:
            h, w = img_array.shape
            bytes_per_line = w
            qt_image = QImage(img_array.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
        elif img_array.ndim == 3 and img_array.shape[2] == 3:
            h, w, ch = img_array.shape
            bytes_per_line = ch * w
            qt_image = QImage(img_array.data, w, h, bytes_per_line, QImage.Format_RGB888)
        else:
            raise ValueError(f"Unsupported image format for display: {img_array.shape}")
            
        label.setPixmap(QPixmap.fromImage(qt_image).scaled(300, 300, QtCore.Qt.KeepAspectRatio))

def main():
    app = QApplication(sys.argv)
    ex = StyleTransferApp()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
