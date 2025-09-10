"""
ColorCast - Style Transfer Application

A PyQt5 GUI application for style transfer using histogram matching (LUT transfer) between images.
Users can load a content image and a style image, apply histogram matching, and save the result.

Features:
- Supports RGB, grayscale, and RGBA images (automatically converts to RGB for processing)
- Automatic image format conversion and preprocessing
- Interactive GUI with image preview and save functionality

Dependencies: numpy, scikit-image, scipy, PyQt5
"""

import sys
import numpy as np
from skimage import io, img_as_float, exposure, transform
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QFileDialog, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QGridLayout, QWidget, QApplication
from PyQt5.QtGui import QPixmap, QImage

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

def apply_lut_transfer(content, style):
    return match_histograms_multichannel(content, style)

class StyleTransferApp(QWidget):
    def __init__(self):
        super().__init__()
        self.content_image = None
        self.style_image = None
        self.result_image = None
        self.setFixedSize(1000, 600)
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
        grid_layout = QGridLayout()
        button_layout = QHBoxLayout()

        grid_layout.addWidget(self.content_label, 0, 0)
        grid_layout.addWidget(self.style_label, 0, 1)
        grid_layout.addWidget(self.result_label, 0, 2)

        grid_layout.addWidget(self.load_content_button, 1, 0)
        grid_layout.addWidget(self.load_style_button, 1, 1)
        grid_layout.addWidget(self.apply_button, 1, 2)

        button_layout.addWidget(self.clear_button)
        button_layout.addWidget(self.save_button)

        main_layout.addLayout(grid_layout)
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

    def apply_style_transfer(self):
        if self.content_image is not None and self.style_image is not None:
            try:
                self.result_image = apply_lut_transfer(self.content_image, self.style_image)
                self.show_image(self.result_image, self.result_label)
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to apply style transfer: {str(e)}")
        else:
            QtWidgets.QMessageBox.warning(self, "Missing Images", "Please load both content and style images before applying style transfer.")

    def clear_images(self):
        self.content_image = None
        self.style_image = None
        self.result_image = None
        
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
