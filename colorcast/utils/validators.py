"""Input validation utilities for ColorCast."""

import os
from pathlib import Path
from typing import Tuple
import numpy as np
from colorcast.utils.exceptions import ValidationError

ALLOWED_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def validate_file_path(path: str, allowed_extensions: Tuple[str, ...] = ALLOWED_IMAGE_EXTENSIONS) -> None:
    """
    Validate file path is safe and has allowed extension.

    Args:
        path: File path to validate
        allowed_extensions: Tuple of allowed extensions (e.g., ('.jpg', '.png'))

    Raises:
        ValidationError: If path is invalid or unsafe
    """
    # Check for obvious path traversal attempts in the path string
    if ".." in Path(path).parts:
        raise ValidationError("Path traversal attempt detected")

    # Check file extension
    path_obj = Path(path)
    if path_obj.suffix.lower() not in allowed_extensions:
        raise ValidationError(
            f"Invalid file extension. Allowed: {', '.join(allowed_extensions)}"
        )

    # Check file exists and is readable
    resolved_path = Path(path).resolve()
    if not resolved_path.exists():
        raise ValidationError(f"File not found: {path}")
    if not resolved_path.is_file():
        raise ValidationError(f"Not a file: {path}")
    if not os.access(resolved_path, os.R_OK):
        raise ValidationError(f"File not readable: {path}")


def validate_image_size(img_array: np.ndarray, max_pixels: int = 50000000) -> None:
    """
    Validate image size is within reasonable limits.

    Args:
        img_array: Image array to validate
        max_pixels: Maximum allowed pixels (default: 50MP)

    Raises:
        ValidationError: If image is too large
    """
    if img_array.ndim < 2:
        raise ValidationError("Image must have at least 2 dimensions")

    total_pixels = img_array.shape[0] * img_array.shape[1]
    if total_pixels > max_pixels:
        raise ValidationError(
            f"Image too large: {total_pixels:,} pixels " f"(max: {max_pixels:,})"
        )