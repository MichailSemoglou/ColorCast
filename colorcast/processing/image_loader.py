"""Image loading and preprocessing utilities."""

from pathlib import Path
from typing import Optional
import numpy as np
from skimage import io, img_as_float, transform
from colorcast.utils.validators import (
    validate_file_path,
    validate_image_size,
    ALLOWED_IMAGE_EXTENSIONS,
)
from colorcast.utils.exceptions import ImageLoadError, InvalidImageFormatError


def ensure_rgb(img: np.ndarray) -> np.ndarray:
    """
    Convert image to RGB format, handling grayscale and RGBA images.

    Args:
        img: Input image array (2D, 3D with 1, 3, or 4 channels)

    Returns:
        RGB image array (H, W, 3)

    Raises:
        InvalidImageFormatError: If image has unsupported number of channels
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
            raise InvalidImageFormatError(
                f"Unsupported number of channels: {img.shape[2]}"
            )
    else:
        raise InvalidImageFormatError(f"Unsupported image dimensions: {img.ndim}")


def load_image(
    path: str, max_pixels: int = 50000000, max_dimension: Optional[int] = None
) -> np.ndarray:
    """
    Load and validate an image file.

    Args:
        path: Path to image file
        max_pixels: Maximum allowed pixels (default: 50MP)
        max_dimension: If set, resize so max dimension doesn't exceed this value

    Returns:
        RGB image array in float format [0, 1]

    Raises:
        FileNotFoundError: If file doesn't exist
        InvalidImageFormatError: If image format is unsupported
        ImageLoadError: If image fails to load
        ValidationError: If image size exceeds limits
    """
    # Validate file path and format
    validate_file_path(path, ALLOWED_IMAGE_EXTENSIONS)

    # Load image
    try:
        img = img_as_float(io.imread(path))
    except IOError as e:
        raise ImageLoadError(f"Failed to read image file: {e}")
    except Exception as e:
        raise ImageLoadError(f"Unexpected error loading image: {e}")

    # Convert to RGB
    try:
        img = ensure_rgb(img)
    except ValueError as e:
        raise InvalidImageFormatError(f"Invalid image format: {e}")

    # Apply dimension limit if specified
    if max_dimension is not None:
        h, w = img.shape[:2]
        if max(h, w) > max_dimension:
            scale = max_dimension / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            img = transform.resize(
                img, (new_h, new_w), anti_aliasing=True, preserve_range=True
            )

    # Validate image size
    validate_image_size(img, max_pixels)

    return img


def save_image(img_array: np.ndarray, path: str) -> None:
    """
    Save image with path validation.

    Args:
        img_array: Image array to save (float [0,1] or uint8 [0,255])
        path: Path where image will be saved

    Raises:
        ValidationError: If path is invalid
        ImageProcessingError: If save fails
    """
    from colorcast.utils.exceptions import ImageProcessingError

    # Validate file extension (don't check existence for new files)
    path_obj = Path(path)
    if path_obj.suffix.lower() not in ALLOWED_IMAGE_EXTENSIONS:
        from colorcast.utils.exceptions import ValidationError
        raise ValidationError(
            f"Invalid file extension. Allowed: {', '.join(ALLOWED_IMAGE_EXTENSIONS)}"
        )

    # Ensure directory exists
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    # Convert to uint8 if needed
    if img_array.dtype == np.float64 or img_array.dtype == np.float32:
        save_img = (np.clip(img_array, 0, 1) * 255).astype(np.uint8)
    else:
        save_img = img_array

    # Save with error handling
    try:
        io.imsave(path, save_img)
    except Exception as e:
        raise ImageProcessingError(f"Failed to save image: {e}")
