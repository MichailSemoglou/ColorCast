"""Image loading and preprocessing utilities.

Public API
----------
- :func:`load_image` — load an image file and return a normalized RGB array.
- :func:`load_image_with_meta` — same as :func:`load_image`, but also return
  metadata about the original format before RGB conversion.
- :func:`save_image` — save an image array to disk with extension validation.
- :func:`ensure_rgb` — convert grayscale or RGBA arrays to RGB.
- :func:`normalize_to_float32` — normalize any supported dtype to float32 in
  [0, 1].

Data structures
---------------
- :class:`ImageMeta` — metadata about an image's original dimensions and
  channel count before normalization.
"""

from pathlib import Path
from typing import NamedTuple, Optional, Tuple
import numpy as np
from skimage import io, img_as_float, transform
from colorcast.utils.validators import (
    validate_file_path,
    validate_image_size,
    ALLOWED_IMAGE_EXTENSIONS,
)
from colorcast.utils.exceptions import ImageLoadError, InvalidImageFormatError


class ImageMeta(NamedTuple):
    """Metadata about an image's original format before normalization."""

    original_ndim: int
    original_channels: int  # 0 when original_ndim < 3


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


def normalize_to_float32(array: np.ndarray) -> np.ndarray:
    """Normalize an image array to float32 in [0, 1].

    Unsigned integer dtypes (uint8, uint16, uint32) are divided by
    ``np.iinfo(dtype).max`` so that the full dynamic range maps to [0, 1].
    Float inputs are clipped to [0, 1] without rescaling.
    Signed integer dtypes are not supported and raise ``TypeError``.

    Args:
        array: Input image as an unsigned integer or float NumPy array.

    Returns:
        float32 array with values clipped to [0, 1].

    Raises:
        TypeError: If ``array`` has a signed integer dtype.

    Example:
        >>> import numpy as np
        >>> normalize_to_float32(np.array([0, 128, 255], dtype=np.uint8))
        array([0.       , 0.5019608, 1.       ], dtype=float32)
    """
    arr = np.asarray(array)
    if np.issubdtype(arr.dtype, np.signedinteger):
        raise TypeError(
            f"Signed integer dtype '{arr.dtype}' is not supported. "
            "Convert to an unsigned integer or float dtype before normalizing."
        )
    if np.issubdtype(arr.dtype, np.unsignedinteger):
        scale = float(np.iinfo(arr.dtype).max)
        return np.clip(arr.astype(np.float32) / scale, 0.0, 1.0)
    return np.clip(arr.astype(np.float32), 0.0, 1.0)


def load_image_with_meta(
    path: str, max_pixels: int = 50000000, max_dimension: Optional[int] = None
) -> Tuple[np.ndarray, "ImageMeta"]:
    """
    Load and validate an image file, returning the array and original format metadata.

    Identical to :func:`load_image` but also returns an :class:`ImageMeta` describing
    the image's shape before ``ensure_rgb`` is applied, so callers can report
    conversions (grayscale→RGB, RGBA→RGB) without re-opening the file.

    Args:
        path: Path to image file
        max_pixels: Maximum allowed pixels (default: 50MP)
        max_dimension: If set, resize so max dimension doesn't exceed this value

    Returns:
        Tuple of (RGB float32 array in [0, 1], ImageMeta)

    Raises:
        FileNotFoundError: If file doesn't exist
        InvalidImageFormatError: If image format is unsupported
        ImageLoadError: If image fails to load
        ValidationError: If image size exceeds limits
    """
    validate_file_path(path, ALLOWED_IMAGE_EXTENSIONS)

    try:
        img = img_as_float(io.imread(path))
    except IOError as e:
        raise ImageLoadError(f"Failed to read image file: {e}")
    except Exception as e:
        raise ImageLoadError(f"Unexpected error loading image: {e}")

    meta = ImageMeta(
        original_ndim=img.ndim,
        original_channels=img.shape[2] if img.ndim == 3 else 0,
    )

    img = ensure_rgb(img)

    if max_dimension is not None:
        h, w = img.shape[:2]
        if max(h, w) > max_dimension:
            scale = max_dimension / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            img = transform.resize(
                img, (new_h, new_w), anti_aliasing=True, preserve_range=True
            )

    validate_image_size(img, max_pixels)
    return img, meta


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
    img, _ = load_image_with_meta(path, max_pixels=max_pixels, max_dimension=max_dimension)
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
