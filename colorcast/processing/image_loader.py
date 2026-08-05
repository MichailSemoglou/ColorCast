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

import logging
import os
import threading
from pathlib import Path
from typing import NamedTuple

import numpy as np
from skimage import img_as_float, io, transform  # type: ignore[attr-defined]

from colorcast.utils.exceptions import ImageLoadError, InvalidImageFormatError, ValidationError
from colorcast.utils.validators_enhanced import (
    ALLOWED_IMAGE_EXTENSIONS,
    MAX_IMAGE_PIXELS,
    validate_image_array,
    validate_image_file,
)

logger = logging.getLogger(__name__)

_MAX_FILE_BYTES = 200_000_000  # 200 MB
_PIL_IMAGE_LOCK = threading.Lock()


def _read_image_array(filepath: str) -> np.ndarray:
    """Read an image with skimage and reject unsupported multi-frame stacks.

    Args:
        filepath: Path to the image file to read.

    Returns:
        np.ndarray: The decoded image array.
    """
    arr = io.imread(filepath)
    if arr.ndim > 3:
        raise InvalidImageFormatError(
            "Multi-frame image stacks are not supported; provide a single image frame."
        )
    if arr.ndim == 3 and arr.shape[-1] not in {1, 3, 4}:
        raise InvalidImageFormatError(
            "Multi-frame image stacks are not supported; provide a single image frame."
        )
    return arr


def _get_image_dimensions(filepath: str, max_pixels: int | None = None) -> tuple[int, int]:
    """Return ``(width, height)`` from the image header without decoding pixels.

    Uses PIL if available; on any PIL failure, decodes with
    ``skimage.io.imread`` (which allocates the full array) and reads the
    shape instead. ``DecompressionBombError`` is treated as a hard limit
    violation and re-raised so the caller can reject oversized images before
    a full decode.

    Args:
        filepath: Path to the image file to inspect.
        max_pixels: Optional pixel-limit override for the PIL header probe.

    Returns:
        tuple[int, int]: The image dimensions as ``(width, height)`` in pixels.
    """
    try:
        from PIL import Image as _PILImage
    except ImportError:
        _PILImage = None  # type: ignore[assignment]

    if _PILImage is not None:
        pixel_limit = max_pixels if max_pixels is not None else MAX_IMAGE_PIXELS
        with _PIL_IMAGE_LOCK:
            original_limit = getattr(_PILImage, "MAX_IMAGE_PIXELS", None)
            if pixel_limit is not None:
                _PILImage.MAX_IMAGE_PIXELS = pixel_limit
            try:
                with _PILImage.open(filepath) as img:
                    return img.size  # (width, height)
            except _PILImage.DecompressionBombError:
                raise
            except (OSError, ValueError):
                pass
            except Exception:
                logger.warning(
                    "PIL header read failed for %r; falling back to full decode", filepath
                )
            finally:
                if pixel_limit is not None:
                    if original_limit is None:
                        delattr(_PILImage, "MAX_IMAGE_PIXELS")
                    else:
                        _PILImage.MAX_IMAGE_PIXELS = original_limit

    arr = _read_image_array(filepath)
    if arr.ndim == 2:
        return (arr.shape[1], arr.shape[0])
    return (arr.shape[1], arr.shape[0])


class ImageMeta(NamedTuple):
    """Metadata about an image's original format before normalization."""

    original_ndim: int
    original_channels: int  # 0 when original_ndim < 3


def ensure_rgb(img: np.ndarray) -> np.ndarray:
    """
    Convert image to RGB format, handling grayscale and RGBA images.

    RGBA images are composited onto a white background
    (``out = rgb * alpha + white * (1 - alpha)``) instead of discarding the
    alpha channel, so transparent pixels do not produce wrong colors. The
    input dtype is preserved.

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
            if np.issubdtype(img.dtype, np.signedinteger):
                raise InvalidImageFormatError(
                    f"Signed integer dtype '{img.dtype}' is not supported for "
                    "RGBA compositing. Convert to an unsigned integer or "
                    "float dtype first."
                )
            if np.issubdtype(img.dtype, np.integer):
                if img.dtype.itemsize > 4:
                    raise InvalidImageFormatError(
                        f"RGBA compositing of '{img.dtype}' images is not "
                        "supported. Convert to float first."
                    )
                maxval = float(np.iinfo(img.dtype).max)
            else:
                maxval = 1.0
            rgb = img[:, :, :3].astype(np.float64) / maxval
            alpha = img[:, :, 3:4].astype(np.float64) / maxval
            composited: np.ndarray = np.clip(rgb * alpha + (1.0 - alpha), 0.0, 1.0)
            if np.issubdtype(img.dtype, np.integer):
                composited = np.round(composited * maxval)
            return composited.astype(img.dtype)
        else:
            raise InvalidImageFormatError(f"Unsupported number of channels: {img.shape[2]}")
    else:
        raise InvalidImageFormatError(f"Unsupported image dimensions: {img.ndim}")


def normalize_to_float32(array: np.ndarray) -> np.ndarray:
    """Normalize an image array to float32 in [0, 1].

    Unsigned integer dtypes (uint8, uint16, uint32) are divided by
    ``np.iinfo(dtype).max`` so that the full dynamic range maps to [0, 1].
    Float inputs are clipped to [0, 1] without rescaling.
    Signed integer dtypes are not supported and raise ``TypeError``.

    Args:
        array: Input image as an unsigned integer or float NumPy array with
            shape ``(H, W, 3)``.

    Returns:
        float32 array of shape ``(H, W, 3)`` with values clipped to [0, 1].

    Raises:
        TypeError: If ``array`` has a signed integer dtype.
        ValueError: If ``array`` does not have shape ``(H, W, 3)``.

    Example:
        >>> import numpy as np
        >>> img = np.full((4, 4, 3), 128, dtype=np.uint8)
        >>> out = normalize_to_float32(img)
        >>> out.shape
        (4, 4, 3)
        >>> out.dtype
        dtype('float32')
        >>> np.allclose(out, 128/255, atol=0.01)
        True
    """
    arr = np.asarray(array)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(
            f"Expected image with shape (H, W, 3), got {arr.shape}. "
            "Convert grayscale or multi-channel images with ensure_rgb() first."
        )
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
    path: str, max_pixels: int = 50000000, max_dimension: int | None = None
) -> tuple[np.ndarray, "ImageMeta"]:
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
    validate_image_file(path, ALLOWED_IMAGE_EXTENSIONS)

    file_bytes = os.path.getsize(path)
    if file_bytes > _MAX_FILE_BYTES:
        raise ValidationError(f"File too large: {file_bytes:,} bytes (max {_MAX_FILE_BYTES:,})")

    width, height = _get_image_dimensions(path, max_pixels=max_pixels)
    total_pixels = width * height
    if total_pixels > max_pixels:
        raise ValidationError(
            f"Image too large: {width}×{height} = {total_pixels:,} pixels " f"(max {max_pixels:,})"
        )

    try:
        img = img_as_float(_read_image_array(path))
    except OSError as e:
        raise ImageLoadError(f"Failed to read image file: {e}") from e
    except Exception as e:
        raise ImageLoadError(f"Unexpected error loading image: {e}") from e

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
            img = transform.resize(img, (new_h, new_w), anti_aliasing=True, preserve_range=True)

    validate_image_array(img, max_pixels=max_pixels)
    return img, meta


def load_image(
    path: str, max_pixels: int = 50000000, max_dimension: int | None = None
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
        img_array: Image array to save (float32/float64 in [0, 1] or uint8
            in [0, 255])
        path: Path where image will be saved

    Raises:
        ValidationError: If path is invalid
        InvalidImageFormatError: If the array dtype is not float32, float64,
            or uint8
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

    # Validate dtype and convert to uint8 if needed
    if img_array.dtype == np.float64 or img_array.dtype == np.float32:
        save_img = (np.clip(img_array, 0, 1) * 255).astype(np.uint8)
    elif img_array.dtype == np.uint8:
        save_img = img_array
    else:
        raise InvalidImageFormatError(
            f"Unsupported dtype for saving: {img_array.dtype}. "
            "Convert to float32, float64, or uint8 before saving."
        )

    # Ensure directory exists
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    # Save with error handling
    try:
        io.imsave(path, save_img)
    except Exception as e:
        raise ImageProcessingError(f"Failed to save image: {e}") from e
