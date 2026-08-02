"""Enhanced input validation utilities for ColorCast.

This module provides comprehensive security-focused validation including:
- Path traversal prevention
- Image file type validation (magic numbers)
- Numeric parameter validation
- Resource limit enforcement
"""

import math
import os

try:
    import imghdr as _imghdr

    _HAS_IMGHDR = True
except ImportError:
    _imghdr = None  # type: ignore[assignment]
    _HAS_IMGHDR = False
try:
    from PIL import Image as _PILImage

    _HAS_PIL = True
except ImportError:
    _PILImage = None  # type: ignore[assignment]
    _HAS_PIL = False
from pathlib import Path

import numpy as np

from colorcast.utils.exceptions import ValidationError

ALLOWED_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# Constants
MIN_IMAGE_SIZE = (1, 1)  # Minimum dimensions
MAX_IMAGE_PIXELS = 50000000  # 50MP default
MAX_IMAGE_DIMENSION = 8192  # 8K max dimension


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
        raise ValidationError(f"Image too large: {total_pixels:,} pixels (max: {max_pixels:,})")


def validate_file_path(
    path: str,
    allowed_extensions: tuple[str, ...] = ALLOWED_IMAGE_EXTENSIONS,
    allowed_base_dirs: tuple[Path, ...] | None = None,
    check_existence: bool = True,
) -> Path:
    """
    Validate file path is safe and has allowed extension.

    Comprehensive security validation including:
    - Path traversal prevention
    - Directory restriction
    - File extension validation
    - Existence and readability checks

    Args:
        path: File path to validate
        allowed_extensions: Tuple of allowed extensions
        allowed_base_dirs: Optional tuple of allowed base directories
        check_existence: Whether to check if file exists (skip for new files)

    Returns:
        Resolved Path object

    Raises:
        ValidationError: If path is invalid or unsafe
        FileNotFoundError: If file doesn't exist (when check_existence=True)

    Example:
        >>> path = validate_file_path("image.jpg", allowed_base_dirs=(Path("/safe"),))
        >>> # Raises ValidationError if path is outside /safe

    Security:
        - Prevents path traversal attacks (../)
        - Restricts to allowed directories
        - Validates file extension
        - Checks file permissions
    """
    path_obj = Path(path)

    # Check for path traversal in original path string
    if ".." in path_obj.parts:
        raise ValidationError("Path traversal attempt detected in path")

    # Resolve to absolute path
    try:
        resolved_path = path_obj.resolve()
    except (OSError, RuntimeError) as e:
        raise ValidationError(f"Invalid path: {e}") from e

    # Check if path is within allowed directories
    if allowed_base_dirs is not None:
        is_allowed = False
        for base_dir in allowed_base_dirs:
            resolved_base = base_dir.resolve()
            try:
                resolved_path.relative_to(resolved_base)
                is_allowed = True
                break
            except ValueError:
                continue
        if not is_allowed:
            raise ValidationError(
                f"Path not within allowed directories: " f"{[str(d) for d in allowed_base_dirs]}"
            )

    # Check file extension
    if path_obj.suffix.lower() not in allowed_extensions:
        raise ValidationError(
            f"Invalid file extension '{path_obj.suffix}'. "
            f"Allowed: {', '.join(allowed_extensions)}"
        )

    # Check file existence and readability
    if check_existence:
        if not resolved_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if not resolved_path.is_file():
            raise ValidationError(f"Not a file: {path}")
        if not os.access(resolved_path, os.R_OK):
            raise ValidationError(f"File not readable: {path}")

    return resolved_path


def _check_extension_matches(
    detected_ext: str,
    path_ext: str,
    fmt_name: str,
) -> None:
    """Raise ValidationError if detected extension does not match path extension.

    Common aliases (``.jpg`` ↔ ``.jpeg``, ``.tif`` ↔ ``.tiff``) are treated as
    equivalent.  A detected_ext of ``""`` means the format was not recognised;
    the call site decides whether that is an error.
    """
    if not detected_ext:
        return
    alias_groups = ({".jpg", ".jpeg"}, {".tif", ".tiff"})
    for group in alias_groups:
        if detected_ext in group and path_ext in group:
            return
    if detected_ext == path_ext:
        return
    raise ValidationError(
        f"File type mismatch. Detected: {fmt_name}, Extension: {path_ext}. "
        "File may be corrupted or renamed."
    )


_FORMAT_TO_EXT: dict[str, str] = {
    "jpeg": ".jpg",
    "png": ".png",
    "bmp": ".bmp",
    "tiff": ".tif",
    "gif": ".gif",
    "webp": ".webp",
}

_PIL_FORMAT_TO_EXT: dict[str, str] = {
    "JPEG": ".jpg",
    "PNG": ".png",
    "BMP": ".bmp",
    "TIFF": ".tif",
    "GIF": ".gif",
    "WEBP": ".webp",
}


def validate_image_file(
    path: str,
    allowed_extensions: tuple[str, ...] = ALLOWED_IMAGE_EXTENSIONS,
) -> None:
    """
    Validate that file is actually an image file using magic numbers.

    Uses magic numbers to detect actual file type, not just extension.
    This prevents file type spoofing attacks.

    Args:
        path: Path to image file
        allowed_extensions: Tuple of allowed extensions

    Raises:
        ValidationError: If file is not a valid image
        FileNotFoundError: If file doesn't exist

    Example:
        >>> validate_image_file("image.jpg")
        >>> # Passes if file is actually JPEG
        >>> validate_image_file("fake.jpg")
        >>> # Raises if file is not actually JPEG

    Security:
        - Validates actual file type, not just extension
        - Prevents file type spoofing
        - Uses magic number detection
    """
    path_obj = validate_file_path(
        path,
        allowed_extensions=allowed_extensions,
        check_existence=True,
    )

    if not _HAS_IMGHDR:
        if not _HAS_PIL:
            raise ValidationError(
                "Cannot validate image content: imghdr is unavailable "
                "(Python 3.13+) and Pillow is not installed."
            )
        try:
            with _PILImage.open(path_obj) as img:  # type: ignore[union-attr]
                fmt = img.format
                img.verify()
        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(f"File is not a valid image: {e}") from e

        path_ext = path_obj.suffix.lower()
        expected_ext = _PIL_FORMAT_TO_EXT.get(fmt or "")
        _check_extension_matches(expected_ext, path_ext, fmt or "unknown")
        return

    try:
        detected_type = _imghdr.what(path)
    except Exception as e:
        raise ValidationError(f"Failed to read file: {e}") from e

    if detected_type is None:
        raise ValidationError(f"File does not appear to be a valid image: {path}")

    expected_ext = _FORMAT_TO_EXT.get(detected_type, "")
    path_ext = path_obj.suffix.lower()
    _check_extension_matches(expected_ext, path_ext, detected_type.upper())
    if not expected_ext:
        raise ValidationError(
            f"Unsupported image type: {detected_type}. "
            f"Supported types: JPEG, PNG, BMP, TIFF, GIF, WebP"
        )


def validate_float_parameter(
    value: float,
    param_name: str,
    min_val: float | None = None,
    max_val: float | None = None,
    allow_nan: bool = False,
    allow_inf: bool = False,
) -> float:
    """
    Validate float parameter is within valid range and properly formed.

    Prevents NaN and Infinity attacks that could cause undefined behavior.

    Args:
        value: Parameter value to validate
        param_name: Name of parameter (for error messages)
        min_val: Optional minimum allowed value
        max_val: Optional maximum allowed value
        allow_nan: Whether NaN is allowed (default: False)
        allow_inf: Whether infinity is allowed (default: False)

    Returns:
        Validated float value

    Raises:
        ValidationError: If value is invalid

    Example:
        >>> alpha = validate_float_parameter(0.5, "alpha", 0.0, 1.0)
        >>> validate_float_parameter(float('nan'), "alpha", 0.0, 1.0)
        >>> ValidationError: alpha cannot be NaN

    Security:
        - Prevents NaN/Infinity injection
        - Validates numeric type
        - Enforces range constraints
    """
    # Check type
    if not isinstance(value, (int, float)):
        raise ValidationError(f"{param_name} must be a number, got {type(value).__name__}")

    # Check for NaN
    if not allow_nan and math.isnan(value):
        raise ValidationError(f"{param_name} cannot be NaN (Not a Number)")

    # Check for Infinity
    if not allow_inf and math.isinf(value):
        raise ValidationError(f"{param_name} cannot be infinite")

    # Check range
    value_float = float(value)
    if min_val is not None and value_float < min_val:
        raise ValidationError(f"{param_name} must be >= {min_val}, got {value_float}")

    if max_val is not None and value_float > max_val:
        raise ValidationError(f"{param_name} must be <= {max_val}, got {value_float}")

    return value_float


def validate_image_array(
    img_array: np.ndarray,
    min_pixels: int = 1,
    max_pixels: int = MAX_IMAGE_PIXELS,
    max_dimension: int = MAX_IMAGE_DIMENSION,
    required_channels: int | None = 3,
    dtype_range: tuple[float, float] = (0.0, 1.0),
) -> np.ndarray:
    """
    Validate image array meets requirements.

    Prevents memory exhaustion attacks and ensures proper format.

    Args:
        img_array: Image array to validate
        min_pixels: Minimum number of pixels
        max_pixels: Maximum number of pixels
        max_dimension: Maximum dimension (width or height)
        required_channels: Required number of channels (None for any)
        dtype_range: Expected value range (min, max)

    Returns:
        Validated image array (unchanged if valid)

    Raises:
        ValidationError: If image array is invalid

    Example:
        >>> img = validate_image_array(image, max_pixels=10000000)
        >>> # Raises if image has > 10 million pixels

    Security:
        - Prevents memory exhaustion with large images
        - Validates array dimensions
        - Checks for NaN/Inf values
        - Validates data type and range
    """
    # Check dimensions
    if img_array.ndim < 2:
        raise ValidationError(f"Image must have at least 2 dimensions, got {img_array.ndim}")

    if img_array.ndim > 3:
        raise ValidationError(f"Image must have at most 3 dimensions, got {img_array.ndim}")

    # Check channels
    if img_array.ndim == 3:
        channels = img_array.shape[2]
        if required_channels is not None and channels != required_channels:
            raise ValidationError(f"Image must have {required_channels} channels, got {channels}")

    # Check dimensions
    height, width = img_array.shape[:2]

    if height < MIN_IMAGE_SIZE[0] or width < MIN_IMAGE_SIZE[1]:
        raise ValidationError(
            f"Image too small: {height}x{width} (minimum: {MIN_IMAGE_SIZE[0]}x{MIN_IMAGE_SIZE[1]})"
        )

    if height > max_dimension or width > max_dimension:
        raise ValidationError(
            f"Image dimension too large: {height}x{width} " f"(maximum dimension: {max_dimension})"
        )

    # Check pixel count
    total_pixels = height * width
    if total_pixels < min_pixels:
        raise ValidationError(
            f"Image too small: {total_pixels:,} pixels " f"(minimum: {min_pixels:,})"
        )

    if total_pixels > max_pixels:
        raise ValidationError(
            f"Image too large: {total_pixels:,} pixels " f"(maximum: {max_pixels:,})"
        )

    # Check for invalid values (NaN, Inf)
    if np.any(np.isnan(img_array)):
        raise ValidationError("Image contains NaN (Not a Number) values")

    if np.any(np.isinf(img_array)):
        raise ValidationError("Image contains infinite values")

    # Check value range
    if dtype_range:
        min_val, max_val = dtype_range
        if np.any(img_array < min_val) or np.any(img_array > max_val):
            # Find actual min/max for error message
            actual_min = float(np.min(img_array))
            actual_max = float(np.max(img_array))
            raise ValidationError(
                f"Image values out of range: [{actual_min:.4f}, {actual_max:.4f}] "
                f"(expected: [{min_val}, {max_val}])"
            )

    return img_array
