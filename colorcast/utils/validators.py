"""Deprecated – import from ``colorcast.utils.validators_enhanced`` instead."""
import warnings
from colorcast.utils.validators_enhanced import (  # noqa: F401
    ALLOWED_IMAGE_EXTENSIONS,
    validate_file_path,
    validate_image_size,
)

warnings.warn(
    "colorcast.utils.validators is deprecated; "
    "import from colorcast.utils.validators_enhanced instead.",
    DeprecationWarning,
    stacklevel=2,
)
