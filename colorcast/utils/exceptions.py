"""Custom exceptions for ColorCast."""


class ImageProcessingError(Exception):
    """Base exception for image processing errors."""

    def __init__(self, message: str, *, show_path: bool = False):
        """
        Initialize image processing error.

        Args:
            message: Error message
            show_path: If False, sanitizes file paths from message for security
        """
        if not show_path:
            import re

            # Sanitize file paths from message
            message = re.sub(r"/[^/]+/[^/]+/", r"/***/", message)
            message = re.sub(r"\\[^\\]+\\[^\\]+\\", r"\\***\\", message)
        super().__init__(message)


class ImageLoadError(ImageProcessingError):
    """Raised when image fails to load."""


class InvalidImageFormatError(ImageProcessingError):
    """Raised when image format is not supported."""


class ValidationError(ImageProcessingError):
    """Raised when input validation fails."""