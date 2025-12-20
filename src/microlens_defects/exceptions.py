"""Custom exception hierarchy for microlens-defects."""

from __future__ import annotations


class MicrolensError(Exception):
    """Base exception for all microlens-defects errors."""

    pass


class DatabaseError(MicrolensError):
    """Raised when database operations fail."""

    pass


class ImageLoadError(MicrolensError):
    """Raised when image loading fails."""

    pass


class DetectionError(MicrolensError):
    """Raised during detection processing errors."""

    pass


class ConfigurationError(MicrolensError):
    """Raised for invalid configuration or parameters."""

    pass
