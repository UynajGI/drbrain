"""Custom exception hierarchy for DrBrain."""

from __future__ import annotations


class DrBrainError(Exception):
    """Base exception for all DrBrain errors."""


class ConfigError(DrBrainError):
    """Configuration loading or validation error."""


class APIError(DrBrainError):
    """External API call failed."""


class APIRateLimitError(APIError):
    """API rate limit exceeded."""


class ExtractionError(DrBrainError):
    """Concept/argument extraction failed."""


class StorageError(DrBrainError):
    """Database or file storage error."""
