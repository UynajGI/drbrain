"""Tests for the custom exception hierarchy."""

from __future__ import annotations

import pytest

from drbrain.exceptions import (
    APIError,
    APIRateLimitError,
    ConfigError,
    DrBrainError,
    ExtractionError,
    StorageError,
)
from drbrain.storage.workspace import WorkspaceError


class TestExceptionHierarchy:
    """DrBrainError is the base; all subclasses inherit from it."""

    def test_drbrain_error_is_base(self):
        assert issubclass(DrBrainError, Exception)

    def test_config_error_extends_drbrain_error(self):
        assert issubclass(ConfigError, DrBrainError)

    def test_api_error_extends_drbrain_error(self):
        assert issubclass(APIError, DrBrainError)

    def test_api_rate_limit_is_api_error(self):
        assert issubclass(APIRateLimitError, APIError)

    def test_extraction_error_extends_drbrain_error(self):
        assert issubclass(ExtractionError, DrBrainError)

    def test_storage_error_extends_drbrain_error(self):
        assert issubclass(StorageError, DrBrainError)

    def test_workspace_error_extends_drbrain_error(self):
        assert issubclass(WorkspaceError, DrBrainError)

    def test_can_raise_and_catch_drbrain_error(self):
        with pytest.raises(DrBrainError):
            raise WorkspaceError("test")


class TestExceptionIsinstance:
    """isinstance checks work for the hierarchy."""

    def test_api_rate_limit_is_api_error_via_isinstance(self):
        err = APIRateLimitError("rate limited")
        assert isinstance(err, APIRateLimitError)
        assert isinstance(err, APIError)
        assert isinstance(err, DrBrainError)
        assert isinstance(err, Exception)

    def test_config_error_isinstance(self):
        err = ConfigError("bad config")
        assert isinstance(err, DrBrainError)
        assert not isinstance(err, APIError)

    def test_workspace_error_isinstance(self):
        err = WorkspaceError("bad workspace")
        assert isinstance(err, DrBrainError)
        assert not isinstance(err, StorageError)
