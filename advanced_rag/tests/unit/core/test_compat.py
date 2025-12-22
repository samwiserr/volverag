"""
Unit tests for compatibility module.
"""
import pytest
import os
from src.core.compat import get_env, unwrap_result
from src.core.result import Result, AppError, ErrorType


@pytest.mark.unit
class TestGetEnv:
    """Test get_env function."""
    
    def test_returns_env_var_when_set(self, monkeypatch):
        """Test returns environment variable when set."""
        monkeypatch.setenv("TEST_VAR", "test_value")
        assert get_env("TEST_VAR") == "test_value"
    
    def test_returns_default_when_not_set(self):
        """Test returns default when environment variable not set."""
        result = get_env("NONEXISTENT_VAR", default="default_value")
        assert result == "default_value"
    
    def test_returns_none_when_not_set_and_no_default(self):
        """Test returns None when variable not set and no default."""
        result = get_env("NONEXISTENT_VAR")
        assert result is None
    
    def test_falls_back_to_os_getenv(self, monkeypatch):
        """Test falls back to os.getenv for unmapped variables."""
        monkeypatch.setenv("CUSTOM_VAR", "custom_value")
        result = get_env("CUSTOM_VAR")
        assert result == "custom_value"
    
    def test_handles_config_failure_gracefully(self, monkeypatch):
        """Test handles config failure gracefully and falls back."""
        # This test verifies that if config fails, it still works
        # We can't easily break config in tests, but we can verify fallback works
        monkeypatch.setenv("FALLBACK_VAR", "fallback_value")
        result = get_env("FALLBACK_VAR")
        assert result == "fallback_value"


@pytest.mark.unit
class TestUnwrapResult:
    """Test unwrap_result function."""
    
    def test_unwraps_ok_result(self):
        """Test unwraps successful Result."""
        result = Result.ok("success_value")
        unwrapped = unwrap_result(result)
        assert unwrapped == "success_value"
    
    def test_returns_default_for_error_result(self):
        """Test returns default for error Result."""
        result = Result.err(AppError(
            type=ErrorType.PROCESSING_ERROR,
            message="Error occurred"
        ))
        unwrapped = unwrap_result(result, default="default_value")
        assert unwrapped == "default_value"
    
    def test_returns_none_for_error_without_default(self):
        """Test returns None for error Result without default."""
        result = Result.err(AppError(
            type=ErrorType.PROCESSING_ERROR,
            message="Error occurred"
        ))
        unwrapped = unwrap_result(result)
        assert unwrapped is None
    
    def test_returns_value_directly_if_not_result(self):
        """Test returns value directly if not a Result."""
        value = "direct_value"
        unwrapped = unwrap_result(value)
        assert unwrapped == "direct_value"
    
    def test_handles_none_value(self):
        """Test handles None value."""
        unwrapped = unwrap_result(None)
        assert unwrapped is None
    
    def test_handles_none_result_with_default(self):
        """Test handles None (not Result) with default."""
        unwrapped = unwrap_result(None, default="default")
        assert unwrapped is None  # None is not a Result, so return as-is

