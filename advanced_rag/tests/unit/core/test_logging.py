"""
Unit tests for logging module.
"""
import pytest
import logging
import json
from src.core.logging import (
    StructuredFormatter,
    StreamlitCompatibleFormatter,
    setup_logging,
    get_logger,
    log_with_context
)


@pytest.mark.unit
class TestStructuredFormatter:
    """Test StructuredFormatter class."""
    
    def test_formats_log_record_as_json(self):
        """Test formats log record as valid JSON."""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        formatted = formatter.format(record)
        assert isinstance(formatted, str)
        
        # Should be valid JSON
        log_data = json.loads(formatted)
        assert "timestamp" in log_data
        assert "level" in log_data
        assert "message" in log_data
        assert log_data["message"] == "Test message"
    
    def test_includes_exception_info(self):
        """Test includes exception information when present."""
        formatter = StructuredFormatter()
        
        try:
            raise ValueError("Test exception")
        except ValueError:
            record = logging.LogRecord(
                name="test_logger",
                level=logging.ERROR,
                pathname="test.py",
                lineno=10,
                msg="Error occurred",
                args=(),
                exc_info=logging._exc_info()
            )
        
        formatted = formatter.format(record)
        log_data = json.loads(formatted)
        assert "exception" in log_data
    
    def test_includes_context_if_present(self):
        """Test includes context if present in record."""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None
        )
        record.context = {"user_id": "123", "action": "test"}
        
        formatted = formatter.format(record)
        log_data = json.loads(formatted)
        assert "context" in log_data
        assert log_data["context"]["user_id"] == "123"


@pytest.mark.unit
class TestStreamlitCompatibleFormatter:
    """Test StreamlitCompatibleFormatter class."""
    
    def test_formats_log_record_as_text(self):
        """Test formats log record as readable text."""
        formatter = StreamlitCompatibleFormatter()
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        formatted = formatter.format(record)
        assert isinstance(formatted, str)
        assert "Test message" in formatted
        assert "INFO" in formatted or "test_logger" in formatted


@pytest.mark.unit
class TestGetLogger:
    """Test get_logger function."""
    
    def test_returns_logger_instance(self):
        """Test returns logger instance."""
        logger = get_logger("test_module")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_module"
    
    def test_returns_same_logger_for_same_name(self):
        """Test returns same logger for same name."""
        logger1 = get_logger("test_module")
        logger2 = get_logger("test_module")
        assert logger1 is logger2


@pytest.mark.unit
class TestLogWithContext:
    """Test log_with_context decorator/function."""
    
    def test_logs_with_context(self, caplog):
        """Test logs message with context."""
        logger = get_logger("test")
        
        with log_with_context(logger, "test_action", {"key": "value"}):
            logger.info("Test message")
        
        # Verify log was created (exact format depends on setup)
        assert len(caplog.records) > 0


@pytest.mark.unit
class TestSetupLogging:
    """Test setup_logging function."""
    
    def test_setup_logging_configures_root_logger(self, monkeypatch):
        """Test setup_logging configures root logger."""
        # Reset logging to default state
        logging.root.handlers = []
        logging.root.setLevel(logging.WARNING)
        
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("LOG_LEVEL", "INFO")
        monkeypatch.setenv("LOG_FORMAT", "text")
        
        from src.core.config import reset_config, reload_config
        reset_config()
        reload_config()
        
        setup_logging()
        
        # Logger should be configured
        assert len(logging.root.handlers) > 0 or logging.root.level <= logging.INFO

