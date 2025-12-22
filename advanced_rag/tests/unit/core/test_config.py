"""
Unit tests for configuration management.
"""
import pytest
import os
from pathlib import Path
from src.core.config import (
    AppConfig,
    get_config,
    reload_config,
    EmbeddingModel,
    LLMModel,
    LogLevel,
)
from src.core.exceptions import ConfigurationError


@pytest.mark.unit
class TestAppConfig:
    """Test AppConfig class."""
    
    def test_config_loads_from_env(self, monkeypatch):
        """Test config loads from environment variables."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-123")
        monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")
        monkeypatch.setenv("CHUNK_SIZE", "600")
        
        reload_config()
        config = get_config()
        
        assert config.openai_api_key == "test-key-123"
        assert config.llm_model == LLMModel.GPT_4O_MINI
        assert config.chunk_size == 600
    
    def test_config_uses_defaults(self, monkeypatch):
        """Test config uses sensible defaults."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        
        reload_config()
        config = get_config()
        
        assert config.embedding_model == EmbeddingModel.TEXT_EMBEDDING_3_SMALL
        assert config.llm_model == LLMModel.GPT_4O
        assert config.chunk_size == 500
        assert config.chunk_overlap == 150
        assert config.mmr_lambda == 0.7
    
    def test_config_validates_chunk_overlap(self, monkeypatch):
        """Test config validates chunk_overlap < chunk_size."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("CHUNK_SIZE", "400")
        monkeypatch.setenv("CHUNK_OVERLAP", "450")  # Invalid: > chunk_size
        
        reload_config()
        with pytest.raises(ConfigurationError):
            get_config()
    
    def test_config_validates_log_format(self, monkeypatch):
        """Test config validates log_format."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("LOG_FORMAT", "invalid")  # Invalid format
        
        reload_config()
        with pytest.raises(ConfigurationError):
            get_config()
    
    def test_config_path_resolution(self, monkeypatch, tmp_path):
        """Test config resolves paths correctly."""
        test_path = tmp_path / "test_vectorstore"
        test_path.mkdir()
        
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("VECTORSTORE_PATH", str(test_path))
        
        reload_config()
        config = get_config()
        
        assert config.persist_directory == test_path.resolve()
    
    def test_config_singleton(self, monkeypatch):
        """Test config is singleton."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        
        reload_config()
        config1 = get_config()
        config2 = get_config()
        
        assert config1 is config2
    
    def test_config_enum_validation(self, monkeypatch):
        """Test config validates enum values."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("LOG_LEVEL", "INVALID_LEVEL")
        
        reload_config()
        # Pydantic will raise ValidationError for invalid enum
        with pytest.raises((ConfigurationError, ValueError)):
            get_config()

