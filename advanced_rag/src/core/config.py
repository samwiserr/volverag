"""
Centralized configuration management using Pydantic Settings.

This module provides a single source of truth for all configuration,
replacing scattered os.getenv() calls throughout the codebase.
"""
from pathlib import Path
from typing import Optional, List, Annotated, Union
from enum import Enum
from pydantic import Field, field_validator, model_validator, BeforeValidator, AliasChoices
from pydantic_settings import BaseSettings, SettingsConfigDict
import os


class LogLevel(str, Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class EmbeddingModel(str, Enum):
    """OpenAI embedding models."""
    TEXT_EMBEDDING_3_SMALL = "text-embedding-3-small"
    TEXT_EMBEDDING_3_LARGE = "text-embedding-3-large"


class LLMModel(str, Enum):
    """OpenAI LLM models."""
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4_TURBO = "gpt-4-turbo"
    
    @classmethod
    def _missing_(cls, value):
        """Allow any string value, defaulting to GPT_4O if not in enum."""
        # For backward compatibility, accept any string and default to GPT_4O
        # But first try to match case-insensitively
        if isinstance(value, str):
            value_lower = value.lower()
            for member in cls:
                if member.value.lower() == value_lower:
                    return member
        return cls.GPT_4O


def _parse_llm_model(v: Union[str, LLMModel]) -> LLMModel:
    """Parse LLM model from string or enum, handling case-insensitive matching."""
    if isinstance(v, LLMModel):
        return v
    if isinstance(v, str):
        # Try exact match first
        for member in LLMModel:
            if member.value == v:
                return member
        # Try case-insensitive match
        v_lower = v.lower()
        for member in LLMModel:
            if member.value.lower() == v_lower:
                return member
        # Fallback to _missing_ handler
        return LLMModel._missing_(v)
    return LLMModel.GPT_4O


class AppConfig(BaseSettings):
    """
    Application configuration with validation.
    
    All configuration is loaded from environment variables with sensible defaults.
    Configuration is validated on load to catch errors early.
    """
    
    # API Keys
    openai_api_key: str = Field(
        ...,
        validation_alias=AliasChoices("OPENAI_API_KEY", "openai_api_key"),
        description="OpenAI API key (required)"
    )
    
    # Models
    embedding_model: EmbeddingModel = Field(
        default=EmbeddingModel.TEXT_EMBEDDING_3_SMALL,
        validation_alias=AliasChoices("EMBEDDING_MODEL", "embedding_model")
    )
    llm_model: Annotated[LLMModel, BeforeValidator(_parse_llm_model)] = Field(
        default=LLMModel.GPT_4O,
        validation_alias=AliasChoices("OPENAI_MODEL", "llm_model")
    )
    
    grade_model: Annotated[LLMModel, BeforeValidator(_parse_llm_model)] = Field(
        default=LLMModel.GPT_4O,
        validation_alias=AliasChoices("OPENAI_GRADE_MODEL", "grade_model")
    )
    
    # Paths
    persist_directory: Path = Field(
        default=Path("./data/vectorstore"),
        env="VECTORSTORE_PATH"
    )
    documents_path: Optional[Path] = Field(
        default=None,
        env="DOCUMENTS_PATH"
    )
    
    # Retrieval settings
    chunk_size: int = Field(
        default=500,
        ge=100,
        le=2000,
        env="CHUNK_SIZE",
        description="Target tokens per chunk"
    )
    chunk_overlap: int = Field(
        default=150,
        ge=0,
        # Note: le constraint removed - validation done in field_validator
        # to allow dynamic validation against chunk_size
        env="CHUNK_OVERLAP",
        description="Token overlap between chunks"
    )
    
    # Reranking
    use_cross_encoder: bool = Field(
        default=True,
        env="RAG_USE_CROSS_ENCODER"
    )
    cross_encoder_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        env="RAG_CROSS_ENCODER_MODEL"
    )
    mmr_enabled: bool = Field(
        default=True,
        env="RAG_MMR"
    )
    mmr_lambda: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        env="RAG_MMR_LAMBDA",
        description="MMR diversification parameter (0=relevance, 1=diversity)"
    )
    rerank_enabled: bool = Field(
        default=True,
        env="RAG_RERANK"
    )
    rerank_model: Annotated[LLMModel, BeforeValidator(_parse_llm_model)] = Field(
        default=LLMModel.GPT_4O,
        env="RAG_RERANK_MODEL"
    )
    
    # Fuzzy matching thresholds
    formation_fuzzy_threshold: float = Field(
        default=85.0,
        ge=0.0,
        le=100.0,
        env="FORMATION_FUZZY_THRESHOLD",
        description="Minimum similarity score (0-100) for formation fuzzy matching"
    )
    formation_fuzzy_margin: float = Field(
        default=10.0,
        ge=0.0,
        le=50.0,
        env="FORMATION_FUZZY_MARGIN",
        description="Minimum margin over second-best match to accept fuzzy match"
    )
    
    # Query processing
    enable_query_decomposition: bool = Field(
        default=True,
        env="RAG_ENABLE_QUERY_DECOMPOSITION"
    )
    enable_query_completion: bool = Field(
        default=True,
        env="RAG_ENABLE_QUERY_COMPLETION"
    )
    decomposition_model: Annotated[LLMModel, BeforeValidator(_parse_llm_model)] = Field(
        default=LLMModel.GPT_4O,
        env="RAG_DECOMPOSITION_MODEL"
    )
    
    # Entity resolution
    enable_entity_resolver: bool = Field(
        default=True,
        env="RAG_ENTITY_RESOLVER"
    )
    entity_resolver_model: LLMModel = Field(
        default=LLMModel.GPT_4O,
        env="RAG_ENTITY_RESOLVER_MODEL"
    )
    
    # Logging
    log_level: LogLevel = Field(
        default=LogLevel.INFO,
        env="LOG_LEVEL"
    )
    log_format: str = Field(
        default="text",
        env="LOG_FORMAT",
        description="'json' or 'text'"
    )
    
    # Rate limiting
    max_requests_per_minute: int = Field(
        default=60,
        ge=1,
        env="MAX_REQUESTS_PER_MINUTE"
    )
    
    # Caching
    enable_llm_cache: bool = Field(
        default=True,
        env="ENABLE_LLM_CACHE"
    )
    cache_ttl_seconds: int = Field(
        default=3600,
        ge=0,
        env="CACHE_TTL_SECONDS"
    )
    
    # External URLs (for Streamlit Cloud)
    vectorstore_url: Optional[str] = Field(
        default=None,
        env="VECTORSTORE_URL"
    )
    pdfs_url: Optional[str] = Field(
        default=None,
        env="PDFS_URL"
    )
    
    @field_validator("persist_directory", "documents_path", mode="before")
    @classmethod
    def validate_paths(cls, v):
        """Ensure paths are resolved to absolute paths."""
        if v is None:
            return None
        path = Path(v)
        if path.is_absolute():
            return path
        # Try to resolve relative to current working directory
        resolved = Path.cwd() / path
        if resolved.exists():
            return resolved.resolve()
        return path.resolve() if path.exists() else path
    
    @model_validator(mode='before')
    @classmethod
    def validate_llm_models_from_env(cls, values):
        """Parse LLM model enums from environment variables before validation."""
        if isinstance(values, dict):
            # Handle LLM model fields that might come as strings from env
            llm_fields = ['llm_model', 'grade_model', 'rerank_model', 'decomposition_model', 'entity_resolver_model']
            for field in llm_fields:
                if field in values and isinstance(values[field], str):
                    # Try exact match first
                    for member in LLMModel:
                        if member.value == values[field]:
                            values[field] = member
                            break
                    else:
                        # Try case-insensitive match
                        v_lower = values[field].lower()
                        for member in LLMModel:
                            if member.value.lower() == v_lower:
                                values[field] = member
                                break
                        else:
                            # Fallback to _missing_ handler
                            values[field] = LLMModel._missing_(values[field])
        return values
    
    @model_validator(mode='after')
    def validate_overlap(self):
        """Overlap must be less than chunk size."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return self
    
    @field_validator("log_format")
    @classmethod
    def validate_log_format(cls, v):
        """Log format must be 'json' or 'text'."""
        if v not in ["json", "text"]:
            raise ValueError("log_format must be 'json' or 'text'")
        return v
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        validate_assignment=True,
        extra="ignore",  # Ignore extra fields from environment (e.g., gemini_api_key)
    )


# Singleton instance
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """
    Get application configuration (singleton).
    
    Configuration is loaded once and cached. For testing, use reload_config().
    
    Returns:
        AppConfig instance
        
    Raises:
        ValidationError: If configuration is invalid
    """
    global _config
    if _config is None:
        # Try to load from Streamlit secrets if available
        try:
            import streamlit as st
            # Check if we're in a Streamlit runtime context
            # st.secrets will raise RuntimeError if not in Streamlit context
            try:
                secrets = st.secrets
                # Only access secrets if they exist (avoids StreamlitSecretNotFoundError)
                if hasattr(secrets, '_secrets') and secrets._secrets:
                    # Merge Streamlit secrets into environment
                    if "OPENAI_API_KEY" in secrets:
                        os.environ.setdefault("OPENAI_API_KEY", str(secrets["OPENAI_API_KEY"]))
                    if "VECTORSTORE_URL" in secrets:
                        os.environ.setdefault("VECTORSTORE_URL", str(secrets["VECTORSTORE_URL"]))
                    if "PDFS_URL" in secrets:
                        os.environ.setdefault("PDFS_URL", str(secrets["PDFS_URL"]))
            except (RuntimeError, AttributeError, KeyError):
                # Not in Streamlit context, or secrets not available
                pass
        except (ImportError, AttributeError):
            # Streamlit not installed or not available
            pass
        
        try:
            _config = AppConfig()
        except Exception as e:
            # If validation fails, provide helpful error message
            from .exceptions import ConfigurationError
            raise ConfigurationError(
                f"Configuration validation failed: {e}. "
                f"Please check your environment variables and .env file."
            ) from e
    return _config


def reload_config() -> AppConfig:
    """
    Reload configuration (useful for testing).
    
    Returns:
        New AppConfig instance
        
    Raises:
        ConfigurationError: If configuration validation fails
    """
    global _config
    _config = None
    try:
        return get_config()
    except ConfigurationError:
        # Ensure _config is None on validation failure to prevent bad state
        _config = None
        raise


def reset_config():
    """
    Reset configuration singleton (useful for testing).
    
    This clears the cached configuration without attempting to reload it.
    Use this in test teardown to ensure clean state.
    """
    global _config
    _config = None

