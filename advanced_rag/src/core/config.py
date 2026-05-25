"""
Centralized configuration management using Pydantic Settings.

This module provides a single source of truth for all configuration,
replacing scattered os.getenv() calls throughout the codebase.
"""
from pathlib import Path
from typing import Optional, List, Union
from enum import Enum
from pydantic import Field, field_validator, model_validator, AliasChoices
from pydantic_settings import BaseSettings, SettingsConfigDict
from .exceptions import ConfigurationError
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
    openai_api_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("OPENAI_API_KEY", "openai_api_key"),
        description="OpenAI API key (required only when using OpenAI models or embeddings)"
    )
    groq_api_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("GROQ_API_KEY", "groq_api_key"),
        description="Groq API key (required when LLM_PROVIDER=groq)"
    )
    
    # Models
    llm_provider: str = Field(
        default="groq",
        validation_alias=AliasChoices("LLM_PROVIDER", "llm_provider"),
        description="'groq' or 'openai'"
    )
    embedding_provider: str = Field(
        default="huggingface",
        validation_alias=AliasChoices("EMBEDDING_PROVIDER", "embedding_provider"),
        description="'huggingface'/'local' or 'openai'"
    )
    embedding_model: str = Field(
        default="nomic-ai/nomic-embed-text-v1.5",
        validation_alias=AliasChoices("EMBEDDING_MODEL", "OPENAI_EMBEDDING_MODEL", "LOCAL_EMBEDDING_MODEL", "embedding_model")
    )
    llm_model: str = Field(
        default="llama-3.3-70b-versatile",
        validation_alias=AliasChoices("OPENAI_MODEL", "GROQ_MODEL", "llm_model")
    )
    
    grade_model: str = Field(
        default="llama-3.3-70b-versatile",
        validation_alias=AliasChoices("OPENAI_GRADE_MODEL", "GROQ_MODEL", "grade_model")
    )
    
    # Paths
    persist_directory: Path = Field(
        default=Path("./data/vectorstore"),
        validation_alias=AliasChoices("VECTORSTORE_PATH", "persist_directory")
    )
    documents_path: Optional[Path] = Field(
        default=None,
        validation_alias=AliasChoices("DOCUMENTS_PATH", "documents_path")
    )
    
    # Retrieval settings
    chunk_size: int = Field(
        default=500,
        ge=100,
        le=2000,
        validation_alias=AliasChoices("CHUNK_SIZE", "chunk_size"),
        description="Target tokens per chunk"
    )
    chunk_overlap: int = Field(
        default=150,
        ge=0,
        # Note: le constraint removed - validation done in field_validator
        # to allow dynamic validation against chunk_size
        validation_alias=AliasChoices("CHUNK_OVERLAP", "chunk_overlap"),
        description="Token overlap between chunks"
    )
    
    # Reranking
    use_cross_encoder: bool = Field(
        default=True,
        validation_alias=AliasChoices("RAG_USE_CROSS_ENCODER", "use_cross_encoder")
    )
    cross_encoder_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        validation_alias=AliasChoices("RAG_CROSS_ENCODER_MODEL", "cross_encoder_model")
    )
    mmr_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("RAG_MMR", "mmr_enabled")
    )
    mmr_lambda: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        validation_alias=AliasChoices("RAG_MMR_LAMBDA", "mmr_lambda"),
        description="MMR diversification parameter (0=relevance, 1=diversity)"
    )
    rerank_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("RAG_RERANK", "rerank_enabled")
    )
    rerank_model: str = Field(
        default="llama-3.3-70b-versatile",
        validation_alias=AliasChoices("RAG_RERANK_MODEL", "rerank_model")
    )

    # ── SOTA techniques (2024-2026) ─────────────────────────────────────────
    # Contextual Chunking — Anthropic Contextual Retrieval (Sept 2024)
    # Runs at index-build time only; baked into the vectorstore.
    contextual_chunking: bool = Field(
        default=True,
        validation_alias=AliasChoices("RAG_CONTEXTUAL", "contextual_chunking"),
        description="Prepend LLM-generated situating context to each chunk before embedding",
    )
    context_model: str = Field(
        default="llama-3.1-8b-instant",
        validation_alias=AliasChoices("RAG_CONTEXT_MODEL", "GROQ_FAST_MODEL", "context_model"),
        description="Model used to generate contextual chunk prefixes (build-time)",
    )
    # HyDE — Hypothetical Document Embeddings (Gao et al., SIGIR 2023)
    # Runs at query time; can be toggled without rebuilding the index.
    hyde_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("RAG_HYDE", "hyde_enabled"),
        description="Generate a hypothetical answer doc and embed it alongside the raw query",
    )
    hyde_model: str = Field(
        default="llama-3.1-8b-instant",
        validation_alias=AliasChoices("RAG_HYDE_MODEL", "GROQ_FAST_MODEL", "hyde_model"),
        description="Model used to generate hypothetical documents (query-time)",
    )
    hyde_n_hypotheses: int = Field(
        default=1,
        ge=1,
        le=5,
        validation_alias=AliasChoices("RAG_HYDE_N", "hyde_n_hypotheses"),
        description="Number of independent hypothetical documents to generate per query",
    )
    # RAPTOR — Recursive Abstractive Processing for Tree-Organised Retrieval
    # (Sarthi et al., ICLR 2024). Runs at index-build time only.
    raptor_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("RAG_RAPTOR", "raptor_enabled"),
        description="Build a multi-level summary tree over leaf chunks at index time",
    )
    raptor_model: str = Field(
        default="llama-3.1-8b-instant",
        validation_alias=AliasChoices("RAG_RAPTOR_MODEL", "GROQ_FAST_MODEL", "raptor_model"),
        description="Model used to summarise RAPTOR cluster nodes (build-time)",
    )
    raptor_levels: int = Field(
        default=2,
        ge=1,
        le=5,
        validation_alias=AliasChoices("RAG_RAPTOR_LEVELS", "raptor_levels"),
        description="Maximum depth of the RAPTOR summary tree",
    )
    raptor_clusters: int = Field(
        default=8,
        ge=2,
        le=50,
        validation_alias=AliasChoices("RAG_RAPTOR_CLUSTERS", "raptor_clusters"),
        description="Target number of clusters per RAPTOR level",
    )
    # Hybrid fusion strategy
    hybrid_fusion: str = Field(
        default="rrf",
        validation_alias=AliasChoices("RAG_HYBRID_FUSION", "hybrid_fusion"),
        description="'rrf' (Reciprocal Rank Fusion) or 'weighted' score merge",
    )
    rrf_k: int = Field(
        default=60,
        ge=1,
        validation_alias=AliasChoices("RAG_RRF_K", "rrf_k"),
        description="RRF smoothing constant k (higher = less rank-sensitive)",
    )
    # ────────────────────────────────────────────────────────────────────────
    
    # Fuzzy matching thresholds
    formation_fuzzy_threshold: float = Field(
        default=85.0,
        ge=0.0,
        le=100.0,
        validation_alias=AliasChoices("FORMATION_FUZZY_THRESHOLD", "formation_fuzzy_threshold"),
        description="Minimum similarity score (0-100) for formation fuzzy matching"
    )
    formation_fuzzy_margin: float = Field(
        default=10.0,
        ge=0.0,
        le=50.0,
        validation_alias=AliasChoices("FORMATION_FUZZY_MARGIN", "formation_fuzzy_margin"),
        description="Minimum margin over second-best match to accept fuzzy match"
    )
    
    # Query processing
    enable_query_decomposition: bool = Field(
        default=True,
        validation_alias=AliasChoices("RAG_ENABLE_QUERY_DECOMPOSITION", "enable_query_decomposition")
    )
    enable_query_completion: bool = Field(
        default=True,
        validation_alias=AliasChoices("RAG_ENABLE_QUERY_COMPLETION", "enable_query_completion")
    )
    decomposition_model: str = Field(
        default="llama-3.3-70b-versatile",
        validation_alias=AliasChoices("RAG_DECOMPOSITION_MODEL", "OPENAI_MODEL", "GROQ_MODEL", "decomposition_model")
    )
    
    # Entity resolution
    enable_entity_resolver: bool = Field(
        default=True,
        validation_alias=AliasChoices("RAG_ENTITY_RESOLVER", "enable_entity_resolver")
    )
    entity_resolver_model: str = Field(
        default="llama-3.3-70b-versatile",
        validation_alias=AliasChoices("RAG_ENTITY_RESOLVER_MODEL", "OPENAI_MODEL", "GROQ_MODEL", "entity_resolver_model")
    )
    
    # Logging
    log_level: LogLevel = Field(
        default=LogLevel.INFO,
        validation_alias=AliasChoices("LOG_LEVEL", "log_level")
    )
    log_format: str = Field(
        default="text",
        validation_alias=AliasChoices("LOG_FORMAT", "log_format"),
        description="'json' or 'text'"
    )
    
    # Rate limiting
    max_requests_per_minute: int = Field(
        default=60,
        ge=1,
        validation_alias=AliasChoices("MAX_REQUESTS_PER_MINUTE", "max_requests_per_minute")
    )
    
    # Resource limits
    max_memory_mb: int = Field(
        default=2048,
        ge=256,
        validation_alias=AliasChoices("MAX_MEMORY_MB", "max_memory_mb"),
        description="Maximum memory usage in MB"
    )
    llm_timeout_seconds: int = Field(
        default=120,
        ge=10,
        le=600,
        validation_alias=AliasChoices("LLM_TIMEOUT_SECONDS", "llm_timeout_seconds"),
        description="LLM call timeout in seconds"
    )
    retrieval_timeout_seconds: int = Field(
        default=30,
        ge=5,
        le=120,
        validation_alias=AliasChoices("RETRIEVAL_TIMEOUT_SECONDS", "retrieval_timeout_seconds"),
        description="Retrieval operation timeout in seconds"
    )
    max_concurrent_requests: int = Field(
        default=10,
        ge=1,
        le=100,
        validation_alias=AliasChoices("MAX_CONCURRENT_REQUESTS", "max_concurrent_requests"),
        description="Maximum concurrent requests"
    )
    max_request_body_size_mb: int = Field(
        default=10,
        ge=1,
        le=100,
        validation_alias=AliasChoices("MAX_REQUEST_BODY_SIZE_MB", "max_request_body_size_mb"),
        description="Maximum request body size in MB"
    )
    
    # Caching
    enable_llm_cache: bool = Field(
        default=True,
        validation_alias=AliasChoices("ENABLE_LLM_CACHE", "enable_llm_cache")
    )
    cache_ttl_seconds: int = Field(
        default=3600,
        ge=0,
        validation_alias=AliasChoices("CACHE_TTL_SECONDS", "cache_ttl_seconds")
    )
    
    # External URLs (for Streamlit Cloud)
    vectorstore_url: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("VECTORSTORE_URL", "vectorstore_url")
    )
    pdfs_url: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("PDFS_URL", "pdfs_url")
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
    
    @model_validator(mode='after')
    def validate_overlap(self):
        """Overlap must be less than chunk size."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        if self.llm_provider.lower() == "groq" and not self.groq_api_key:
            raise ValueError("GROQ_API_KEY is required when LLM_PROVIDER=groq")
        if self.llm_provider.lower() == "openai" and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        if self.embedding_provider.lower() == "openai" and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai")
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
                    if "GROQ_API_KEY" in secrets:
                        os.environ.setdefault("GROQ_API_KEY", str(secrets["GROQ_API_KEY"]))
                    if "OPENAI_API_KEY" in secrets:
                        os.environ.setdefault("OPENAI_API_KEY", str(secrets["OPENAI_API_KEY"]))
                    if "LLM_PROVIDER" in secrets:
                        os.environ.setdefault("LLM_PROVIDER", str(secrets["LLM_PROVIDER"]))
                    if "EMBEDDING_PROVIDER" in secrets:
                        os.environ.setdefault("EMBEDDING_PROVIDER", str(secrets["EMBEDDING_PROVIDER"]))
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

