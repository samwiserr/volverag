"""
Model factory for chat models and embeddings.

Defaults are tuned for a no-paid-LLM deployment:
- Chat/generation: Groq (`llama-3.3-70b-versatile`)
- Fast helper tasks: Groq (`llama-3.1-8b-instant`)
- Embeddings: local Hugging Face (`nomic-ai/nomic-embed-text-v1.5`)

OpenAI remains supported by setting:
    LLM_PROVIDER=openai
    EMBEDDING_PROVIDER=openai
"""
from __future__ import annotations

import os
from typing import Any, Optional

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

try:
    from langchain_groq import ChatGroq
except ImportError:  # pragma: no cover - dependency controlled by requirements
    ChatGroq = None  # type: ignore

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:  # pragma: no cover - fallback for older LangChain installs
    from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore


DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_GROQ_FAST_MODEL = "llama-3.1-8b-instant"
DEFAULT_LOCAL_EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"


def _provider(name: str, default: str) -> str:
    return (os.getenv(name, default) or default).strip().lower()


def _is_openai_model(model: Optional[str]) -> bool:
    if not model:
        return False
    return model.startswith(("gpt-", "o1", "o3", "o4"))


def get_chat_model(
    model: Optional[str] = None,
    *,
    temperature: float = 0,
    max_tokens: Optional[int] = None,
    role: str = "default",
) -> Any:
    """Create a chat model using the configured provider."""
    provider = _provider("LLM_PROVIDER", "groq")
    kwargs: dict[str, Any] = {"temperature": temperature}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    if provider == "groq":
        if ChatGroq is None:
            raise ImportError("langchain-groq is required when LLM_PROVIDER=groq")
        if not os.getenv("GROQ_API_KEY"):
            raise ValueError("GROQ_API_KEY environment variable not set")

        if model and not _is_openai_model(model):
            selected = model
        elif role in {"fast", "context", "hyde", "raptor"}:
            selected = os.getenv("GROQ_FAST_MODEL", DEFAULT_GROQ_FAST_MODEL)
        else:
            selected = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)
        return ChatGroq(model=selected, **kwargs)

    if provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable not set")
        selected = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        return ChatOpenAI(model=selected, **kwargs)

    raise ValueError(f"Unsupported LLM_PROVIDER='{provider}'")


def get_embedding_model(model: Optional[str] = None) -> Any:
    """Create an embedding model using the configured provider."""
    provider = _provider("EMBEDDING_PROVIDER", "huggingface")

    if provider in {"huggingface", "hf", "local"}:
        selected = model or os.getenv("LOCAL_EMBEDDING_MODEL", DEFAULT_LOCAL_EMBEDDING_MODEL)
        if selected.startswith("text-embedding-"):
            selected = os.getenv("LOCAL_EMBEDDING_MODEL", DEFAULT_LOCAL_EMBEDDING_MODEL)

        model_kwargs: dict[str, Any] = {
            "device": os.getenv("LOCAL_EMBEDDING_DEVICE", "cpu"),
        }
        if "nomic-ai/" in selected:
            model_kwargs["trust_remote_code"] = True

        return HuggingFaceEmbeddings(
            model_name=selected,
            model_kwargs=model_kwargs,
            encode_kwargs={"normalize_embeddings": True},
        )

    if provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable not set")
        selected = model or os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        return OpenAIEmbeddings(model=selected)

    raise ValueError(f"Unsupported EMBEDDING_PROVIDER='{provider}'")
