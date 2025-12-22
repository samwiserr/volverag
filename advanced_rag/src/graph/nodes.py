"""
LangGraph nodes for agentic RAG workflow.
Following the pattern from: https://docs.langchain.com/oss/python/langgraph/agentic-rag
"""
import os
import re
import logging
from pathlib import Path
from typing import Literal, Optional, List
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from ..normalize.query_normalizer import normalize_query
from ..normalize.property_registry import default_registry, resolve_property_deterministic, PropertyEntry
from ..normalize.agent_disambiguator import choose_property_with_agent
from ..normalize.query_normalizer import extract_well, normalize_formation
from ..normalize.entity_resolver import resolve_with_bounded_agent
from ..query.incomplete_query_handler import is_incomplete_query, complete_incomplete_query
from ..query.query_decomposer import decompose_query, expand_query_synonyms
from .generation.answer import generate_answer
from .generation.rewriter import rewrite_question
from .routing.router import generate_query_or_respond
from .retrieval.grader import grade_documents
from ..core.container import get_container

logger = logging.getLogger(__name__)

# Lazy-init chat models (avoid import-time crash if OPENAI_API_KEY not set)
# DEPRECATED: These globals will be removed in favor of DI container
_response_model: Optional[ChatOpenAI] = None
_grader_model: Optional[ChatOpenAI] = None


# Service keys for DI container
SERVICE_KEY_RESPONSE_MODEL = "response_model"
SERVICE_KEY_GRADER_MODEL = "grader_model"
SERVICE_KEY_REGISTRY = "property_registry"


def _get_response_model() -> ChatOpenAI:
    """
    Get response model (ChatOpenAI instance).
    
    Priority:
    1. DI container (if registered with key 'response_model')
    2. Global singleton (fallback for backward compatibility)
    
    DEPRECATED: Use _get_response_model_from_container() for new code.
    """
    # Try DI container first
    try:
        container = get_container()
        if container.is_registered(ChatOpenAI, key=SERVICE_KEY_RESPONSE_MODEL):
            model = container.get(ChatOpenAI, key=SERVICE_KEY_RESPONSE_MODEL)
            if model is not None:
                return model
    except Exception:
        pass  # Fallback to global
    
    # Fallback to global singleton (backward compatibility)
    global _response_model
    if _response_model is None:
        model = os.getenv("OPENAI_MODEL", "gpt-4o")
        _response_model = ChatOpenAI(model=model, temperature=0)
    return _response_model


def _get_grader_model() -> ChatOpenAI:
    """
    Get grader model (ChatOpenAI instance).
    
    Priority:
    1. DI container (if registered with key 'grader_model')
    2. Global singleton (fallback for backward compatibility)
    
    DEPRECATED: Use _get_grader_model_from_container() for new code.
    """
    # Try DI container first
    try:
        container = get_container()
        if container.is_registered(ChatOpenAI, key=SERVICE_KEY_GRADER_MODEL):
            model = container.get(ChatOpenAI, key=SERVICE_KEY_GRADER_MODEL)
            if model is not None:
                return model
    except Exception:
        pass  # Fallback to global
    
    # Fallback to global singleton (backward compatibility)
    global _grader_model
    if _grader_model is None:
        model = os.getenv("OPENAI_GRADE_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o"))
        _grader_model = ChatOpenAI(model=model, temperature=0)
    return _grader_model

# Initialize property registry (lazy-loaded)
# DEPRECATED: This global will be removed in favor of DI container
_registry_cache: Optional[List[PropertyEntry]] = None

def _get_registry() -> List[PropertyEntry]:
    """
    Lazy-load the property registry.
    
    Priority:
    1. DI container (if registered with key 'property_registry')
    2. Global singleton (fallback for backward compatibility)
    
    DEPRECATED: Use _get_registry_from_container() for new code.
    """
    # Try DI container first
    try:
        container = get_container()
        registry = container.get_by_key_or_none(SERVICE_KEY_REGISTRY)
        if registry is not None:
            return registry
    except Exception:
        pass  # Fallback to global
    
    # Fallback to global singleton (backward compatibility)
    global _registry_cache
    if _registry_cache is None:
        _registry_cache = default_registry("./data/vectorstore")
    return _registry_cache


def _latest_user_question(messages) -> str:
    """
    Multi-turn support: use the most recent HumanMessage as the "current question".
    """
    try:
        from langchain_core.messages import HumanMessage

        for m in reversed(messages or []):
            if isinstance(m, HumanMessage) and isinstance(getattr(m, "content", None), str):
                return m.content
    except Exception:
        pass
    # Fallback (single-turn behavior)
    try:
        if messages and isinstance(messages[-1].get("content"), str):
            return messages[-1]["content"]
    except Exception:
        pass
    try:
        return messages[0].content if messages else ""
    except Exception:
        return ""


def _iter_message_texts(messages):
    """
    Yield (role, content) for both dict-style and LangChain message objects.
    """
    for m in messages or []:
        # dict-style
        if isinstance(m, dict):
            role = m.get("role")
            content = m.get("content")
            if isinstance(content, str):
                yield role, content
            continue
        # LangChain message objects
        role = getattr(m, "type", None) or getattr(m, "__class__", type("x", (), {})).__name__
        content = getattr(m, "content", None)
        if isinstance(content, str):
            yield str(role), content


def _infer_recent_context(messages) -> tuple[Optional[str], Optional[str]]:
    """
    Infer (well, formation) from recent conversation history.
    This enables follow-ups like "matrix density" after a prior turn specifying well/formation.
    """
    well = None
    formation = None
    # Scan backward through recent messages
    for role, txt in reversed(list(_iter_message_texts(messages))[-25:]):
        # Ignore tool messages (they may contain many formations/wells from retrieved context)
        if str(role).lower() in {"tool", "toolmessage"}:
            continue
        if well is None:
            w = extract_well(txt)
            if w:
                well = w
        if formation is None:
            f = normalize_formation(txt)
            if f:
                formation = f
        if well and formation:
            break
    return well, formation


# grade_documents is now imported from .retrieval.grader
# rewrite_question is now imported from .generation.rewriter
# generate_query_or_respond is now imported from .routing.router
# generate_answer is now imported from .generation.answer
    
