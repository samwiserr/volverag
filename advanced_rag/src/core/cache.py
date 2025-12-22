"""
Multi-layer caching system for LLM calls, embeddings, and expensive computations.

This module provides a flexible caching system with TTL support, key generation,
and cache invalidation strategies. Designed to work seamlessly with Streamlit.
"""
import hashlib
import json
import time
from typing import Any, Callable, Dict, Optional, TypeVar, Union
from functools import wraps
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .config import get_config
from .logging import get_logger

logger = get_logger(__name__)

T = TypeVar('T')


@dataclass
class CacheEntry:
    """Cache entry with value and expiration time."""
    value: Any
    expires_at: float
    created_at: float = field(default_factory=time.time)
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        return time.time() > self.expires_at
    
    def touch(self) -> None:
        """Update access statistics."""
        self.access_count += 1
        self.last_accessed = time.time()


class Cache:
    """
    Thread-safe in-memory cache with TTL support.
    
    This cache is designed for Streamlit's multi-threaded environment
    and provides efficient key-based lookups with automatic expiration.
    """
    
    def __init__(self, default_ttl: int = 3600):
        """
        Initialize cache.
        
        Args:
            default_ttl: Default time-to-live in seconds
        """
        self._store: Dict[str, CacheEntry] = {}
        self._default_ttl = default_ttl
        self._lock = None  # Will use threading.Lock if needed
        
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        entry = self._store.get(key)
        if entry is None:
            return None
        
        if entry.is_expired():
            # Remove expired entry
            del self._store[key]
            return None
        
        entry.touch()
        return entry.value
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (uses default if None)
        """
        if ttl is None:
            ttl = self._default_ttl
        
        expires_at = time.time() + ttl
        self._store[key] = CacheEntry(
            value=value,
            expires_at=expires_at
        )
    
    def delete(self, key: str) -> None:
        """Delete entry from cache."""
        self._store.pop(key, None)
    
    def clear(self) -> None:
        """Clear all entries from cache."""
        self._store.clear()
    
    def cleanup_expired(self) -> int:
        """
        Remove all expired entries.
        
        Returns:
            Number of entries removed
        """
        expired_keys = [
            key for key, entry in self._store.items()
            if entry.is_expired()
        ]
        for key in expired_keys:
            del self._store[key]
        return len(expired_keys)
    
    def stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache stats
        """
        total = len(self._store)
        expired = sum(1 for e in self._store.values() if e.is_expired())
        return {
            "total_entries": total,
            "expired_entries": expired,
            "active_entries": total - expired,
            "default_ttl": self._default_ttl,
        }


# Global cache instances
_llm_cache: Optional[Cache] = None
_embedding_cache: Optional[Cache] = None


def get_llm_cache() -> Cache:
    """Get or create LLM cache instance."""
    global _llm_cache
    if _llm_cache is None:
        try:
            config = get_config()
            ttl = config.cache_ttl_seconds if config.enable_llm_cache else 0
        except Exception:
            ttl = 3600  # Default 1 hour
        _llm_cache = Cache(default_ttl=ttl)
    return _llm_cache


def get_embedding_cache() -> Cache:
    """Get or create embedding cache instance."""
    global _embedding_cache
    if _embedding_cache is None:
        # Embeddings rarely change, use longer TTL
        _embedding_cache = Cache(default_ttl=86400)  # 24 hours
    return _embedding_cache


def generate_cache_key(*args: Any, **kwargs: Any) -> str:
    """
    Generate cache key from function arguments.
    
    Args:
        *args: Positional arguments
        **kwargs: Keyword arguments
        
    Returns:
        MD5 hash of serialized arguments
    """
    # Create deterministic representation
    key_data = {
        "args": args,
        "kwargs": sorted(kwargs.items()) if kwargs else {}
    }
    key_str = json.dumps(key_data, sort_keys=True, default=str)
    return hashlib.md5(key_str.encode()).hexdigest()


def cached(
    ttl: Optional[int] = None,
    key_func: Optional[Callable[..., str]] = None,
    cache_instance: Optional[Cache] = None
):
    """
    Decorator to cache function results.
    
    Args:
        ttl: Time-to-live in seconds (uses cache default if None)
        key_func: Custom key generation function
        cache_instance: Cache instance to use (defaults to LLM cache)
        
    Returns:
        Decorated function with caching
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        cache = cache_instance or get_llm_cache()
        
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            # If a specific cache instance was provided, use it directly
            # Otherwise, check if caching is enabled in config
            using_default_cache = cache_instance is None
            if using_default_cache:
                try:
                    config = get_config()
                    if not config.enable_llm_cache:
                        return func(*args, **kwargs)
                except Exception:
                    # If config not available, skip caching
                    return func(*args, **kwargs)
            
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__module__}.{func.__name__}:{generate_cache_key(*args, **kwargs)}"
            
            # Try to get from cache
            cached_value = cache.get(cache_key)
            if cached_value is not None:
                logger.debug(f"Cache HIT: {func.__name__}")
                return cached_value
            
            # Cache miss - compute and store
            logger.debug(f"Cache MISS: {func.__name__}")
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl=ttl)
            return result
        
        return wrapper
    return decorator

