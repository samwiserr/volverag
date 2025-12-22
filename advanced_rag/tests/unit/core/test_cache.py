"""
Unit tests for cache module.
"""
import pytest
import time
from src.core.cache import Cache, CacheEntry, generate_cache_key, cached, get_llm_cache, get_embedding_cache


@pytest.mark.unit
class TestCacheEntry:
    """Test CacheEntry dataclass."""
    
    def test_is_expired_false_when_not_expired(self):
        """Test is_expired returns False for non-expired entry."""
        entry = CacheEntry(
            value="test",
            expires_at=time.time() + 3600  # 1 hour from now
        )
        assert not entry.is_expired()
    
    def test_is_expired_true_when_expired(self):
        """Test is_expired returns True for expired entry."""
        entry = CacheEntry(
            value="test",
            expires_at=time.time() - 1  # 1 second ago
        )
        assert entry.is_expired()
    
    def test_touch_updates_statistics(self):
        """Test touch updates access count and last_accessed."""
        entry = CacheEntry(
            value="test",
            expires_at=time.time() + 3600
        )
        initial_count = entry.access_count
        initial_access = entry.last_accessed
        
        time.sleep(0.01)  # Small delay
        entry.touch()
        
        assert entry.access_count == initial_count + 1
        assert entry.last_accessed > initial_access


@pytest.mark.unit
class TestCache:
    """Test Cache class."""
    
    def test_get_returns_none_for_missing_key(self):
        """Test get returns None for non-existent key."""
        cache = Cache()
        assert cache.get("nonexistent") is None
    
    def test_set_and_get(self):
        """Test basic set and get operations."""
        cache = Cache()
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
    
    def test_get_returns_none_for_expired_entry(self):
        """Test get returns None for expired entry."""
        cache = Cache(default_ttl=1)
        cache.set("key1", "value1", ttl=0.01)  # Very short TTL
        assert cache.get("key1") == "value1"  # Should still be valid
        
        time.sleep(0.02)  # Wait for expiration
        assert cache.get("key1") is None  # Should be expired
    
    def test_set_with_custom_ttl(self):
        """Test set with custom TTL."""
        cache = Cache(default_ttl=3600)
        cache.set("key1", "value1", ttl=1)
        
        assert cache.get("key1") == "value1"
        time.sleep(1.1)
        assert cache.get("key1") is None
    
    def test_delete(self):
        """Test delete removes entry."""
        cache = Cache()
        cache.set("key1", "value1")
        cache.delete("key1")
        assert cache.get("key1") is None
    
    def test_delete_nonexistent_key(self):
        """Test delete on non-existent key doesn't raise."""
        cache = Cache()
        cache.delete("nonexistent")  # Should not raise
    
    def test_clear(self):
        """Test clear removes all entries."""
        cache = Cache()
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.clear()
        assert cache.get("key1") is None
        assert cache.get("key2") is None
    
    def test_stats(self):
        """Test stats returns cache statistics."""
        cache = Cache()
        assert cache.stats()["total_entries"] == 0
        cache.set("key1", "value1")
        stats = cache.stats()
        assert stats["total_entries"] == 1
        assert stats["active_entries"] == 1
    
    def test_cleanup_expired(self):
        """Test cleanup_expired removes expired entries."""
        cache = Cache(default_ttl=0.01)  # Very short TTL
        cache.set("key1", "value1")
        cache.set("key2", "value2", ttl=3600)  # Long TTL
        
        time.sleep(0.02)  # Wait for key1 to expire
        removed = cache.cleanup_expired()
        assert removed >= 1
        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"  # Should still be valid


@pytest.mark.unit
class TestGenerateCacheKey:
    """Test generate_cache_key function."""
    
    def test_generates_deterministic_key(self):
        """Test generates same key for same inputs."""
        key1 = generate_cache_key("func", "arg1", "arg2", kwarg1="val1")
        key2 = generate_cache_key("func", "arg1", "arg2", kwarg1="val1")
        assert key1 == key2
    
    def test_generates_different_keys_for_different_args(self):
        """Test generates different keys for different arguments."""
        key1 = generate_cache_key("func", "arg1")
        key2 = generate_cache_key("func", "arg2")
        assert key1 != key2
    
    def test_handles_kwargs(self):
        """Test handles keyword arguments."""
        key1 = generate_cache_key("func", kwarg1="val1", kwarg2="val2")
        key2 = generate_cache_key("func", kwarg2="val2", kwarg1="val1")  # Different order
        # Should be same (kwargs are sorted)
        assert key1 == key2
    
    def test_handles_complex_types(self):
        """Test handles complex types in arguments."""
        key1 = generate_cache_key("func", {"key": "value"}, [1, 2, 3])
        key2 = generate_cache_key("func", {"key": "value"}, [1, 2, 3])
        assert key1 == key2


@pytest.mark.unit
class TestCachedDecorator:
    """Test cached decorator."""
    
    def test_caches_function_result(self, monkeypatch):
        """Test decorator caches function results."""
        # Enable caching for this test
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("ENABLE_LLM_CACHE", "true")
        from src.core.config import reset_config, reload_config
        reset_config()
        reload_config()
        
        call_count = 0
        
        @cached(ttl=3600)
        def expensive_func(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2
        
        result1 = expensive_func(5)
        result2 = expensive_func(5)  # Should use cache
        
        assert result1 == 10
        assert result2 == 10
        assert call_count == 1  # Function only called once
    
    def test_cache_expires_after_ttl(self, monkeypatch):
        """Test cache expires after TTL."""
        # Enable caching for this test
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("ENABLE_LLM_CACHE", "true")
        from src.core.config import reset_config, reload_config
        reset_config()
        reload_config()
        
        call_count = 0
        
        @cached(ttl=0.01)  # Very short TTL
        def func(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2
        
        func(5)  # First call
        func(5)  # Should use cache
        assert call_count == 1
        
        time.sleep(0.02)  # Wait for expiration
        func(5)  # Should call again
        assert call_count == 2
    
    def test_different_args_create_different_cache_entries(self, monkeypatch):
        """Test different arguments create different cache entries."""
        # Enable caching for this test
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("ENABLE_LLM_CACHE", "true")
        from src.core.config import reset_config, reload_config
        reset_config()
        reload_config()
        
        call_count = 0
        
        @cached(ttl=3600)
        def func(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2
        
        func(5)
        func(10)  # Different arg, should call again
        assert call_count == 2


@pytest.mark.unit
class TestCacheSingletons:
    """Test cache singleton functions."""
    
    def test_get_llm_cache_returns_singleton(self, monkeypatch):
        """Test get_llm_cache returns same instance."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        from src.core.config import reset_config, reload_config
        reset_config()
        reload_config()
        
        cache1 = get_llm_cache()
        cache2 = get_llm_cache()
        assert cache1 is cache2
    
    def test_get_embedding_cache_returns_singleton(self):
        """Test get_embedding_cache returns same instance."""
        cache1 = get_embedding_cache()
        cache2 = get_embedding_cache()
        assert cache1 is cache2
    
    def test_llm_and_embedding_caches_are_different(self, monkeypatch):
        """Test LLM and embedding caches are different instances."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        from src.core.config import reset_config, reload_config
        reset_config()
        reload_config()
        
        llm_cache = get_llm_cache()
        embedding_cache = get_embedding_cache()
        assert llm_cache is not embedding_cache

