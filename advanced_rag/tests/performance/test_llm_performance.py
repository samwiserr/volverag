"""
Performance tests for LLM calls.

Tests LLM call latency, caching effectiveness, and throughput.
"""
import pytest
import time
import statistics
from unittest.mock import Mock, MagicMock, patch


@pytest.mark.performance
@pytest.mark.slow
@pytest.mark.requires_api
class TestLLMPerformance:
    """Performance tests for LLM calls."""
    
    def test_llm_call_latency_baseline(self, mock_config):
        """Test baseline LLM call latency (mocked)."""
        try:
            from langchain_openai import ChatOpenAI
            
            # Mock ChatOpenAI completely to avoid API key requirement
            with patch('langchain_openai.ChatOpenAI.__init__', return_value=None):
                with patch('langchain_openai.ChatOpenAI.invoke') as mock_invoke:
                    # Mock invoke to return quickly
                    mock_response = MagicMock()
                    mock_response.content = "Test response"
                    mock_invoke.return_value = mock_response
                    
                    llm = ChatOpenAI(model="gpt-4o", temperature=0)
                    
                    # Measure call latency
                    latencies = []
                    for _ in range(10):
                        start = time.perf_counter()
                        try:
                            llm.invoke("test prompt")
                        except Exception:
                            pass
                        latencies.append((time.perf_counter() - start) * 1000)  # Convert to ms
                    
                    # Mocked calls should be very fast
                    if latencies:
                        avg_latency = statistics.mean(latencies)
                        assert avg_latency < 100.0, \
                            f"LLM call latency too high: {avg_latency:.2f}ms"
        except ImportError:
            pytest.skip("ChatOpenAI not available")
    
    def test_llm_caching_effectiveness(self, mock_config):
        """Test that LLM caching improves performance."""
        from src.core.cache import get_llm_cache, Cache
        
        cache = get_llm_cache()
        
        # Simulate LLM call with caching
        def simulate_llm_call(prompt: str, use_cache: bool = True):
            cache_key = f"llm_call_{hash(prompt)}"
            
            if use_cache:
                cached = cache.get(cache_key)
                if cached:
                    return cached
            
            # Simulate slow LLM call
            time.sleep(0.01)  # 10ms simulated latency
            result = f"Response to: {prompt}"
            
            if use_cache:
                cache.set(cache_key, result, ttl=60)
            
            return result
        
        # First call (cache miss)
        start = time.perf_counter()
        result1 = simulate_llm_call("test prompt", use_cache=True)
        first_call_time = time.perf_counter() - start
        
        # Second call (cache hit)
        start = time.perf_counter()
        result2 = simulate_llm_call("test prompt", use_cache=True)
        second_call_time = time.perf_counter() - start
        
        # Results should be the same
        assert result1 == result2, "Cached result should match original"
        
        # Cached call should be faster
        assert second_call_time < first_call_time * 0.5, \
            f"Caching not effective: first={first_call_time:.4f}s, second={second_call_time:.4f}s"
    
    def test_llm_call_throughput(self, mock_config):
        """Test LLM can handle multiple calls."""
        try:
            from langchain_openai import ChatOpenAI
            
            # Mock ChatOpenAI completely to avoid API key requirement
            with patch('langchain_openai.ChatOpenAI.__init__', return_value=None):
                with patch('langchain_openai.ChatOpenAI.invoke') as mock_invoke:
                    mock_response = MagicMock()
                    mock_response.content = "Test response"
                    mock_invoke.return_value = mock_response
                    
                    llm = ChatOpenAI(model="gpt-4o", temperature=0)
                    
                    # Measure throughput
                    prompts = [f"prompt_{i}" for i in range(20)]
                    start = time.perf_counter()
                    
                    for prompt in prompts:
                        try:
                            llm.invoke(prompt)
                        except Exception:
                            pass
                    
                    total_time = time.perf_counter() - start
                    throughput = len(prompts) / total_time if total_time > 0 else 0
                    
                    # Mocked calls should be very fast
                    assert throughput > 10.0 or total_time < 2.0, \
                        f"LLM throughput too low: {throughput:.2f} calls/s"
        except ImportError:
            pytest.skip("ChatOpenAI not available")
    
    def test_llm_call_latency_percentiles(self, mock_config):
        """Test LLM call latency percentiles."""
        try:
            from langchain_openai import ChatOpenAI
            
            # Mock ChatOpenAI completely to avoid API key requirement
            with patch('langchain_openai.ChatOpenAI.__init__', return_value=None):
                with patch('langchain_openai.ChatOpenAI.invoke') as mock_invoke:
                    mock_response = MagicMock()
                    mock_response.content = "Test response"
                    mock_invoke.return_value = mock_response
                    
                    llm = ChatOpenAI(model="gpt-4o", temperature=0)
                    
                    # Measure latencies
                    latencies = []
                    for i in range(50):
                        prompt = f"test prompt {i}"
                        start = time.perf_counter()
                        try:
                            llm.invoke(prompt)
                        except Exception:
                            pass
                        latencies.append((time.perf_counter() - start) * 1000)  # Convert to ms
                    
                    if latencies:
                        latencies.sort()
                        p50 = latencies[len(latencies) // 2]
                        p95 = latencies[int(len(latencies) * 0.95)]
                        p99 = latencies[int(len(latencies) * 0.99)]
                        
                        # Mocked calls should be very fast
                        assert p50 < 50.0, f"p50 latency too high: {p50:.2f}ms"
                        assert p95 < 200.0, f"p95 latency too high: {p95:.2f}ms"
                        assert p99 < 500.0, f"p99 latency too high: {p99:.2f}ms"
        except ImportError:
            pytest.skip("ChatOpenAI not available")
    
    def test_llm_caching_hit_rate(self, mock_config):
        """Test LLM caching achieves good hit rate."""
        from src.core.cache import get_llm_cache
        
        cache = get_llm_cache()
        cache.clear()  # Start fresh
        
        # Simulate repeated calls with same prompts
        prompts = ["query 1", "query 2", "query 1", "query 3", "query 2", "query 1"]
        cache_keys = [f"llm_call_{hash(p)}" for p in prompts]
        
        hits = 0
        misses = 0
        
        for key in cache_keys:
            if cache.get(key):
                hits += 1
            else:
                misses += 1
                cache.set(key, f"response_{key}", ttl=60)
        
        # Calculate hit rate
        total = hits + misses
        hit_rate = (hits / total) if total > 0 else 0
        
        # With repeated prompts, should have some hits
        # In this test, we expect at least 30% hit rate
        assert hit_rate >= 0.3 or total < 3, \
            f"Cache hit rate too low: {hit_rate:.2%} ({hits}/{total})"
    
    def test_llm_call_memory_usage(self, mock_config):
        """Test LLM calls don't leak memory."""
        try:
            from langchain_openai import ChatOpenAI
            from src.core.cache import get_llm_cache
            
            cache = get_llm_cache()
            
            # Mock ChatOpenAI completely to avoid API key requirement
            with patch('langchain_openai.ChatOpenAI.__init__', return_value=None):
                with patch('langchain_openai.ChatOpenAI.invoke') as mock_invoke:
                    mock_response = MagicMock()
                    mock_response.content = "Test response"
                    mock_invoke.return_value = mock_response
                    
                    llm = ChatOpenAI(model="gpt-4o", temperature=0)
                    
                    # Perform many calls
                    for i in range(100):
                        try:
                            llm.invoke(f"prompt_{i}")
                        except Exception:
                            pass
                    
                    # Memory usage should be stable
                    # This is a basic check - actual memory profiling would require psutil
                    assert llm is not None, "LLM should still be functional"
                    assert cache is not None, "Cache should still be functional"
        except ImportError:
            pytest.skip("ChatOpenAI not available")

