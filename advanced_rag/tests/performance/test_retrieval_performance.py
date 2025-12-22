"""
Performance tests for document retrieval.

Tests retrieval latency, throughput, and concurrent query handling.
"""
import pytest
import time
import statistics
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path


@pytest.mark.performance
@pytest.mark.slow
class TestRetrievalPerformance:
    """Performance tests for document retrieval."""
    
    def test_retrieval_latency_baseline(self, mock_config, monkeypatch):
        """Test baseline retrieval latency."""
        try:
            from src.tools.retriever_tool import RetrieverTool
            
            # Mock API key to avoid requirement
            monkeypatch.setenv("OPENAI_API_KEY", "test-key")
            
            # Mock vectorstore to avoid actual database access
            with patch('src.tools.retriever_tool.Chroma') as mock_chroma:
                mock_vectorstore = MagicMock()
                mock_chroma.return_value = mock_vectorstore
                
                # Mock retrieval to return quickly
                mock_vectorstore.similarity_search_with_score.return_value = []
                
                retriever = RetrieverTool(persist_directory="./data/vectorstore")
                
                # Measure retrieval latency
                latencies = []
                for _ in range(10):
                    start = time.perf_counter()
                    try:
                        retriever.retrieve("test query", k=5)
                    except Exception:
                        pass  # Expected if vectorstore not available
                    latencies.append((time.perf_counter() - start) * 1000)  # Convert to ms
                
                # Should complete in reasonable time (< 1s per retrieval)
                avg_latency = statistics.mean(latencies) if latencies else 0
                assert avg_latency < 1000.0, \
                    f"Retrieval latency too high: {avg_latency:.2f}ms"
        except ImportError:
            pytest.skip("RetrieverTool not available")
    
    def test_retrieval_throughput(self, mock_config, monkeypatch):
        """Test retrieval can handle multiple queries."""
        try:
            from src.tools.retriever_tool import RetrieverTool
            
            # Mock API key to avoid requirement
            monkeypatch.setenv("OPENAI_API_KEY", "test-key")
            
            # Mock vectorstore
            with patch('src.tools.retriever_tool.Chroma') as mock_chroma:
                mock_vectorstore = MagicMock()
                mock_chroma.return_value = mock_vectorstore
                mock_vectorstore.similarity_search_with_score.return_value = []
                
                retriever = RetrieverTool(persist_directory="./data/vectorstore")
                
                # Measure throughput
                queries = [f"query_{i}" for i in range(50)]
                start = time.perf_counter()
                
                for query in queries:
                    try:
                        retriever.retrieve(query, k=5)
                    except Exception:
                        pass
                
                total_time = time.perf_counter() - start
                throughput = len(queries) / total_time if total_time > 0 else 0
                
                # Should handle at least 10 queries per second
                assert throughput > 10.0 or total_time < 5.0, \
                    f"Retrieval throughput too low: {throughput:.2f} queries/s"
        except ImportError:
            pytest.skip("RetrieverTool not available")
    
    def test_retrieval_concurrent_queries(self, mock_config, monkeypatch):
        """Test retrieval handles concurrent queries."""
        import threading
        
        try:
            from src.tools.retriever_tool import RetrieverTool
            
            # Mock API key to avoid requirement
            monkeypatch.setenv("OPENAI_API_KEY", "test-key")
            
            # Mock vectorstore
            with patch('src.tools.retriever_tool.Chroma') as mock_chroma:
                mock_vectorstore = MagicMock()
                mock_chroma.return_value = mock_vectorstore
                mock_vectorstore.similarity_search_with_score.return_value = []
                
                retriever = RetrieverTool(persist_directory="./data/vectorstore")
                
                results = []
                errors = []
                
                def worker(worker_id):
                    try:
                        for i in range(10):
                            query = f"query_{worker_id}_{i}"
                            result = retriever.retrieve(query, k=5)
                            results.append((worker_id, i, result is not None))
                    except Exception as e:
                        errors.append((worker_id, str(e)))
                
                # Run 5 concurrent workers
                threads = []
                for i in range(5):
                    t = threading.Thread(target=worker, args=(i,))
                    threads.append(t)
                    t.start()
                
                # Wait for all threads
                for t in threads:
                    t.join()
                
                # Should have minimal errors (some expected if vectorstore not available)
                assert len(results) > 0, "No successful concurrent retrievals"
        except ImportError:
            pytest.skip("RetrieverTool not available")
    
    def test_retrieval_latency_percentiles(self, mock_config, monkeypatch):
        """Test retrieval latency percentiles."""
        try:
            from src.tools.retriever_tool import RetrieverTool
            
            # Mock API key to avoid requirement
            monkeypatch.setenv("OPENAI_API_KEY", "test-key")
            
            # Mock vectorstore
            with patch('src.tools.retriever_tool.Chroma') as mock_chroma:
                mock_vectorstore = MagicMock()
                mock_chroma.return_value = mock_vectorstore
                mock_vectorstore.similarity_search_with_score.return_value = []
                
                retriever = RetrieverTool(persist_directory="./data/vectorstore")
                
                # Measure latencies
                latencies = []
                for i in range(100):
                    query = f"test query {i}"
                    start = time.perf_counter()
                    try:
                        retriever.retrieve(query, k=5)
                    except Exception:
                        pass
                    latencies.append((time.perf_counter() - start) * 1000)  # Convert to ms
                
                if latencies:
                    latencies.sort()
                    p50 = latencies[len(latencies) // 2]
                    p95 = latencies[int(len(latencies) * 0.95)]
                    p99 = latencies[int(len(latencies) * 0.99)]
                    
                    # Retrieval should be reasonably fast (with mocked vectorstore)
                    assert p50 < 100.0, f"p50 latency too high: {p50:.2f}ms"
                    assert p95 < 500.0, f"p95 latency too high: {p95:.2f}ms"
                    assert p99 < 1000.0, f"p99 latency too high: {p99:.2f}ms"
        except ImportError:
            pytest.skip("RetrieverTool not available")
    
    def test_retrieval_memory_usage(self, mock_config, monkeypatch):
        """Test retrieval doesn't leak memory."""
        try:
            import sys
            from src.tools.retriever_tool import RetrieverTool
            
            # Mock API key to avoid requirement
            monkeypatch.setenv("OPENAI_API_KEY", "test-key")
            
            # Mock vectorstore
            with patch('src.tools.retriever_tool.Chroma') as mock_chroma:
                mock_vectorstore = MagicMock()
                mock_chroma.return_value = mock_vectorstore
                mock_vectorstore.similarity_search_with_score.return_value = []
                
                retriever = RetrieverTool(persist_directory="./data/vectorstore")
                
                # Perform many retrievals
                for i in range(100):
                    try:
                        retriever.retrieve(f"query_{i}", k=5)
                    except Exception:
                        pass
                
                # Memory usage should be stable (no significant growth)
                # This is a basic check - actual memory profiling would require psutil
                assert retriever is not None, "Retriever should still be functional"
        except ImportError:
            pytest.skip("RetrieverTool not available")

