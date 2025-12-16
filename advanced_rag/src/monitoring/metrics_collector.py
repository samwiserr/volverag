"""
Decorator-based metrics collection for automatic instrumentation.

Tracks query latency, token usage, cache hit rates, and retrieval quality.
"""

import time
import functools
import logging
from typing import Callable, Any, Dict, Optional
from datetime import datetime
import json
from pathlib import Path

logger = logging.getLogger(__name__)

# Global metrics storage
_metrics_log_path: Optional[Path] = None


def set_metrics_log_path(path: str):
    """Set the path for metrics logging."""
    global _metrics_log_path
    _metrics_log_path = Path(path)
    _metrics_log_path.parent.mkdir(parents=True, exist_ok=True)


def get_metrics_log_path() -> Optional[Path]:
    """Get the current metrics log path."""
    return _metrics_log_path


def log_metric(metric_name: str, value: float, metadata: Optional[Dict[str, Any]] = None):
    """
    Log a single metric.
    
    Args:
        metric_name: Name of the metric
        value: Metric value
        metadata: Optional metadata dictionary
    """
    log_path = get_metrics_log_path()
    if not log_path:
        return
    
    metric_entry = {
        "timestamp": datetime.now().isoformat(),
        "metric": metric_name,
        "value": value,
        "metadata": metadata or {}
    }
    
    try:
        with open(log_path, 'a') as f:
            f.write(json.dumps(metric_entry) + '\n')
    except Exception as e:
        logger.warning(f"[METRICS] Failed to log metric: {e}")


def track_latency(func: Callable) -> Callable:
    """
    Decorator to track function execution latency.
    
    Usage:
        @track_latency
        def my_function():
            ...
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            latency = time.time() - start_time
            
            log_metric(
                f"{func.__module__}.{func.__name__}.latency",
                latency,
                {"function": f"{func.__module__}.{func.__name__}"}
            )
            
            return result
        except Exception as e:
            latency = time.time() - start_time
            log_metric(
                f"{func.__module__}.{func.__name__}.latency",
                latency,
                {"function": f"{func.__module__}.{func.__name__}", "error": str(e)}
            )
            raise
    
    return wrapper


def track_token_usage(func: Callable) -> Callable:
    """
    Decorator to track LLM token usage.
    
    Expects the function to return a result with token usage information,
    or to have token usage logged separately.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        
        # Try to extract token usage from result
        if hasattr(result, 'response_metadata'):
            metadata = result.response_metadata
            if 'token_usage' in metadata:
                usage = metadata['token_usage']
                input_tokens = usage.get('prompt_tokens', 0)
                output_tokens = usage.get('completion_tokens', 0)
                total_tokens = usage.get('total_tokens', 0)
                
                log_metric(
                    f"{func.__module__}.{func.__name__}.tokens.input",
                    input_tokens,
                    {"function": f"{func.__module__}.{func.__name__}"}
                )
                log_metric(
                    f"{func.__module__}.{func.__name__}.tokens.output",
                    output_tokens,
                    {"function": f"{func.__module__}.{func.__name__}"}
                )
                log_metric(
                    f"{func.__module__}.{func.__name__}.tokens.total",
                    total_tokens,
                    {"function": f"{func.__module__}.{func.__name__}"}
                )
        
        return result
    
    return wrapper


def track_retrieval_quality(func: Callable) -> Callable:
    """
    Decorator to track retrieval quality metrics.
    
    Expects function to return a list of documents or a tuple (docs, scores).
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        
        # Extract documents
        if isinstance(result, tuple):
            docs = result[0]
        elif isinstance(result, list):
            docs = result
        else:
            docs = []
        
        num_docs = len(docs) if docs else 0
        
        log_metric(
            f"{func.__module__}.{func.__name__}.retrieval.num_docs",
            num_docs,
            {"function": f"{func.__module__}.{func.__name__}"}
        )
        
        return result
    
    return wrapper


def track_cache_hit(func: Callable) -> Callable:
    """
    Decorator to track cache hit/miss rates.
    
    Function should set a 'cache_hit' attribute on the result or in metadata.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        
        # Check for cache hit indicator
        cache_hit = getattr(result, 'cache_hit', None)
        if cache_hit is None and hasattr(result, 'metadata'):
            cache_hit = result.metadata.get('cache_hit', None)
        
        if cache_hit is not None:
            log_metric(
                f"{func.__module__}.{func.__name__}.cache.hit",
                1.0 if cache_hit else 0.0,
                {"function": f"{func.__module__}.{func.__name__}"}
            )
        
        return result
    
    return wrapper

