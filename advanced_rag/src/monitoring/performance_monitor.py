"""
Performance monitoring for RAG system.

Tracks query latency, token usage, cache hit rates, and retrieval quality.
"""

import os
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
import json

from .metrics_collector import set_metrics_log_path, log_metric

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Monitors and tracks RAG system performance metrics."""
    
    def __init__(self, metrics_log_path: Optional[str] = None):
        """
        Initialize performance monitor.
        
        Args:
            metrics_log_path: Path to metrics log file (JSONL format)
        """
        if metrics_log_path is None:
            metrics_log_path = os.getenv(
                "RAG_METRICS_LOG_PATH",
                "./data/monitoring/metrics.jsonl"
            )
        
        set_metrics_log_path(metrics_log_path)
        self.metrics_log_path = Path(metrics_log_path)
        self.metrics_log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"[MONITOR] Initialized with metrics log: {self.metrics_log_path}")
    
    def track_query(
        self,
        query: str,
        retrieval_time: float,
        llm_time: float,
        total_time: float,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        num_docs_retrieved: Optional[int] = None,
        cache_hit: Optional[bool] = None
    ):
        """
        Track a complete query execution.
        
        Args:
            query: The user query
            retrieval_time: Time spent on retrieval (seconds)
            llm_time: Time spent on LLM calls (seconds)
            total_time: Total query time (seconds)
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            num_docs_retrieved: Number of documents retrieved
            cache_hit: Whether cache was hit
        """
        log_metric(
            "query.total_time",
            total_time,
            {"query": query[:100]}  # Truncate long queries
        )
        
        log_metric(
            "query.retrieval_time",
            retrieval_time,
            {"query": query[:100]}
        )
        
        log_metric(
            "query.llm_time",
            llm_time,
            {"query": query[:100]}
        )
        
        if input_tokens is not None:
            log_metric(
                "query.tokens.input",
                input_tokens,
                {"query": query[:100]}
            )
        
        if output_tokens is not None:
            log_metric(
                "query.tokens.output",
                output_tokens,
                {"query": query[:100]}
            )
        
        if num_docs_retrieved is not None:
            log_metric(
                "query.retrieval.num_docs",
                num_docs_retrieved,
                {"query": query[:100]}
            )
        
        if cache_hit is not None:
            log_metric(
                "query.cache.hit",
                1.0 if cache_hit else 0.0,
                {"query": query[:100]}
            )
    
    def get_recent_metrics(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent metrics from log file.
        
        Args:
            limit: Maximum number of metrics to return
            
        Returns:
            List of metric dictionaries
        """
        if not self.metrics_log_path.exists():
            return []
        
        metrics = []
        try:
            with open(self.metrics_log_path, 'r') as f:
                lines = f.readlines()
                # Get last N lines
                for line in lines[-limit:]:
                    try:
                        metric = json.loads(line.strip())
                        metrics.append(metric)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.warning(f"[MONITOR] Error reading metrics: {e}")
        
        return metrics
    
    def get_statistics(self, metric_name: str, limit: int = 100) -> Dict[str, float]:
        """
        Get statistics for a specific metric.
        
        Args:
            metric_name: Name of the metric
            limit: Number of recent metrics to analyze
            
        Returns:
            Dictionary with statistics (mean, min, max, count)
        """
        metrics = self.get_recent_metrics(limit)
        relevant_metrics = [m for m in metrics if m.get('metric') == metric_name]
        
        if not relevant_metrics:
            return {"count": 0, "mean": 0.0, "min": 0.0, "max": 0.0}
        
        values = [m['value'] for m in relevant_metrics if isinstance(m.get('value'), (int, float))]
        
        if not values:
            return {"count": 0, "mean": 0.0, "min": 0.0, "max": 0.0}
        
        return {
            "count": len(values),
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values)
        }

