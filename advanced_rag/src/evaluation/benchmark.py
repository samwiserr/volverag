"""
Benchmark runner for RAG system evaluation.

Runs evaluation suite and generates reports comparing metrics.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from .evaluator import evaluate_batch
from .test_suite import get_test_queries

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Runs benchmarks and generates evaluation reports."""
    
    def __init__(self, retriever_func):
        """
        Initialize benchmark runner.
        
        Args:
            retriever_func: Function that takes a query string and returns list of document IDs
        """
        self.retriever_func = retriever_func
    
    def run_evaluation(
        self,
        queries: Optional[List[Dict[str, Any]]] = None,
        k_values: List[int] = [1, 3, 5, 10],
        export_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run evaluation on test queries.
        
        Args:
            queries: List of test queries (uses default if None)
            k_values: List of K values for metrics
            export_path: Optional path to export results as JSON
            
        Returns:
            Dictionary with evaluation results
        """
        if queries is None:
            queries = get_test_queries()
        
        logger.info(f"[BENCHMARK] Running evaluation on {len(queries)} queries")
        
        # Retrieve documents for each query
        retrieved_docs_map = {}
        for query_data in queries:
            query = query_data.get("query", "")
            if not query:
                continue
            
            try:
                # Call retriever function to get document IDs
                # Note: This assumes retriever_func returns document IDs or we extract them
                retrieved = self.retriever_func(query)
                
                # Extract document IDs from retrieved results
                # This depends on the retriever_func implementation
                if isinstance(retrieved, list):
                    # If it returns Document objects, extract source IDs
                    doc_ids = []
                    for item in retrieved:
                        if hasattr(item, 'metadata'):
                            source = item.metadata.get('source', '')
                            if source:
                                doc_ids.append(Path(source).name)
                        elif isinstance(item, str):
                            doc_ids.append(item)
                    retrieved_docs_map[query] = doc_ids
                else:
                    retrieved_docs_map[query] = []
                
                logger.info(f"[BENCHMARK] Retrieved {len(retrieved_docs_map[query])} docs for query: {query[:50]}...")
            except Exception as e:
                logger.error(f"[BENCHMARK] Error retrieving for query '{query}': {e}")
                retrieved_docs_map[query] = []
        
        # Evaluate
        avg_metrics = evaluate_batch(queries, retrieved_docs_map, k_values)
        
        # Prepare results
        results = {
            "timestamp": datetime.now().isoformat(),
            "num_queries": len(queries),
            "k_values": k_values,
            "average_metrics": avg_metrics,
            "per_query_results": []
        }
        
        # Add per-query results
        for query_data in queries:
            query = query_data.get("query", "")
            relevant_docs = set(query_data.get("relevant_docs", []))
            retrieved_docs = retrieved_docs_map.get(query, [])
            
            if not query:
                continue
            
            from .evaluator import evaluate_query
            query_metrics = evaluate_query(query, retrieved_docs, relevant_docs, k_values)
            
            results["per_query_results"].append({
                "query": query,
                "category": query_data.get("category", "unknown"),
                "relevant_docs": list(relevant_docs),
                "retrieved_docs": retrieved_docs,
                "metrics": query_metrics
            })
        
        # Export if path provided
        if export_path:
            export_path = Path(export_path)
            export_path.parent.mkdir(parents=True, exist_ok=True)
            with open(export_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"[BENCHMARK] Results exported to {export_path}")
        
        return results
    
    def compare_results(
        self,
        baseline_results: Dict[str, Any],
        current_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compare two evaluation results.
        
        Args:
            baseline_results: Baseline evaluation results
            current_results: Current evaluation results
            
        Returns:
            Dictionary with comparison metrics
        """
        comparison = {
            "baseline": baseline_results.get("average_metrics", {}),
            "current": current_results.get("average_metrics", {}),
            "improvements": {}
        }
        
        baseline_metrics = baseline_results.get("average_metrics", {})
        current_metrics = current_results.get("average_metrics", {})
        
        # Calculate improvements
        for metric_name in baseline_metrics:
            if metric_name in current_metrics:
                baseline_val = baseline_metrics[metric_name]
                current_val = current_metrics[metric_name]
                improvement = current_val - baseline_val
                improvement_pct = (improvement / baseline_val * 100) if baseline_val > 0 else 0.0
                
                comparison["improvements"][metric_name] = {
                    "absolute": improvement,
                    "percentage": improvement_pct
                }
        
        return comparison

