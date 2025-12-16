"""
Evaluation metrics for RAG system performance.

Implements standard IR metrics: Precision@K, Recall@K, MRR, NDCG@K
"""

import logging
from typing import List, Dict, Set, Optional
import numpy as np

logger = logging.getLogger(__name__)


def precision_at_k(relevant_docs: Set[str], retrieved_docs: List[str], k: int) -> float:
    """
    Calculate Precision@K: Fraction of top-K retrieved docs that are relevant.
    
    Args:
        relevant_docs: Set of relevant document IDs
        retrieved_docs: List of retrieved document IDs (ordered by relevance)
        k: Number of top documents to consider
        
    Returns:
        Precision@K score (0.0 to 1.0)
    """
    if k == 0 or not retrieved_docs:
        return 0.0
    
    top_k = retrieved_docs[:k]
    relevant_retrieved = sum(1 for doc_id in top_k if doc_id in relevant_docs)
    return relevant_retrieved / len(top_k)


def recall_at_k(relevant_docs: Set[str], retrieved_docs: List[str], k: int) -> float:
    """
    Calculate Recall@K: Fraction of relevant docs that were retrieved in top-K.
    
    Args:
        relevant_docs: Set of relevant document IDs
        retrieved_docs: List of retrieved document IDs (ordered by relevance)
        k: Number of top documents to consider
        
    Returns:
        Recall@K score (0.0 to 1.0)
    """
    if not relevant_docs:
        return 0.0 if retrieved_docs else 1.0
    
    if k == 0 or not retrieved_docs:
        return 0.0
    
    top_k = retrieved_docs[:k]
    relevant_retrieved = sum(1 for doc_id in top_k if doc_id in relevant_docs)
    return relevant_retrieved / len(relevant_docs)


def mean_reciprocal_rank(relevant_docs: Set[str], retrieved_docs: List[str]) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR): Average of 1/rank of first relevant doc.
    
    Args:
        relevant_docs: Set of relevant document IDs
        retrieved_docs: List of retrieved document IDs (ordered by relevance)
        
    Returns:
        MRR score (0.0 to 1.0)
    """
    if not relevant_docs or not retrieved_docs:
        return 0.0
    
    for rank, doc_id in enumerate(retrieved_docs, start=1):
        if doc_id in relevant_docs:
            return 1.0 / rank
    
    return 0.0


def ndcg_at_k(relevant_docs: Set[str], retrieved_docs: List[str], k: int) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain@K (NDCG@K).
    
    Args:
        relevant_docs: Set of relevant document IDs
        retrieved_docs: List of retrieved document IDs (ordered by relevance)
        k: Number of top documents to consider
        
    Returns:
        NDCG@K score (0.0 to 1.0)
    """
    if k == 0 or not retrieved_docs:
        return 0.0
    
    top_k = retrieved_docs[:k]
    
    # Calculate DCG
    dcg = 0.0
    for i, doc_id in enumerate(top_k, start=1):
        if doc_id in relevant_docs:
            # Binary relevance (1 if relevant, 0 otherwise)
            relevance = 1.0
            dcg += relevance / np.log2(i + 1)
    
    # Calculate IDCG (ideal DCG - all relevant docs at top)
    idcg = 0.0
    num_relevant = min(len(relevant_docs), k)
    for i in range(1, num_relevant + 1):
        idcg += 1.0 / np.log2(i + 1)
    
    # Normalize
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def evaluate_query(
    query: str,
    retrieved_docs: List[str],
    relevant_docs: Set[str],
    k_values: List[int] = [1, 3, 5, 10]
) -> Dict[str, float]:
    """
    Evaluate a single query against ground truth.
    
    Args:
        query: The query string
        retrieved_docs: List of retrieved document IDs (ordered by relevance)
        relevant_docs: Set of relevant document IDs
        k_values: List of K values to compute metrics for
        
    Returns:
        Dictionary of metric scores
    """
    metrics = {}
    
    for k in k_values:
        metrics[f"precision@{k}"] = precision_at_k(relevant_docs, retrieved_docs, k)
        metrics[f"recall@{k}"] = recall_at_k(relevant_docs, retrieved_docs, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(relevant_docs, retrieved_docs, k)
    
    metrics["mrr"] = mean_reciprocal_rank(relevant_docs, retrieved_docs)
    
    return metrics


def evaluate_batch(
    queries: List[Dict[str, any]],
    retrieved_docs_map: Dict[str, List[str]],
    k_values: List[int] = [1, 3, 5, 10]
) -> Dict[str, float]:
    """
    Evaluate a batch of queries and return average metrics.
    
    Args:
        queries: List of query dictionaries with 'query' and 'relevant_docs' keys
        retrieved_docs_map: Dictionary mapping query strings to retrieved doc IDs
        k_values: List of K values to compute metrics for
        
    Returns:
        Dictionary of average metric scores
    """
    all_metrics = []
    
    for query_data in queries:
        query = query_data.get("query", "")
        relevant_docs = set(query_data.get("relevant_docs", []))
        retrieved_docs = retrieved_docs_map.get(query, [])
        
        if not query:
            continue
        
        metrics = evaluate_query(query, retrieved_docs, relevant_docs, k_values)
        all_metrics.append(metrics)
    
    if not all_metrics:
        return {}
    
    # Average metrics across all queries
    avg_metrics = {}
    for k in k_values:
        avg_metrics[f"precision@{k}"] = np.mean([m[f"precision@{k}"] for m in all_metrics])
        avg_metrics[f"recall@{k}"] = np.mean([m[f"recall@{k}"] for m in all_metrics])
        avg_metrics[f"ndcg@{k}"] = np.mean([m[f"ndcg@{k}"] for m in all_metrics])
    
    avg_metrics["mrr"] = np.mean([m["mrr"] for m in all_metrics])
    
    return avg_metrics

