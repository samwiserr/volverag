"""
Cross-encoder reranker for improved document relevance scoring.

Uses sentence-transformers cross-encoder models to re-rank retrieved documents
based on query-document relevance. This provides more accurate relevance scoring
than bi-encoder (embedding-based) approaches.
"""

import os
import logging
from typing import List, Tuple, Optional
from functools import lru_cache
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Lazy-load model
_cross_encoder_model = None


def _get_cross_encoder_model():
    """Get or load the cross-encoder model."""
    global _cross_encoder_model
    if _cross_encoder_model is None:
        try:
            from sentence_transformers import CrossEncoder
            
            model_name = os.getenv(
                "RAG_CROSS_ENCODER_MODEL",
                "cross-encoder/ms-marco-MiniLM-L-6-v2"
            )
            logger.info(f"[CROSS_ENCODER] Loading model: {model_name}")
            _cross_encoder_model = CrossEncoder(model_name)
            logger.info("[CROSS_ENCODER] Model loaded successfully")
        except ImportError:
            logger.error(
                "[CROSS_ENCODER] sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
            raise
        except Exception as e:
            logger.error(f"[CROSS_ENCODER] Failed to load model: {e}")
            raise
    return _cross_encoder_model


def rerank_documents(
    query: str,
    documents: List[Document],
    top_k: Optional[int] = None
) -> List[Document]:
    """
    Re-rank documents using cross-encoder model.
    
    Args:
        query: The search query
        documents: List of Document objects to re-rank
        top_k: Optional limit on number of documents to return (None = return all)
        
    Returns:
        Re-ranked list of Document objects, sorted by relevance (highest first)
    """
    if not documents:
        return []
    
    if len(documents) == 1:
        return documents
    
    try:
        model = _get_cross_encoder_model()
    except Exception as e:
        logger.warning(f"[CROSS_ENCODER] Reranking failed, returning original order: {e}")
        return documents[:top_k] if top_k else documents
    
    # Prepare query-document pairs
    pairs = []
    for doc in documents:
        # Use page_content for scoring
        doc_text = doc.page_content[:512]  # Limit length for efficiency
        pairs.append([query, doc_text])
    
    try:
        # Get relevance scores
        scores = model.predict(pairs)
        
        # Sort documents by score (descending)
        scored_docs = list(zip(scores, documents))
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        # Extract re-ranked documents
        reranked = [doc for _, doc in scored_docs]
        
        # Apply top_k limit if specified
        if top_k is not None:
            reranked = reranked[:top_k]
        
        logger.info(
            f"[CROSS_ENCODER] Re-ranked {len(documents)} documents, "
            f"returning top {len(reranked)} (scores: {scores[:len(reranked)]})"
        )
        
        return reranked
        
    except Exception as e:
        logger.warning(f"[CROSS_ENCODER] Reranking failed, returning original order: {e}")
        return documents[:top_k] if top_k else documents


def rerank_with_scores(
    query: str,
    documents: List[Document],
    top_k: Optional[int] = None
) -> List[Tuple[Document, float]]:
    """
    Re-rank documents and return with relevance scores.
    
    Args:
        query: The search query
        documents: List of Document objects to re-rank
        top_k: Optional limit on number of documents to return
        
    Returns:
        List of (Document, score) tuples, sorted by score (highest first)
    """
    if not documents:
        return []
    
    if len(documents) == 1:
        return [(documents[0], 1.0)]
    
    try:
        model = _get_cross_encoder_model()
    except Exception as e:
        logger.warning(f"[CROSS_ENCODER] Reranking failed, returning original order: {e}")
        # Return with dummy scores
        result = [(doc, 0.5) for doc in documents]
        return result[:top_k] if top_k else result
    
    # Prepare query-document pairs
    pairs = []
    for doc in documents:
        doc_text = doc.page_content[:512]
        pairs.append([query, doc_text])
    
    try:
        # Get relevance scores
        scores = model.predict(pairs)
        
        # Sort by score (descending)
        scored_docs = list(zip(scores, documents))
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        # Apply top_k limit if specified
        if top_k is not None:
            scored_docs = scored_docs[:top_k]
        
        logger.info(
            f"[CROSS_ENCODER] Re-ranked {len(documents)} documents with scores, "
            f"returning top {len(scored_docs)}"
        )
        
        return [(doc, float(score)) for score, doc in scored_docs]
        
    except Exception as e:
        logger.warning(f"[CROSS_ENCODER] Reranking failed, returning original order: {e}")
        result = [(doc, 0.5) for doc in documents]
        return result[:top_k] if top_k else result




