"""
Handler for incomplete queries in the RAG workflow.

Integrates query completion into the retrieval and answer generation pipeline.
"""

import logging
from typing import List, Optional
from langchain_core.documents import Document

from .query_completer import is_incomplete_query, complete_incomplete_query

logger = logging.getLogger(__name__)


def handle_incomplete_query(
    query: str,
    retrieve_func,
    max_completions: int = 3
) -> tuple[List[Document], Optional[str]]:
    """
    Handle an incomplete query by completing it and retrieving documents.
    
    Args:
        query: The user's query (may be incomplete)
        retrieve_func: Function to call for retrieval: retrieve_func(query: str, k: int) -> List[Document]
        max_completions: Maximum number of query variations to try
        
    Returns:
        Tuple of (retrieved_documents, completed_query_or_none)
        - If query was incomplete, returns documents from completed queries and the completed query
        - If query was complete, returns documents from original query and None
    """
    if not is_incomplete_query(query):
        # Query is complete, use as-is
        try:
            docs = retrieve_func(query, k=10)
            return docs, None
        except Exception as e:
            logger.error(f"[INCOMPLETE_HANDLER] Error retrieving for complete query: {e}")
            return [], None
    
    # Query is incomplete, complete it
    logger.info(f"[INCOMPLETE_HANDLER] Detected incomplete query: '{query}'")
    completed_queries = complete_incomplete_query(query, max_variations=max_completions)
    
    if not completed_queries or completed_queries == [query]:
        # Completion failed, use original
        try:
            docs = retrieve_func(query, k=10)
            return docs, None
        except Exception as e:
            logger.error(f"[INCOMPLETE_HANDLER] Error retrieving for original query: {e}")
            return [], None
    
    # Retrieve documents for all completed query variations
    all_docs = []
    seen_doc_ids = set()
    
    for completed_query in completed_queries:
        try:
            docs = retrieve_func(completed_query, k=10)
            # Deduplicate documents
            for doc in docs:
                # Use source + page as unique identifier
                doc_id = (
                    str(doc.metadata.get("source", "")),
                    doc.metadata.get("page", doc.metadata.get("page_number", ""))
                )
                if doc_id not in seen_doc_ids:
                    seen_doc_ids.add(doc_id)
                    all_docs.append(doc)
        except Exception as e:
            logger.warning(f"[INCOMPLETE_HANDLER] Error retrieving for completed query '{completed_query}': {e}")
            continue
    
    # Use the first completed query as the "primary" completed query
    primary_completed = completed_queries[0] if completed_queries else None
    
    logger.info(f"[INCOMPLETE_HANDLER] Retrieved {len(all_docs)} unique documents from {len(completed_queries)} completed queries")
    
    return all_docs, primary_completed




