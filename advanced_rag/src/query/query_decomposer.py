"""
Query decomposition and rewriting for complex multi-part queries.

Decomposes complex queries into simpler sub-queries and performs query rewriting
for better retrieval performance.
"""

import os
import logging
from typing import List, Optional
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Lazy-init LLM
_decomposition_model: Optional[ChatOpenAI] = None


def _get_decomposition_model() -> ChatOpenAI:
    """Get or create the query decomposition LLM."""
    global _decomposition_model
    if _decomposition_model is None:
        model = os.getenv("RAG_DECOMPOSITION_MODEL", "gpt-4o")
        _decomposition_model = ChatOpenAI(model=model, temperature=0)
    return _decomposition_model


class QueryDecomposition(BaseModel):
    """Structured output for query decomposition."""
    is_complex: bool = Field(description="Whether the query is complex and needs decomposition")
    sub_queries: List[str] = Field(description="List of decomposed sub-queries if complex, otherwise empty list")
    rewritten_query: str = Field(description="Rewritten version of the query for better retrieval")


def decompose_query(query: str) -> tuple[bool, List[str], str]:
    """
    Decompose a complex query into simpler sub-queries.
    
    Args:
        query: The user's query
        
    Returns:
        Tuple of (is_complex, sub_queries, rewritten_query)
        - is_complex: Whether the query was decomposed
        - sub_queries: List of decomposed sub-queries
        - rewritten_query: Rewritten version of the original query
    """
    if not query or not isinstance(query, str):
        return False, [], query
    
    try:
        llm = _get_decomposition_model()
        
        prompt = f"""You are a query analysis system for a petrophysical document RAG system.

Analyze the following query and determine:
1. If it is a complex query that asks multiple things (e.g., "What is the porosity AND permeability for Hugin in 15/9-F-5?")
2. If so, decompose it into simpler sub-queries
3. Rewrite the query (or each sub-query) for better retrieval using domain-specific terminology

Query: "{query}"

For complex queries, decompose into sub-queries. For example:
- "What is the porosity and permeability for Hugin in 15/9-F-5?" 
  → Sub-queries: ["What is the porosity for Hugin formation in well 15/9-F-5?", "What is the permeability for Hugin formation in well 15/9-F-5?"]

For simple queries, just rewrite for better retrieval. For example:
- "poro hugin 15/9-F-5" → "What is the porosity for Hugin formation in well 15/9-F-5?"

Return a JSON object with:
- is_complex: boolean (true if query needs decomposition)
- sub_queries: list of strings (empty if not complex)
- rewritten_query: string (rewritten version of the query or first sub-query if complex)
"""
        
        response = llm.with_structured_output(QueryDecomposition).invoke([
            {"role": "user", "content": prompt}
        ])
        
        is_complex = response.is_complex
        sub_queries = response.sub_queries or []
        rewritten = response.rewritten_query or query
        
        if is_complex:
            logger.info(f"[QUERY_DECOMPOSITION] Decomposed query into {len(sub_queries)} sub-queries")
        else:
            logger.info(f"[QUERY_DECOMPOSITION] Rewrote query: '{query}' → '{rewritten}'")
        
        return is_complex, sub_queries, rewritten
        
    except Exception as e:
        logger.warning(f"[QUERY_DECOMPOSITION] Decomposition failed, using original query: {e}")
        return False, [], query


def rewrite_query(query: str) -> str:
    """
    Rewrite a query for better retrieval using domain-specific terminology.
    
    Args:
        query: The user's query
        
    Returns:
        Rewritten query string
    """
    _, _, rewritten = decompose_query(query)
    return rewritten


def expand_query_synonyms(query: str) -> List[str]:
    """
    Expand query with domain-specific synonyms and variations.
    
    Args:
        query: The user's query
        
    Returns:
        List of query variations
    """
    variations = [query]
    
    # Domain-specific expansions
    query_lower = query.lower()
    
    # Porosity synonyms
    if "poro" in query_lower or "porosity" in query_lower:
        variations.append(query.replace("poro", "porosity").replace("Poro", "Porosity"))
        variations.append(query.replace("porosity", "phif").replace("Porosity", "PHIF"))
    
    # Permeability synonyms
    if "perm" in query_lower or "permeability" in query_lower:
        variations.append(query.replace("perm", "permeability").replace("Perm", "Permeability"))
        variations.append(query.replace("permeability", "klogh").replace("Permeability", "KLOGH"))
    
    # Water saturation synonyms
    if "sw" in query_lower or "water saturation" in query_lower:
        variations.append(query.replace("sw", "water saturation").replace("SW", "Water Saturation"))
    
    # Net/gross synonyms
    if "ntg" in query_lower or "net" in query_lower and "gross" in query_lower:
        variations.append(query.replace("ntg", "net to gross").replace("NTG", "Net to Gross"))
        variations.append(query.replace("net to gross", "net/gross").replace("Net to Gross", "Net/Gross"))
    
    # Remove duplicates while preserving order
    seen = set()
    unique_variations = []
    for v in variations:
        if v.lower() not in seen:
            seen.add(v.lower())
            unique_variations.append(v)
    
    return unique_variations

