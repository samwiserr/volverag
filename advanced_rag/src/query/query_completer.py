"""
Query completer for handling incomplete queries.

Detects incomplete queries (ending with "...", "was", "is", etc.) and uses LLM
to expand them into complete, answerable questions.
"""

import os
import re
import logging
from typing import List, Optional
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

# Lazy-init LLM
_completion_model: Optional[ChatOpenAI] = None


def _get_completion_model() -> ChatOpenAI:
    """Get or create the query completion LLM."""
    global _completion_model
    if _completion_model is None:
        model = os.getenv("RAG_QUERY_COMPLETION_MODEL", "gpt-4o")
        _completion_model = ChatOpenAI(model=model, temperature=0)
    return _completion_model


def is_incomplete_query(query: str) -> bool:
    """
    Detect if a query is incomplete.
    
    Patterns:
    - Ends with "..." (ellipsis)
    - Ends with incomplete verb: "was", "is", "has", "did", "will", "were", "are", "have"
    - Ends with "and", "or", "but"
    - Contains "Wellbore X was" or "Well X was" without completion
    """
    if not query or not isinstance(query, str):
        return False
    
    query = query.strip()
    if not query:
        return False
    
    # Pattern 1: Ends with ellipsis
    if query.endswith("..."):
        return True
    
    # Pattern 2: Ends with incomplete verb
    incomplete_verbs = ["was", "is", "has", "did", "will", "were", "are", "have", "had", "would", "could", "should"]
    query_lower = query.lower().strip()
    for verb in incomplete_verbs:
        if query_lower.endswith(f" {verb}") or query_lower.endswith(f" {verb}."):
            return True
    
    # Pattern 3: Ends with conjunction
    if query_lower.endswith((" and", " or", " but", " and.", " or.", " but.")):
        return True
    
    # Pattern 4: Contains "Wellbore X was" or "Well X was" without completion
    # Check if it ends with "was" or "was..." after a well name
    wellbore_pattern = r"(?:Wellbore|Well)\s+[A-Z0-9/_-]+\s+was\s*\.\.?\.?$"
    if re.search(wellbore_pattern, query, re.IGNORECASE):
        return True
    
    return False


def complete_incomplete_query(query: str, max_variations: int = 3) -> List[str]:
    """
    Complete an incomplete query using LLM.
    
    Args:
        query: The incomplete query string
        max_variations: Maximum number of question variations to generate
        
    Returns:
        List of complete question variations
    """
    if not is_incomplete_query(query):
        return [query]  # Return original if not incomplete
    
    try:
        llm = _get_completion_model()
        
        prompt = f"""You are helping to complete an incomplete query about oil and gas well data.

The user started typing a query but didn't finish it. Your task is to expand this incomplete query into {max_variations} complete, answerable questions that would help retrieve the information the user is likely seeking.

Incomplete query: "{query}"

Generate {max_variations} complete questions that:
1. Complete the thought expressed in the incomplete query
2. Are specific and answerable from well data documents
3. Cover different aspects if the query is ambiguous
4. Use proper well naming conventions (e.g., "15/9-F-5", "15/9-F-15 C")

Return ONLY the questions, one per line, without numbering or bullets. Each question should be complete and standalone.

Example:
If the query is "Wellbore 15/9-F-15 C was sidetracked..."
You might generate:
What happened to wellbore 15/9-F-15 C? Was it sidetracked?
What are the details about the sidetrack of wellbore 15/9-F-15 C?
When and why was wellbore 15/9-F-15 C sidetracked?

Now complete this query: "{query}"
"""
        
        response = llm.invoke([{"role": "user", "content": prompt}])
        content = response.content.strip()
        
        # Parse the response into individual questions
        questions = []
        for line in content.split("\n"):
            line = line.strip()
            # Remove numbering/bullets if present
            line = re.sub(r"^[\d\.\-\*\)]\s*", "", line)
            if line and len(line) > 10:  # Filter out very short lines
                questions.append(line)
        
        # Limit to max_variations
        questions = questions[:max_variations]
        
        # If we got questions, return them; otherwise return original
        if questions:
            logger.info(f"[QUERY_COMPLETION] Expanded '{query}' into {len(questions)} questions")
            return questions
        else:
            logger.warning(f"[QUERY_COMPLETION] Failed to generate questions, returning original")
            return [query]
            
    except Exception as e:
        logger.error(f"[QUERY_COMPLETION] Error completing query: {e}")
        return [query]  # Fallback to original query





