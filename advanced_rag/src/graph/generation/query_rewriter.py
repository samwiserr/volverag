"""
Query rewriter for improving retrieval.
"""
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage
from src.core.config import get_config
from src.core.model_factory import get_chat_model
from src.core.result import Result, AppError, ErrorType
from src.core.logging import get_logger
from src.graph.utils.message_utils import _latest_user_question

logger = get_logger(__name__)


REWRITE_PROMPT = (
    "You are a question rewriter. Rewrite the following question to improve retrieval "
    "from a document database. Make it more specific and clear, but DO NOT add new "
    "entities (well names, formations, properties) that were not in the original question.\n\n"
    "Original question: {question}\n"
    "Context from previous retrieval: {context}\n\n"
    "Rewritten question:"
)


class QueryRewriter:
    """
    Rewrites user questions to improve retrieval.
    """
    
    def __init__(self):
        """Initialize query rewriter."""
        self._response_model = None
    
    def _get_response_model(self):
        """Get or create response model."""
        if self._response_model is None:
            try:
                config = get_config()
                model_name = str(config.llm_model)
            except Exception:
                import os
                model_name = os.getenv("GROQ_MODEL", os.getenv("OPENAI_MODEL", "llama-3.3-70b-versatile"))
            self._response_model = get_chat_model(model_name, temperature=0, role="rewriter")
        return self._response_model
    
    def rewrite(self, state: MessagesState) -> Result[dict, AppError]:
        """
        Rewrite the original user question to improve retrieval.
        
        Args:
            state: MessagesState containing messages
            
        Returns:
            Result containing dict with 'messages' key containing rewritten question
        """
        try:
            messages = state.get("messages", [])
            question = _latest_user_question(messages)
            context = messages[-1].content if messages else ""
            
            prompt = REWRITE_PROMPT.format(question=question, context=context)
            
            # Use caching for query rewriting
            from src.core.cache import get_llm_cache, generate_cache_key
            cache = get_llm_cache()
            cache_key = f"rewrite_question:{generate_cache_key(question, context)}"
            
            # Try cache first
            cached_response = cache.get(cache_key)
            if cached_response is not None:
                logger.debug("[REWRITE] Cache HIT for query rewriting")
                response = cached_response
            else:
                logger.debug("[REWRITE] Cache MISS for query rewriting")
                response = self._get_response_model().invoke([{"role": "user", "content": prompt}])
                # Cache the response
                cache.set(cache_key, response)
            
            logger.info(f"[REWRITE] Original: {question}")
            logger.info(f"[REWRITE] Rewritten: {response.content}")
            
            return Result.ok({"messages": [HumanMessage(content=response.content)]})
            
        except Exception as e:
            logger.error(f"[REWRITE] Error during rewrite: {e}", exc_info=True)
            return Result.from_exception(e, ErrorType.PROCESSING_ERROR)

