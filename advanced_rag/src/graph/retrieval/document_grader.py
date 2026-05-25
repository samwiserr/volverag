"""
Document grader for determining relevance of retrieved documents.
"""
from typing import Literal
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, ToolMessage
from pydantic import BaseModel, Field
from src.core.config import get_config
from src.core.model_factory import get_chat_model
from src.core.result import Result, AppError, ErrorType
from src.core.logging import get_logger
from src.graph.utils.message_utils import _latest_user_question

logger = get_logger(__name__)


class GradeDocuments(BaseModel):
    """Grade documents using a binary score for relevance check."""
    binary_score: Literal["yes", "no"] = Field(
        description="Documents answer the question: 'yes' or 'no'"
    )


GRADE_PROMPT = (
    "You are a grader assessing whether retrieved documents are relevant to the question.\n"
    "If the documents contain the answer to the question, grade them as relevant.\n"
    "Give a binary score 'yes' or 'no'.\n\n"
    "Question: {question}\n\n"
    "Documents: {context}\n\n"
    "Answer:"
)


class DocumentGrader:
    """
    Determines whether retrieved documents are relevant to the question.
    """
    
    def __init__(self):
        """Initialize document grader."""
        self._grader_model = None
    
    def _get_grader_model(self):
        """Get or create grader model."""
        if self._grader_model is None:
            try:
                config = get_config()
                model_name = str(config.grade_model)
            except Exception:
                import os
                model_name = os.getenv("GROQ_MODEL", os.getenv("OPENAI_GRADE_MODEL", os.getenv("OPENAI_MODEL", "llama-3.3-70b-versatile")))
            self._grader_model = get_chat_model(model_name, temperature=0, role="grader")
        return self._grader_model
    
    def grade(self, state: MessagesState) -> Result[Literal["generate_answer", "rewrite_question"], AppError]:
        """
        Determine whether the retrieved documents are relevant to the question.
        
        Args:
            state: MessagesState containing messages
            
        Returns:
            Result containing "generate_answer" or "rewrite_question"
        """
        try:
            messages = state.get("messages", [])
            question = _latest_user_question(messages)
            
            # Extract context from tool messages
            context_parts = []
            rewrite_count = 0
            
            for msg in messages:
                # Count rewrite attempts
                if isinstance(msg, HumanMessage) and msg != messages[0]:
                    rewrite_count += 1
                
                # Extract tool message content
                if isinstance(msg, ToolMessage):
                    tool_name = getattr(msg, 'name', 'unknown')
                    logger.info(
                        f"[GRADE] Found ToolMessage from tool '{tool_name}', "
                        f"content length: {len(msg.content) if isinstance(msg.content, str) else 'non-string'}"
                    )
                    context_parts.append(msg.content)
                elif hasattr(msg, 'content') and msg.content:
                    if isinstance(msg.content, str) and len(msg.content) > 100:
                        logger.debug(f"[GRADE] Found potential tool message content (length: {len(msg.content)})")
                        context_parts.append(msg.content)
            
            # Prevent infinite rewrite loops
            if rewrite_count >= 2:
                logger.warning(f"[GRADE] Too many rewrite attempts ({rewrite_count}) - proceeding to generate answer anyway")
                return Result.ok("generate_answer")
            
            if not context_parts:
                context = messages[-1].content if messages else ""
            else:
                context = "\n\n".join(context_parts)
            
            logger.info(f"[GRADE] Question: {question[:100]}...")
            logger.info(f"[GRADE] Context length: {len(context)} chars, Rewrite count: {rewrite_count}")
            
            # Quick heuristic checks
            if len(context) < 50:
                logger.info("[GRADE] Context too short - rewriting question")
                return Result.ok("rewrite_question")
            
            # Quick check for partial sentences with substantial context
            question_lower = question.lower()
            is_partial_sentence = (
                not question.strip().endswith(('.', '!', '?', ':')) and
                ('is' in question_lower or 'accordingly' in question_lower or 'reported' in question_lower)
            )
            
            if is_partial_sentence and len(context) > 500:
                logger.info("[GRADE] Quick check: Partial sentence query with substantial context - marking as relevant")
                return Result.ok("generate_answer")
            
            # Use LLM for final grading (with caching)
            prompt = GRADE_PROMPT.format(question=question, context=context[:3000])
            
            # Generate cache key for this grading operation
            from src.core.cache import get_llm_cache, generate_cache_key
            cache = get_llm_cache()
            cache_key = f"grade_documents:{generate_cache_key(question, context[:3000])}"
            
            # Try cache first
            cached_response = cache.get(cache_key)
            if cached_response is not None:
                logger.debug("[GRADE] Cache HIT for document grading")
                response = cached_response
            else:
                logger.debug("[GRADE] Cache MISS for document grading")
                response = self._get_grader_model().with_structured_output(GradeDocuments).invoke(
                    [{"role": "user", "content": prompt}]
                )
                # Cache the response
                cache.set(cache_key, response)
            score = response.binary_score
            
            if score == "yes":
                logger.info("[GRADE] Documents are relevant - proceeding to generate answer")
                return Result.ok("generate_answer")
            else:
                logger.info("[GRADE] Documents are not relevant - rewriting question")
                return Result.ok("rewrite_question")
                
        except Exception as e:
            logger.error(f"[GRADE] Error during grading: {e}", exc_info=True)
            # On error, default to generating answer
            return Result.ok("generate_answer")

