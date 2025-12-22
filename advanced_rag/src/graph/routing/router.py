"""
Query router that orchestrates routing strategies.
"""
from typing import List
from langgraph.graph import MessagesState
from langchain_core.messages import AIMessage
from src.core.result import Result, AppError, ErrorType
from src.core.logging import get_logger
from src.normalize.query_normalizer import normalize_query, extract_well
from src.graph.utils.message_utils import _latest_user_question, _infer_recent_context
from .strategies import (
    DepthRoutingStrategy,
    PetroParamsRoutingStrategy,
    EvalParamsRoutingStrategy,
    SectionRoutingStrategy,
)

logger = get_logger(__name__)


class QueryRouter:
    """
    Orchestrates routing strategies to determine which tool(s) to call.
    
    Strategies are evaluated in priority order, and the first matching
    strategy handles the routing.
    """
    
    def __init__(self, tools: List):
        """
        Initialize router with tools.
        
        Args:
            tools: List of tools available for routing
        """
        self.tools = tools
        self.strategies = [
            DepthRoutingStrategy(),
            PetroParamsRoutingStrategy(),
            EvalParamsRoutingStrategy(),
            SectionRoutingStrategy(),
            # Note: Formation and FactLike strategies are complex and will be added incrementally
        ]
        # Sort by priority (lower = higher priority)
        self.strategies.sort(key=lambda s: s.priority)
    
    def route(self, state: MessagesState) -> Result[dict, AppError]:
        """
        Route query to appropriate tool(s).
        
        Args:
            state: LangGraph state containing messages
            
        Returns:
            Result containing dict with 'messages' key, or error
        """
        try:
            messages = state.get("messages", [])
            question = _latest_user_question(messages)
            
            # Edge case: Empty or None question
            if not question or not isinstance(question, str) or not question.strip():
                logger.warning(f"[ROUTING] Empty or invalid question: '{question}'")
                return Result.ok({
                    "messages": [
                        AIMessage(content="I didn't receive a valid question. Please ask a question about the Volve petrophysical reports.")
                    ]
                })
            
            # Edge case: Very long queries
            MAX_QUERY_LENGTH = 5000
            if len(question) > MAX_QUERY_LENGTH:
                logger.warning(f"[ROUTING] Query too long: {len(question)} characters")
                return Result.ok({
                    "messages": [
                        AIMessage(content=f"Your question is too long ({len(question)} characters). Please keep questions under {MAX_QUERY_LENGTH} characters.")
                    ]
                })
            
            # Normalize query
            ql = question.lower()
            persist_dir = "./data/vectorstore"  # Default, will be resolved properly in strategies
            nq = normalize_query(question, persist_dir=persist_dir)
            
            # Infer context from recent messages
            if not nq.well or not nq.formation:
                cw, cf = _infer_recent_context(messages)
                if not nq.well and cw:
                    nq = nq.__class__(
                        raw=nq.raw, well=cw, formation=nq.formation,
                        property=nq.property, tool=nq.tool, intent=nq.intent
                    )
                if not nq.formation and cf:
                    nq = nq.__class__(
                        raw=nq.raw, well=nq.well, formation=cf,
                        property=nq.property, tool=nq.tool, intent=nq.intent
                    )
            
            # Build tool query (simplified for now)
            tool_query = question.strip()
            
            # Try each strategy in priority order
            for strategy in self.strategies:
                if strategy.should_route(question, nq, tool_query, persist_dir):
                    logger.info(f"[ROUTING] Strategy '{strategy.__class__.__name__}' matched")
                    result = strategy.route(question, nq, tool_query, persist_dir)
                    if result.is_ok():
                        return Result.ok({"messages": [result.unwrap()]})
                    else:
                        # Log error but continue to next strategy or fallback
                        logger.warning(f"[ROUTING] Strategy '{strategy.__class__.__name__}' failed: {result.error()}")
            
            # Fallback: Use LLM to decide (original behavior)
            logger.info("[ROUTING] No strategy matched, using LLM fallback")
            from langchain_openai import ChatOpenAI
            from src.core.config import get_config
            try:
                config = get_config()
                model = ChatOpenAI(model=config.llm_model.value, temperature=0)
            except Exception:
                # Fallback if config not available
                import os
                model_name = os.getenv("OPENAI_MODEL", "gpt-4o")
                model = ChatOpenAI(model=model_name, temperature=0)
            
            response = model.bind_tools(self.tools).invoke(messages)
            return Result.ok({"messages": [response]})
            
        except Exception as e:
            return Result.from_exception(
                e,
                ErrorType.ROUTING_ERROR,
                context={"state_keys": list(state.keys()) if isinstance(state, dict) else []}
            )

