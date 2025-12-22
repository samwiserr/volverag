"""
Base routing strategy interface.
"""
from abc import ABC, abstractmethod
from typing import Optional
from langchain_core.messages import AIMessage
from src.normalize.query_normalizer import NormalizedQuery
from src.core.result import Result, AppError, ErrorType

class RoutingStrategy(ABC):
    """
    Abstract base class for routing strategies.
    
    Each strategy determines whether a query should route to a specific tool
    based on query characteristics (well, formation, property, keywords, etc.).
    """
    
    @property
    @abstractmethod
    def priority(self) -> int:
        """
        Priority of this strategy (lower = higher priority).
        
        Strategies are evaluated in priority order. The first strategy
        that matches should handle the routing.
        """
        pass
    
    @abstractmethod
    def should_route(
        self,
        question: str,
        normalized_query: NormalizedQuery,
        tool_query: str,
        persist_dir: str,
    ) -> bool:
        """
        Determine if this strategy should handle the query.
        
        Args:
            question: Original user question
            normalized_query: Normalized query with extracted entities
            tool_query: Query string to pass to tools
            persist_dir: Path to vectorstore directory
            
        Returns:
            True if this strategy should handle routing
        """
        pass
    
    @abstractmethod
    def route(
        self,
        question: str,
        normalized_query: NormalizedQuery,
        tool_query: str,
        persist_dir: str,
    ) -> Result[AIMessage, AppError]:
        """
        Route the query to appropriate tool(s).
        
        Args:
            question: Original user question
            normalized_query: Normalized query with extracted entities
            tool_query: Query string to pass to tools
            persist_dir: Path to vectorstore directory
            
        Returns:
            Result containing AIMessage with tool calls, or error
        """
        pass
    
    def _create_tool_call_message(
        self,
        tool_name: str,
        query: str,
        call_id: Optional[str] = None,
        additional_tools: Optional[list] = None,
    ) -> AIMessage:
        """
        Helper to create AIMessage with tool calls.
        
        Args:
            tool_name: Name of primary tool to call
            query: Query string for tool
            call_id: Optional call ID (auto-generated if not provided)
            additional_tools: Optional list of additional tool calls
            
        Returns:
            AIMessage with tool calls
        """
        import uuid
        
        tool_calls = [
            {
                "name": tool_name,
                "args": {"query": query},
                "id": call_id or f"call_{tool_name}_{uuid.uuid4().hex[:8]}",
            }
        ]
        
        if additional_tools:
            tool_calls.extend(additional_tools)
        
        return AIMessage(content="", tool_calls=tool_calls)

