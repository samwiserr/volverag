"""
LangGraph workflow for agentic RAG.
Following the pattern from: https://docs.langchain.com/oss/python/langgraph/agentic-rag
"""
import logging
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import ToolMessage
from .nodes import generate_query_or_respond, grade_documents, rewrite_question, generate_answer

logger = logging.getLogger(__name__)


def build_rag_graph(tools):
    """
    Build the LangGraph workflow for agentic RAG.
    
    Args:
        tools: List of tools to use (e.g., vector retriever + structured well picks lookup)
        
    Returns:
        Compiled LangGraph workflow
    """
    workflow = StateGraph(MessagesState)
    
    # Create a partial function that includes tools
    def generate_query_with_tool(state: MessagesState):
        return generate_query_or_respond(state, tools)
    
    # Define the nodes we will cycle between
    workflow.add_node("generate_query_or_respond", generate_query_with_tool)
    workflow.add_node("retrieve", ToolNode(tools))
    workflow.add_node("rewrite_question", rewrite_question)
    workflow.add_node("generate_answer", generate_answer)
    
    # Set entry point
    workflow.add_edge(START, "generate_query_or_respond")
    
    # Decide whether to retrieve or respond directly
    workflow.add_conditional_edges(
        "generate_query_or_respond",
        # Assess LLM decision (call retriever tool or respond to user)
        tools_condition,
        {
            # Translate the condition outputs to nodes in our graph
            "tools": "retrieve",
            END: END,
        },
    )
    
    # Edges taken after the retrieve node is called
    workflow.add_conditional_edges(
        "retrieve",
        # Assess agent decision
        grade_documents,
        {
            "generate_answer": "generate_answer",
            "rewrite_question": "rewrite_question",
        }
    )
    
    # After generating answer, end
    workflow.add_edge("generate_answer", END)
    
    # After rewriting question, go back to generate_query_or_respond
    workflow.add_edge("rewrite_question", "generate_query_or_respond")
    
    # Compile the graph
    graph = workflow.compile()
    
    logger.info("[OK] LangGraph workflow compiled successfully")
    return graph

