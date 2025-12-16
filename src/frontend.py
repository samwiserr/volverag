"""
Conversational Streamlit frontend for the Volve Wells RAG System with chat interface.
"""

import streamlit as st
import pandas as pd
import json
from pathlib import Path
import sys
import os
import altair as alt

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Load .env file from project root
    project_root = Path(__file__).parent.parent
    env_path = project_root / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass  # python-dotenv not installed, skip

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.main import VolveRAGSystem
from src.visualization import WellLogVisualizer

# Page configuration
st.set_page_config(
    page_title="Volve Wells RAG System",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern chat interface
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        display: flex;
        align-items: flex-start;
    }
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: 20%;
    }
    .assistant-message {
        background: #f0f2f6;
        color: #333;
        margin-right: 20%;
    }
    .message-content {
        flex: 1;
    }
    .message-timestamp {
        font-size: 0.75rem;
        opacity: 0.7;
        margin-top: 0.5rem;
    }
    .stButton>button {
        border-radius: 20px;
        font-weight: 600;
    }
    .source-badge {
        display: inline-block;
        background: #e0e0e0;
        padding: 0.25rem 0.5rem;
        border-radius: 5px;
        font-size: 0.85rem;
        margin: 0.25rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
    st.session_state.index_built = False
    st.session_state.messages = []
    st.session_state.chat_history = []

@st.cache_resource
def initialize_rag_system(rebuild_index=False):
    """Initialize the RAG system with caching."""
    try:
        system = VolveRAGSystem(rebuild_index=rebuild_index)
        # Refresh collection reference after initialization to ensure it's current
        if rebuild_index:
            system.vector_store.refresh_collection()
        return system, True
    except Exception as e:
        import traceback
        error_msg = f"Error initializing RAG system: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        print(error_msg)
        return None, False

def display_message(role: str, content: str, metadata: dict = None):
    """Display a chat message."""
    if role == "user":
        st.markdown(f"""
        <div class="chat-message user-message">
            <div class="message-content">
                <strong>You:</strong><br>
                {content}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message assistant-message">
            <div class="message-content">
                <strong>Assistant:</strong><br>
                {content}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Display sources if available
        if metadata and 'sources' in metadata:
            sources = metadata['sources']
            if sources:
                st.markdown("**Sources:**")
                source_text = " ".join([f'<span class="source-badge">{s.get("well_name", "Unknown")}</span>' 
                                       for s in sources[:5]])
                st.markdown(source_text, unsafe_allow_html=True)
        
        # Display tool results if available
        if metadata and 'tool_results' in metadata:
            tool_results = metadata['tool_results']
            if tool_results:
                for tool_name, result in tool_results.items():
                    if isinstance(result, dict) and result.get('success'):
                        if tool_name == 'plot_formation_log':
                            st.markdown("**📊 Formation Visualization Available**")
                            # Display chart if available
                            chart = result.get('chart')
                            if chart:
                                try:
                                    st.altair_chart(chart, use_container_width=True)
                                except Exception as e:
                                    st.info(f"Chart available for {result.get('well_name')} - {result.get('formation_name')} formation")

def main():
    # Header
    st.markdown('<h1 class="main-header">🛢️ Volve Wells RAG System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Chat with your well data - Ask questions and get answers with visualizations</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ System Settings")
        
        # Initialize system
        if st.session_state.rag_system is None:
            st.info("Click 'Initialize System' to start")
            if st.button("🚀 Initialize System", use_container_width=True):
                with st.spinner("Initializing RAG system..."):
                    system, success = initialize_rag_system(rebuild_index=False)
                    if success:
                        st.session_state.rag_system = system
                        st.session_state.index_built = True
                        st.success("System initialized!")
                        st.rerun()
                    else:
                        st.error("Failed to initialize system. Check console for details.")
        else:
            st.success("✅ System Ready")
            
            # Rebuild index option
            if st.button("🔄 Rebuild Index", use_container_width=True):
                with st.spinner("Rebuilding index..."):
                    system, success = initialize_rag_system(rebuild_index=True)
                    if success:
                        st.session_state.rag_system = system
                        st.success("Index rebuilt!")
                        st.rerun()
        
        st.markdown("---")
        
        # Clear chat
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.rerun()
        
        st.markdown("---")
        
        # Example queries
        st.subheader("💡 Example Queries")
        example_queries = [
            "What is the average porosity in the Hugin formation?",
            "Which well has the highest permeability?",
            "What is the shale volume in well 15/9-F-1?",
            "Show me the Hugin formation in well 15/9-F-1",
            "Compare porosity across all wells"
        ]
        
        for example in example_queries:
            if st.button(example, key=f"example_{example}", use_container_width=True):
                st.session_state.example_query = example
                st.rerun()
        
        st.markdown("---")
        
        # System info
        if st.session_state.rag_system:
            try:
                collection_info = st.session_state.rag_system.vector_store.get_collection_info()
                count = collection_info.get('count', 0)
                if count == 0 or collection_info.get('error'):
                    st.warning("⚠️ Index appears empty or corrupted. Click 'Rebuild Index' to fix.")
                st.metric("Indexed Items", count)
            except Exception as e:
                st.error(f"Error accessing index: {str(e)}")
                st.info("💡 Try clicking 'Rebuild Index' to fix this issue.")
    
    # Main chat interface
    if st.session_state.rag_system is None:
        st.info("👈 Please initialize the system from the sidebar to start chatting.")
        return
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        if not st.session_state.messages:
            st.info("👋 Start a conversation! Ask questions about well data, formations, or request visualizations.")
        else:
            for msg in st.session_state.messages:
                display_message(
                    msg['role'],
                    msg['content'],
                    msg.get('metadata', {})
                )
    
    # Chat input
    st.markdown("---")
    
    # Get query from example or input
    default_query = st.session_state.get('example_query', '')
    if default_query:
        query = st.text_input(
            "💬 Ask a question",
            value=default_query,
            placeholder="e.g., What is the average porosity in the Hugin formation?",
            key="chat_input"
        )
        # Clear example query after using it
        if 'example_query' in st.session_state:
            del st.session_state.example_query
    else:
        query = st.text_input(
            "💬 Ask a question",
            placeholder="e.g., What is the average porosity in the Hugin formation?",
            key="chat_input"
        )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        send_button = st.button("📤 Send", use_container_width=True, type="primary")
    
    # Process query
    if send_button and query:
        # Add user message to chat
        st.session_state.messages.append({
            'role': 'user',
            'content': query,
            'metadata': {}
        })
        
        # Get conversation history for context
        conversation_history = [
            {'role': m['role'], 'content': m['content']} 
            for m in st.session_state.messages[-5:-1]  # Last 5 messages excluding current
        ]
        
        # Execute query
        with st.spinner("Thinking..."):
            try:
                result = st.session_state.rag_system.query(query, n_results=20, 
                                                          conversation_context=conversation_history)
                
                # Add assistant response to chat
                st.session_state.messages.append({
                    'role': 'assistant',
                    'content': result.get('answer', 'No answer generated.'),
                    'metadata': {
                        'sources': result.get('sources', []),
                        'tools_used': result.get('tools_used', []),
                        'tool_results': result.get('tool_results', {}),
                        'aggregated_data': result.get('aggregated_data')
                    }
                })
                
                # Store in chat history
                st.session_state.chat_history.append({
                    'query': query,
                    'answer': result.get('answer'),
                    'sources': result.get('sources', [])
                })
                
            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                # Log full error for debugging
                print(f"Error processing query: {type(e).__name__}: {str(e)}")
                print(f"Traceback:\n{error_details}")
                
                # Show user-friendly error message
                error_type = type(e).__name__
                error_msg = f"Error processing query: {error_type}"
                if str(e):
                    error_msg += f" - {str(e)}"
                
                st.session_state.messages.append({
                    'role': 'assistant',
                    'content': error_msg,
                    'metadata': {'error': str(e), 'error_type': error_type}
                })
                st.error(error_msg)
                # Show detailed error in expander for debugging
                with st.expander("Error Details (for debugging)"):
                    st.code(error_details)
        
        st.rerun()
    
    # Handle visualization requests
    if st.session_state.messages:
        last_msg = st.session_state.messages[-1]
        if last_msg['role'] == 'assistant':
            metadata = last_msg.get('metadata', {})
            tool_results = metadata.get('tool_results', {})
            
            # Check if there's a formation visualization available
            if 'plot_formation_log' in tool_results:
                result = tool_results['plot_formation_log']
                if result.get('success'):
                    well_name = result.get('well_name')
                    formation_name = result.get('formation_name')
                    
                    st.markdown("---")
                    st.subheader(f"📊 Formation Visualization: {formation_name} in {well_name}")
                    
                    chart = result.get('chart')
                    if chart:
                        try:
                            st.altair_chart(chart, use_container_width=True)
                        except Exception as e:
                            st.info(f"Visualization available for {formation_name} formation in {well_name}")
                    
                    # Show formation details
                    if result.get('formation_top'):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Formation Top (MD)", f"{result['formation_top']:.2f} m")
                        with col2:
                            if result.get('formation_base'):
                                st.metric("Formation Base (MD)", f"{result['formation_base']:.2f} m")
                        with col3:
                            if result.get('formation_base'):
                                thickness = result['formation_base'] - result['formation_top']
                                st.metric("Thickness", f"{thickness:.2f} m")
                    
                    # Show curves plotted
                    if result.get('curves'):
                        st.info(f"**Curves displayed:** {', '.join(result['curves'])}")

if __name__ == "__main__":
    main()
