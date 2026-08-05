"""
Modern UI Components for the VolveRAG Application.
These components provide a sleek, professional interface with enhanced UX.
"""

import streamlit as st
from typing import Optional, List, Dict, Any


def render_header(title: str = "VolveRAG", subtitle: Optional[str] = None):
    """Render a modern header with gradient styling."""
    st.markdown(
        f"""
        <style>
        .modern-header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-size: 3rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }}
        .modern-subtitle {{
            color: #6c757d;
            font-size: 1.2rem;
            margin-bottom: 2rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown(f'<h1 class="modern-header">{title}</h1>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<p class="modern-subtitle">{subtitle}</p>', unsafe_allow_html=True)


def render_chat_message(role: str, content: str, citations: Optional[List[Dict]] = None, idx: int = 0):
    """Render a modern chat message with improved styling."""
    is_user = role == "user"
    
    avatar = "👤" if is_user else "🤖"
    bg_color = "#e3f2fd" if is_user else "#f5f5f5"
    border_color = "#2196f3" if is_user else "#e0e0e0"
    
    # Clean source lines from content for display
    import re
    clean_content = re.sub(r"(?m)^\s*Source:\s*.*\n?", "", content).strip() if isinstance(content, str) else content
    
    st.markdown(
        f"""
        <style>
        .chat-message-{idx} {{
            padding: 1.5rem;
            border-radius: 12px;
            margin: 1rem 0;
            border-left: 4px solid {border_color};
            background: {bg_color};
        }}
        .message-role {{
            font-weight: 600;
            color: #1a237e;
            margin-bottom: 0.5rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        .message-content {{
            line-height: 1.6;
            color: #262730;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    with st.container():
        st.markdown(
            f"""
            <div class="chat-message-{idx}">
                <div class="message-role">
                    <span>{avatar}</span>
                    <span>{"You" if is_user else "VolveRAG Assistant"}</span>
                </div>
                <div class="message-content">
                    {clean_content}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_citation_card(source_path: str, page_start: Optional[int], page_end: Optional[int], on_click=None, key: str = ""):
    """Render a modern citation card with click functionality."""
    from .logic.citation_parser import _clean_source_path
    
    clean_path = _clean_source_path(source_path)
    page_info = ""
    if page_start and page_end:
        page_info = f"Pages {page_start}-{page_end}"
    elif page_start:
        page_info = f"Page {page_start}"
    
    col1, col2 = st.columns([0.75, 0.25])
    
    with col1:
        st.markdown(
            f"""
            <style>
            .citation-card {{
                background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                padding: 1rem;
                border-radius: 8px;
                border-left: 3px solid #667eea;
                margin: 0.5rem 0;
            }}
            .citation-path {{
                font-family: 'Courier New', monospace;
                font-size: 0.9rem;
                color: #495057;
            }}
            .citation-page {{
                font-size: 0.85rem;
                color: #6c757d;
                margin-top: 0.25rem;
            }}
            </style>
            <div class="citation-card">
                <div class="citation-path">📄 {clean_path}</div>
                <div class="citation-page">{page_info}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    with col2:
        label = f"View p.{page_start}" if page_start else "View"
        if st.button(label, key=key, use_container_width=True):
            if on_click:
                on_click(source_path, page_start)


def render_sota_panel():
    """Render a modern SOTA techniques status panel."""
    import os
    
    st.markdown(
        """
        <style>
        .sota-panel {
            background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
            padding: 1.5rem;
            border-radius: 12px;
            margin: 1rem 0;
        }
        .sota-title {
            font-weight: 700;
            color: #1a237e;
            margin-bottom: 1rem;
            font-size: 1.1rem;
        }
        .sota-item {
            display: inline-block;
            padding: 0.5rem 1rem;
            margin: 0.25rem;
            border-radius: 20px;
            font-size: 0.9rem;
            background: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .status-active {
            color: #2e7d32;
            border: 2px solid #4caf50;
        }
        .status-inactive {
            color: #757575;
            border: 2px solid #e0e0e0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    def _env_on(key: str, default: str = "true") -> bool:
        return os.getenv(key, default).lower() in {"1", "true", "yes"}
    
    st.markdown('<div class="sota-panel">', unsafe_allow_html=True)
    st.markdown('<div class="sota-title">⚡ SOTA Techniques Status</div>', unsafe_allow_html=True)
    
    # Index-time techniques
    st.markdown("**Index-time** *(baked into vectorstore)*")
    cols = st.columns(2)
    
    contextual_on = _env_on("RAG_CONTEXTUAL")
    raptor_on = _env_on("RAG_RAPTOR")
    
    with cols[0]:
        status_class = "status-active" if contextual_on else "status-inactive"
        icon = "🟢" if contextual_on else "⚪"
        st.markdown(
            f'<span class="sota-item {status_class}">{icon} Contextual</span>',
            unsafe_allow_html=True,
        )
    
    with cols[1]:
        status_class = "status-active" if raptor_on else "status-inactive"
        icon = "🟢" if raptor_on else "⚪"
        st.markdown(
            f'<span class="sota-item {status_class}">{icon} RAPTOR</span>',
            unsafe_allow_html=True,
        )
    
    st.divider()
    
    # Query-time techniques
    st.markdown("**Query-time** *(active every query)*")
    
    hyde_default = _env_on("RAG_HYDE")
    if "hyde_enabled" not in st.session_state:
        st.session_state.hyde_enabled = hyde_default
    
    hyde_toggle = st.toggle(
        "🎯 HyDE — Hypothetical Doc Embeddings",
        value=st.session_state.hyde_enabled,
        help="Generates a hypothetical ideal answer passage for each query and embeds it alongside the raw query for better dense retrieval. Adds ~1s latency.",
    )
    
    if hyde_toggle != st.session_state.hyde_enabled:
        st.session_state.hyde_enabled = hyde_toggle
        os.environ["RAG_HYDE"] = "true" if hyde_toggle else "false"
    
    fusion = os.getenv("RAG_HYBRID_FUSION", "rrf").upper()
    cross_encoder_on = _env_on("RAG_USE_CROSS_ENCODER")
    llm_rerank_on = _env_on("RAG_RERANK", "llm")
    
    tech_cols = st.columns(3)
    with tech_cols[0]:
        st.markdown(
            f'<span class="sota-item status-active">🟢 RRF Fusion ({fusion})</span>',
            unsafe_allow_html=True,
        )
    with tech_cols[1]:
        status_class = "status-active" if cross_encoder_on else "status-inactive"
        icon = "🟢" if cross_encoder_on else "⚪"
        st.markdown(
            f'<span class="sota-item {status_class}">{icon} Cross-Encoder</span>',
            unsafe_allow_html=True,
        )
    with tech_cols[2]:
        status_class = "status-active" if llm_rerank_on else "status-inactive"
        icon = "🟢" if llm_rerank_on else "⚪"
        st.markdown(
            f'<span class="sota-item {status_class}">{icon} LLM Rerank</span>',
            unsafe_allow_html=True,
        )
    
    st.markdown('</div>', unsafe_allow_html=True)


def render_pdf_viewer_placeholder():
    """Render a modern PDF viewer placeholder."""
    st.markdown(
        """
        <style>
        .pdf-placeholder {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            padding: 3rem;
            border-radius: 12px;
            text-align: center;
            border: 2px dashed #ced4da;
        }
        .pdf-icon {
            font-size: 4rem;
            margin-bottom: 1rem;
        }
        .pdf-text {
            color: #6c757d;
            font-size: 1.1rem;
        }
        </style>
        <div class="pdf-placeholder">
            <div class="pdf-icon">📄</div>
            <div class="pdf-text">Click <strong>"View page"</strong> next to a source to open the PDF here</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_loading_spinner(message: str = "Thinking..."):
    """Render a modern loading spinner with custom message."""
    st.markdown(
        """
        <style>
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        .loading-container {
            display: flex;
            align-items: center;
            gap: 1rem;
            padding: 1.5rem;
            background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
            border-radius: 12px;
            margin: 1rem 0;
        }
        .loading-spinner {
            width: 24px;
            height: 24px;
            border: 3px solid #667eea;
            border-top: 3px solid transparent;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .loading-text {
            color: #1a237e;
            font-weight: 500;
            font-size: 1.1rem;
        }
        </style>
        <div class="loading-container">
            <div class="loading-spinner"></div>
            <div class="loading-text">""" + message + """</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_example_queries():
    """Render example query chips for quick interaction."""
    examples = [
        "What is the water saturation value of Hugin formation in 15/9-F-5?",
        "Tell me about the porosity in well 15/9-F-4",
        "What are the petrophysical parameters for Heather formation?",
        "Describe the reservoir properties of Statfjord formation",
    ]
    
    st.markdown(
        """
        <style>
        .example-chip {
            display: inline-block;
            padding: 0.5rem 1rem;
            margin: 0.25rem;
            background: white;
            border: 1px solid #667eea;
            border-radius: 20px;
            color: #667eea;
            cursor: pointer;
            transition: all 0.3s ease;
            font-size: 0.9rem;
        }
        .example-chip:hover {
            background: #667eea;
            color: white;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(102, 126, 234, 0.3);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown("**💡 Try asking:**")
    cols = st.columns(2)
    
    for i, example in enumerate(examples):
        col_idx = i % 2
        with cols[col_idx]:
            if st.button(example, key=f"example_{i}", use_container_width=True):
                st.session_state.example_query = example
                st.rerun()


def render_debug_expander(result: Dict[str, Any], user_input: str):
    """Render a modern debug information expander."""
    with st.expander("🔍 Debug Information", expanded=False):
        st.markdown(
            """
            <style>
            .debug-section {
                background: #f8f9fa;
                padding: 1rem;
                border-radius: 8px;
                margin: 0.5rem 0;
                border-left: 3px solid #667eea;
            }
            .debug-label {
                font-weight: 600;
                color: #1a237e;
                margin-bottom: 0.5rem;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        
        st.markdown(f'<div class="debug-label">Query:</div>', unsafe_allow_html=True)
        st.code(user_input, language="text")
        
        # HyDE document
        if st.session_state.get("last_hyde_doc"):
            st.markdown(f'<div class="debug-label">HyDE Hypothetical Document:</div>', unsafe_allow_html=True)
            st.info(st.session_state.last_hyde_doc)
            st.session_state.last_hyde_doc = None
        
        # Petro cache status
        try:
            from pathlib import Path
            vectorstore_dir = Path(__file__).resolve().parents[2] / "data" / "vectorstore"
            cache_path = vectorstore_dir / "petro_params_cache.json"
            cache_exists = cache_path.exists()
            
            st.markdown(f'<div class="debug-label">Petro Cache Status:</div>', unsafe_allow_html=True)
            status_icon = "✅" if cache_exists else "❌"
            st.write(f"{status_icon} Cache exists: {cache_exists}")
            if cache_exists:
                st.caption(f"Path: `{cache_path}`")
        except Exception as e:
            st.error(f"Error checking cache: {e}")
        
        # Normalization info
        try:
            from src.normalize.query_normalizer import normalize_query, extract_well
            nq = normalize_query(user_input)
            extracted_well = extract_well(user_input)
            
            st.markdown(f'<div class="debug-label">Query Normalization:</div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Extracted Well:** `{extracted_well}`")
                st.write(f"**Normalized Well:** `{nq.well}`")
            with col2:
                st.write(f"**Formation:** `{nq.formation}`")
                st.write(f"**Property:** `{nq.property}`")
        except Exception as e:
            st.write(f"**Error extracting well:** {e}")
        
        # Routing analysis
        st.divider()
        st.markdown(f'<div class="debug-label">Routing Analysis:</div>', unsafe_allow_html=True)
        
        ql = user_input.lower()
        param_keywords = ["petrophysical parameters", "net to gross", "porosity", "water saturation", "permeability"]
        has_param_keyword = any(k in ql for k in param_keywords)
        
        st.json({
            "has_param_keyword": has_param_keyword,
            "query_lower": ql[:100] + "..." if len(ql) > 100 else ql
        })
        
        # Tool calls
        tool_calls = []
        for msg in result.get("messages", []):
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls.append(tc.get("name", "unknown"))
        
        st.markdown(f'<div class="debug-label">Tools Called:</div>', unsafe_allow_html=True)
        if tool_calls:
            st.success(f"✅ {', '.join(tool_calls)}")
        else:
            st.info("No tools called (direct LLM response)")
