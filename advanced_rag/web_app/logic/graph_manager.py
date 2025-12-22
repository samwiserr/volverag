"""
Graph manager for initializing and caching the LangGraph RAG workflow.
"""
import logging
from pathlib import Path

import streamlit as st

from src.graph.rag_graph import build_rag_graph
from src.tools.formation_properties_tool import FormationPropertiesTool
from src.tools.eval_params_tool import EvalParamsTool
from src.tools.petro_params_tool import PetroParamsTool
from src.tools.retriever_tool import RetrieverTool
from src.tools.section_lookup_tool import SectionLookupTool
from src.tools.structured_facts_tool import StructuredFactsTool
from src.tools.well_picks_tool import WellPicksTool

logger = logging.getLogger(__name__)


@st.cache_resource
def _get_graph(persist_dir: str, embedding_model: str, cache_version: int = 2):
    """
    Build and cache the RAG graph.
    
    Args:
        persist_dir: Directory containing the vectorstore
        embedding_model: Embedding model name
        cache_version: Increment this to force cache invalidation when code changes
    
    Returns:
        RAG graph or None if vectorstore not found
    """
    rt = RetrieverTool(persist_directory=persist_dir, embedding_model=embedding_model)
    if not rt.load_vectorstore():
        return None  # Return None instead of raising - let UI handle it

    retrieve_tool = rt.get_retriever_tool()

    # Well picks: best-effort file discovery (users can still query via vectorstore too,
    # but deterministic tool is better for exhaustive lists).
    candidates = [
        Path("spwla_volve-main") / "Well_picks_Volve_v1.dat",
        Path("../spwla_volve-main") / "Well_picks_Volve_v1.dat",
        Path("Well_picks_Volve_v1.dat"),
    ]
    dat_path = next((str(p) for p in candidates if p.exists()), str(Path.cwd() / "Well_picks_Volve_v1.dat"))

    # Start with retrieval tool; add WellPicksTool only if data/cache is available
    tools = [retrieve_tool]
    
    # Resolve well_picks_cache_path - it's in the vectorstore directory (from app_assets.zip)
    # Try multiple locations: vectorstore/well_picks_cache.json, or data/well_picks_cache.json
    well_picks_cache_path = Path(persist_dir) / "well_picks_cache.json"  # First try vectorstore directory
    if not well_picks_cache_path.exists():
        # Fallback: try data directory
        data_dir = Path(persist_dir).parent  # Go up from vectorstore to data directory
        well_picks_cache_path = data_dir / "well_picks_cache.json"
        if not well_picks_cache_path.exists():
            # Final fallback: try relative to web_app.py location
            well_picks_cache_path = Path(__file__).resolve().parents[2] / "data" / "well_picks_cache.json"
    
    try:
        well_picks_tool = WellPicksTool(dat_path=dat_path, cache_path=str(well_picks_cache_path))
        if getattr(well_picks_tool, "_rows", None):
            tools.insert(0, well_picks_tool.get_tool())
    except Exception as e:
        logger.warning(f"[WEB_APP] WellPicksTool not available: {e}. Continuing without it.")

    # Optional caches (if present they'll be used for deterministic lookup + citations)
    section_index = Path(persist_dir) / "section_index.json"
    if section_index.exists():
        tools.append(SectionLookupTool(str(section_index)).get_tool())

    petro_cache = Path(persist_dir) / "petro_params_cache.json"
    if petro_cache.exists():
        tools.append(PetroParamsTool(str(petro_cache)).get_tool())

        # One-shot formations + properties tool
        try:
            # Use the same well_picks_cache_path that was used for WellPicksTool
            tools.append(
                FormationPropertiesTool(
                    well_picks_dat_path=dat_path,
                    petro_params_cache_path=str(petro_cache),
                    well_picks_cache_path=str(well_picks_cache_path),  # Use same path as WellPicksTool
                ).get_tool()
            )
        except Exception:
            pass

    # Evaluation parameters tool (uses its own cache)
    eval_cache = Path(persist_dir) / "eval_params_cache.json"
    if eval_cache.exists():
        try:
            tools.append(EvalParamsTool(str(eval_cache)).get_tool())
        except Exception:
            pass

    # Structured facts tool (notes/narrative numeric statements)
    facts_cache = Path(persist_dir) / "facts_cache.json"
    if facts_cache.exists():
        try:
            tools.append(StructuredFactsTool(str(facts_cache)).get_tool())
        except Exception:
            pass

    return build_rag_graph(tools)

