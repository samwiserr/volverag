"""
LangGraph nodes for agentic RAG workflow.
Following the pattern from: https://docs.langchain.com/oss/python/langgraph/agentic-rag
"""
import os
import re
import logging
from pathlib import Path
from typing import Literal, Optional, List
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from ..normalize.query_normalizer import normalize_query
from ..normalize.property_registry import default_registry, resolve_property_deterministic, PropertyEntry
from ..normalize.agent_disambiguator import choose_property_with_agent
from ..normalize.query_normalizer import extract_well, normalize_formation
from ..normalize.entity_resolver import resolve_with_bounded_agent
from ..query.incomplete_query_handler import is_incomplete_query, complete_incomplete_query
from ..query.query_decomposer import decompose_query, expand_query_synonyms

logger = logging.getLogger(__name__)

# Lazy-init chat models (avoid import-time crash if OPENAI_API_KEY not set)
_response_model: Optional[ChatOpenAI] = None
_grader_model: Optional[ChatOpenAI] = None


def _get_response_model() -> ChatOpenAI:
    global _response_model
    if _response_model is None:
        model = os.getenv("OPENAI_MODEL", "gpt-4o")
        _response_model = ChatOpenAI(model=model, temperature=0)
    return _response_model


def _get_grader_model() -> ChatOpenAI:
    global _grader_model
    if _grader_model is None:
        model = os.getenv("OPENAI_GRADE_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o"))
        _grader_model = ChatOpenAI(model=model, temperature=0)
    return _grader_model

# Initialize property registry (lazy-loaded)
_registry_cache: Optional[List[PropertyEntry]] = None

def _get_registry() -> List[PropertyEntry]:
    """Lazy-load the property registry."""
    global _registry_cache
    if _registry_cache is None:
        _registry_cache = default_registry("./data/vectorstore")
    return _registry_cache


def _latest_user_question(messages) -> str:
    """
    Multi-turn support: use the most recent HumanMessage as the "current question".
    """
    try:
        from langchain_core.messages import HumanMessage

        for m in reversed(messages or []):
            if isinstance(m, HumanMessage) and isinstance(getattr(m, "content", None), str):
                return m.content
    except Exception:
        pass
    # Fallback (single-turn behavior)
    try:
        if messages and isinstance(messages[-1].get("content"), str):
            return messages[-1]["content"]
    except Exception:
        pass
    try:
        return messages[0].content if messages else ""
    except Exception:
        return ""


def _iter_message_texts(messages):
    """
    Yield (role, content) for both dict-style and LangChain message objects.
    """
    for m in messages or []:
        # dict-style
        if isinstance(m, dict):
            role = m.get("role")
            content = m.get("content")
            if isinstance(content, str):
                yield role, content
            continue
        # LangChain message objects
        role = getattr(m, "type", None) or getattr(m, "__class__", type("x", (), {})).__name__
        content = getattr(m, "content", None)
        if isinstance(content, str):
            yield str(role), content


def _infer_recent_context(messages) -> tuple[Optional[str], Optional[str]]:
    """
    Infer (well, formation) from recent conversation history.
    This enables follow-ups like "matrix density" after a prior turn specifying well/formation.
    """
    well = None
    formation = None
    # Scan backward through recent messages
    for role, txt in reversed(list(_iter_message_texts(messages))[-25:]):
        # Ignore tool messages (they may contain many formations/wells from retrieved context)
        if str(role).lower() in {"tool", "toolmessage"}:
            continue
        if well is None:
            w = extract_well(txt)
            if w:
                well = w
        if formation is None:
            f = normalize_formation(txt)
            if f:
                formation = f
        if well and formation:
            break
    return well, formation


class GradeDocuments(BaseModel):
    """Grade documents using a binary score for relevance check."""
    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )


GRADE_PROMPT = (
    "You are a grader assessing relevance of retrieved documents to a user question. \n "
    "Here are the retrieved documents: \n\n {context} \n\n"
    "Here is the user question: {question} \n"
    "IMPORTANT: Be VERY lenient. If the documents contain:\n"
    "- The well name mentioned in the question\n"
    "- The word 'model' or 'evaluation' or related terms\n"
    "- Any text that appears to be a continuation or completion of the question\n"
    "- Any information about the well, formation, or evaluation\n"
    "Then mark it as relevant (yes).\n"
    "Only mark as not relevant (no) if the documents are completely unrelated (e.g., about a different well with no connection).\n"
    "Give a binary score 'yes' or 'no'."
)

REWRITE_PROMPT = (
    "You are a question rewriter. Given the following question and context, "
    "rewrite the question to be more specific and retrieval-friendly. "
    "If the context is not relevant, ask a better question based on the original question. \n"
    "Original question: {question} \n"
    "Context: {context} \n"
    "Rewritten question:"
)

GENERATE_PROMPT = (
    "You are an assistant for question-answering tasks about petrophysical documents. "
    "Use the following pieces of retrieved context to answer the question accurately and COMPLETELY. "
    "\n"
    "IMPORTANT INSTRUCTIONS:\n"
    "1. If the question is a partial sentence or incomplete quote, find the EXACT completion in the context.\n"
    "2. Extract the exact text that completes or answers the question - do not paraphrase unless necessary.\n"
    "3. If you find text that starts with the question text, provide the complete sentence or paragraph.\n"
    "4. For DEPTH questions (e.g., 'depth of X formation in well Y'):\n"
    "   - Search the context CAREFULLY for the formation name - look for EXACT matches and variations:\n"
    "     * 'Sleipner Fm.' or 'Sleipner Fm. Top' or 'Sleipner Fm. Base'\n"
    "     * 'Sleipner Formation' or 'Sleipner'\n"
    "     * The formation name may appear in a list format like '- Sleipner Fm. Top: MD X m, TVD Y m, TVDSS Z m'\n"
    "   - Extract ALL depth information: MD (Measured Depth), TVD (True Vertical Depth), TVDSS (True Vertical Depth Sub Sea)\n"
    "   - Include depth ranges if the formation has a top and base\n"
    "   - Specify which well the depth refers to\n"
    "   - Format example: 'Sleipner Fm. in well 15/9-19A: Top at MD X m, TVD Y m, TVDSS Z m; Base at MD A m, TVD B m, TVDSS C m'\n"
    "   - If only one depth is given, specify if it's Top or Base\n"
    "   - Include all available depth measurements, not just one type\n"
    "   - READ THE ENTIRE CONTEXT - formation data may be in a structured list format\n"
    "   - If you find the formation name anywhere in the context, extract its depth information\n"
    "   - If the formation is truly not found after searching the entire context, state that clearly\n"
    "5. For questions about 'all wells', 'each well', 'list all', or 'every well':\n"
    "   - You MUST include information about EVERY well mentioned in the context\n"
    "   - Do NOT skip any wells - if the context has 35 wells, list all 35\n"
    "   - Organize by well name for clarity\n"
    "6. If the context contains formation picks data (Well_picks_Volve_v1.dat), use it to provide comprehensive formation lists.\n"
    "7. Be precise and cite specific details from the documents.\n"
    "8. Do not make up information - only use what's in the context.\n"
    "9. If the context contains information about multiple wells, ensure your answer is for the CORRECT well mentioned in the question.\n"
    "10. For listing queries, be exhaustive - include every item mentioned in the context.\n"
    "11. Well names may appear in different formats (15/9-19A, 15_9-19A, 15-9-19A) - treat them as the same well.\n"
    "12. If the context includes structured well picks lookup output (prefixed with '[WELL_PICKS]'), treat it as authoritative.\n"
    "\n"
    "Question: {question} \n"
    "Context: {context}\n"
    "\n"
    "Answer (extract the exact information that completes or answers the question COMPLETELY):"
)


def generate_query_or_respond(state: MessagesState, tools):
    """
    Call the model to generate a response based on the current state.
    Given the question, it will decide to retrieve using the retriever tool,
    or simply respond to the user.
    
    Args:
        state: MessagesState containing messages
        tools: Tools to bind to the model (e.g., vector retriever + structured lookup)
        
    Returns:
        Dictionary with 'messages' key containing the model response
    """
    # Deterministic routing: for "complete list ... formations ... each/all wells" queries,
    # force the structured well-picks tool to run (no embeddings / no LLM summarization).
    try:
        question = _latest_user_question(state.get("messages"))
        ql = question.lower() if isinstance(question, str) else ""
        
        # Store original question for "all wells" detection (before any modifications)
        original_question = question if isinstance(question, str) else ""
        
        # Phase 1.5: Handle incomplete queries
        completed_query = None
        if isinstance(question, str) and os.getenv("RAG_ENABLE_QUERY_COMPLETION", "true").lower() in {"1", "true", "yes"}:
            if is_incomplete_query(question):
                logger.info(f"[QUERY_COMPLETION] Detected incomplete query: '{question}'")
                completed_queries = complete_incomplete_query(question, max_variations=3)
                if completed_queries and completed_queries != [question]:
                    # Use the first completed query as the primary query
                    completed_query = completed_queries[0]
                    logger.info(f"[QUERY_COMPLETION] Completed query: '{completed_query}'")
                    # Update question for processing
                    question = completed_query

        # Phase 3: Query decomposition and rewriting
        # BUT: Skip decomposition for "all wells" queries to preserve original query text
        decomposed_sub_queries = []
        is_all_wells_query = (
            ("formation" in ql or "formations" in ql)
            and ("properties" in ql or "petrophysical" in ql)
            and any(k in ql for k in ["all", "each", "every", "complete", "entire", "list all"])
        )
        
        if not is_all_wells_query and isinstance(question, str) and os.getenv("RAG_ENABLE_QUERY_DECOMPOSITION", "true").lower() in {"1", "true", "yes"}:
            is_complex, sub_queries, rewritten = decompose_query(question)
            if is_complex and sub_queries:
                decomposed_sub_queries = sub_queries
                # Use first sub-query as primary, but store all for multi-step retrieval
                question = sub_queries[0] if sub_queries else question
                logger.info(f"[QUERY_DECOMPOSITION] Using decomposed query: '{question}' (from {len(sub_queries)} sub-queries)")
            elif rewritten != question:
                question = rewritten
                logger.info(f"[QUERY_DECOMPOSITION] Using rewritten query: '{question}'")

        # Normalize -> Resolve (deterministic, runs before any retrieval)
        nq = normalize_query(question if isinstance(question, str) else "")
        if (not nq.well) or (not nq.formation):
            cw, cf = _infer_recent_context(state.get("messages"))
            if not nq.well and cw:
                nq = nq.__class__(raw=nq.raw, well=cw, formation=nq.formation, property=nq.property, tool=nq.tool, intent=nq.intent)
            if not nq.formation and cf:
                nq = nq.__class__(raw=nq.raw, well=nq.well, formation=cf, property=nq.property, tool=nq.tool, intent=nq.intent)

        # Smart next-step: bounded agentic resolver (well+formation+property) for heavy-typo inputs.
        # Only invoke when deterministic parse is missing key entities for a fact-like query.
        if isinstance(question, str):
            looks_facty = any(t in ql for t in ["dens", "rho", "archie", "gr", "net", "gross", "phif", "sw", "klogh", "rw", "temperature"])
            if looks_facty and (nq.well is None or nq.property is None or nq.formation is None):
                try:
                    er = resolve_with_bounded_agent(question, persist_dir="./data/vectorstore")
                    if er.needs_clarification and er.clarification_question:
                        return {"messages": [AIMessage(content=er.clarification_question)]}
                    # CRITICAL: If still no well after resolution, ask for clarification
                    if not er.well and not nq.well:
                        return {"messages": [AIMessage(content="I couldn't identify which well you're asking about. Please specify the well name clearly (e.g., 15/9-F-5, 19/9-19 bt2).")]}
                    if er.well or er.formation or er.property:
                        # Update nq with resolved entities
                        nq = nq.__class__(
                            raw=nq.raw,
                            well=er.well or nq.well,
                            formation=er.formation or nq.formation,
                            property=er.property or nq.property,
                            tool=er.tool or nq.tool,
                            intent="fact" if (er.well or nq.well) and (er.property or nq.property) else nq.intent,
                        )
                except Exception:
                    # If resolver fails and we still don't have a well for a fact query, ask for clarification
                    if looks_facty and not nq.well:
                        return {"messages": [AIMessage(content="I couldn't identify which well you're asking about. Please specify the well name clearly (e.g., 15/9-F-5, 19/9-19 bt2).")]}
                    pass

        # Build an "effective query" for tools so follow-ups like "matrix density"
        # still include well/formation context inferred from chat history.
        tool_query = question if isinstance(question, str) else ""
        tool_query = tool_query.strip()
        if nq.well and extract_well(tool_query) is None:
            tool_query = (tool_query + f" in well {nq.well}").strip()
        if nq.formation and nq.formation.lower() not in tool_query.lower():
            tool_query = (tool_query + f" formation {nq.formation}").strip()

        def _tool_call(name: str, call_id: str):
            return {"name": name, "args": {"query": tool_query}, "id": call_id}

        # Deterministic routing: Depth queries (MD/TVD/TVDSS) -> structured well picks
        # This prevents fuzzy property resolution from misrouting to evaluation parameters.
        # Check if ANY well is detected (not just hardcoded "15" and "9")
        is_depth_query = (
            (nq.well is not None or extract_well(question) is not None)  # well detected
            and any(k in ql for k in ["depth", "md", "tvd", "tvdss", "measured depth", "true vertical depth", "depth of"])
            and (nq.formation or "formation" in ql)  # we expect a formation context
        )
        if is_depth_query:
            forced = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "lookup_well_picks",
                        "args": {"query": tool_query},
                        "id": "call_lookup_well_picks_depth_1",
                    }
                ],
            )
            return {"messages": [forced]}

        # For fact-like queries, do typo-tolerant property resolution:
        # - fuzzy match over registry
        # - if ambiguous, ask a clarification question (agent)
        if nq.well and (("density" in ql) or ("rho" in ql) or ("archie" in ql) or ("gr" in ql) or ("dens" in ql) or ("permeability" in ql) or ("permeab" in ql) or ("perm" in ql) or ("porosity" in ql) or ("poro" in ql) or ("phi" in ql) or re.search(r'\bk\b', ql, re.IGNORECASE) or nq.intent == "fact"):
            registry = _get_registry()
            # Start from normalized property if present, otherwise resolve with fuzzy
            entry = next((e for e in registry if e.canonical == nq.property), None) if nq.property else None
            
            resolved_entry, candidates = (entry, []) if entry else resolve_property_deterministic(question, registry)

            # Constrain ambiguity candidates by strong intent signals to avoid wrong picks.
            # (This is the "smart" pattern: bounded agent over correct candidate set.)
            if ("density" in ql or "rho" in ql or "dens" in ql):
                density_cands = [e for e in registry if e.canonical in {"matrix_density", "fluid_density"}]
                # If the resolver picked a non-density property (common failure mode on typo-y inputs),
                # treat it as ambiguous and force clarification between the density choices.
                if resolved_entry is not None and resolved_entry.canonical not in {"matrix_density", "fluid_density"}:
                    resolved_entry = None
                if resolved_entry is None and density_cands:
                    candidates = density_cands

            # Ambiguous: ask for clarification, do not run tools.
            if resolved_entry is None and candidates and len(candidates) >= 2:
                dis = choose_property_with_agent(question, candidates)
                if dis.clarification_question:
                    return {"messages": [AIMessage(content=dis.clarification_question)]}
                # If agent chose anyway, proceed
                if dis.canonical:
                    resolved_entry = next((e for e in registry if e.canonical == dis.canonical), None)
                # If the agent failed to respond cleanly, fall back to a deterministic clarification.
                if resolved_entry is None and ("density" in ql or "rho" in ql):
                    # We know the candidate set is bounded (matrix vs fluid density)
                    opts = []
                    for c in candidates:
                        if c.canonical == "matrix_density":
                            opts.append("matrix density (ρma)")
                        elif c.canonical == "fluid_density":
                            opts.append("fluid density (ρfl)")
                    if len(opts) >= 2:
                        return {"messages": [AIMessage(content=f"Did you mean {opts[0]} or {opts[1]}?")]}

            tool_calls = []
            if resolved_entry and nq.well:
                tool_calls.append(_tool_call(resolved_entry.tool, f"call_{resolved_entry.tool}_norm_1"))

            # Add retriever as fallback only when we did not route to a deterministic structured tool.
            # This avoids expensive embeddings for purely-structured lookups.
            if not tool_calls or (resolved_entry and resolved_entry.tool not in {"lookup_evaluation_parameters", "lookup_petrophysical_params", "lookup_structured_facts"}):
                tool_calls.append(_tool_call("retrieve_petrophysical_docs", "call_retrieve_petrophysical_docs_norm_1"))

            if tool_calls:
                forced = AIMessage(content="", tool_calls=tool_calls)
                return {"messages": [forced]}
        # Deterministic routing: for section-like queries, force section lookup tool
        # Examples: "Summary 15/9-F-11 / 15/9-F-11 T2", "Introduction 15/9-F-11..."
        is_section_like = (
            any(k in ql for k in ["summary", "introduction", "conclusion", "results", "discussion", "abstract"])
            and "15" in ql and "9" in ql
        )
        if is_section_like:
            forced = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "lookup_section",
                        "args": {"query": tool_query},
                        "id": "call_lookup_section_1",
                    }
                ],
            )
            return {"messages": [forced]}

        # Deterministic routing: Petrophysical parameters table (Net/Gross, PHIF, SW, KLOGH) by well/formation/parameter
        is_param_query = (
            (any(k in ql for k in ["petrophysical parameters", "petrophysical parameter", "net to gross", "net-to-gross", "netgros", "net/gross", "ntg", "n/g", "phif", "phi", "poro", "porosity", "water saturation", " sw", "klogh", "permeability", "permeab", "perm"]) or re.search(r'\bk\b', ql, re.IGNORECASE))
            and "15" in ql and "9" in ql
        )
        if is_param_query:
            forced = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "lookup_petrophysical_params",
                        "args": {"query": tool_query},
                        "id": "call_lookup_petrophysical_params_1",
                    }
                ],
            )
            return {"messages": [forced]}

        # Check for specific well name FIRST before checking wants_all_wells
        # This prevents "all formations in Well NO 15/9-F-15 A" from matching wants_all_wells
        # Import well extraction function for direct check
        try:
            from ..tools.well_picks_tool import _extract_query_well
            extracted_well = _extract_query_well(question)
        except Exception:
            extracted_well = None
        
        has_specific_well = (
            nq.well or 
            extracted_well is not None or  # Direct well extraction check (handles "A" suffix)
            extract_well(question) is not None or  # Generic well extraction
            ("well" in ql and re.search(r"\d+[\s_/-]*\d+", ql))  # Any well pattern (XX/YY or XX-YY)
        )
        
        # Deterministic routing: one-shot "formations + petrophysical properties"
        # Check for "all formations and properties" queries first (no well required)
        # Improved detection to handle "list all well formations and their properties"
        is_all_formations_properties = (
            ("formation" in ql or "formations" in ql)
            and ("properties" in ql or "petrophysical" in ql)
            and any(k in ql for k in ["all", "each", "every", "complete", "entire", "list all"])
            and not has_specific_well  # No specific well mentioned
        ) or (
            # Also match "list all well formations" pattern
            ("list" in ql and "all" in ql)
            and ("formation" in ql or "formations" in ql)
            and ("properties" in ql or "petrophysical" in ql)
            and not has_specific_well
        )
        if is_all_formations_properties:
            logger.info(f"[ROUTING] Routing to formation properties tool for ALL wells")
            # Use original question for "all wells" queries to preserve detection keywords like "every", "all", etc.
            all_wells_query = original_question if original_question else tool_query
            logger.info(f"[ROUTING] Using original query for tool: '{all_wells_query}'")
            forced = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "lookup_formation_properties",
                        "args": {"query": all_wells_query},
                        "id": "call_lookup_formation_properties_all_wells_1",
                    }
                ],
            )
            return {"messages": [forced]}
        
        # For queries like "complete list formations present in 15/9-F-4 and their petrophysical properties"
        is_formation_properties = (
            ("formation" in ql or "formations" in ql)
            and any(k in ql for k in ["petrophysical", "properties", "parameters", "porosity", "phif", "sw", "net to gross", "net-to-gross", "klogh"])
            and has_specific_well  # Specific well mentioned
        )
        if is_formation_properties:
            logger.info(f"[ROUTING] Routing to formation properties tool (petrophysical properties requested)")
            forced = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "lookup_formation_properties",
                        "args": {"query": tool_query},
                        "id": "call_lookup_formation_properties_1",
                    }
                ],
            )
            return {"messages": [forced]}

        # Deterministic routing: "Evaluation parameters" tables (Rhoma/Rhofl/GRmin/GRmax/Archie a,n,m, etc.)
        # Users often ask for these values without saying "evaluation parameters", so we key off parameter terms too.
        has_well_159 = ("15" in ql and "9" in ql)
        eval_params_terms = [
            "evaluation parameter",
            "evaluation parameters",
            "grmax",
            "grmin",
            "rhoma",
            "rhofl",
            "archie a",
            "archie m",
            "archie n",
            "tortuosity factor",
            "cementation exponent",
            "saturation exponent",
            "matrix density",
            "fluid density",
            "ρma",
            "ρfl",
        ]
        is_eval_params = has_well_159 and any(t in ql for t in eval_params_terms)
        if is_eval_params:
            forced = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "lookup_evaluation_parameters",
                        "args": {"query": tool_query},
                        "id": "call_lookup_evaluation_parameters_1",
                    }
                ],
            )
            return {"messages": [forced]}

        # Deterministic routing: general numeric facts in notes/narrative (Rw, temperature gradient, reservoir temperature, cutoffs, etc.)
        # Check if ANY well is detected (not just hardcoded "15" and "9")
        is_fact_query = (
            (nq.well is not None or extract_well(question) is not None)
            and any(k in ql for k in ["rw", "temperature gradient", "reservoir temperature", "cutoff", "cut-offs", "cut off", "matrix density", "fluid density"])
        )
        if is_fact_query:
            forced = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "lookup_structured_facts",
                        "args": {"query": tool_query},
                        "id": "call_lookup_structured_facts_1",
                    }
                ],
            )
            return {"messages": [forced]}
        # Check for specific well name FIRST before checking wants_all_wells
        # This prevents "all formations in Well NO 15/9-F-15 A" from matching wants_all_wells
        # Import well extraction function for direct check
        try:
            from ..tools.well_picks_tool import _extract_query_well
            extracted_well = _extract_query_well(question)
        except Exception:
            extracted_well = None
        
        has_specific_well = (
            nq.well or 
            extracted_well is not None or  # Direct well extraction check (handles "A" suffix)
            extract_well(question) is not None or  # Generic well extraction
            ("well" in ql and re.search(r"\d+[\s_/-]*\d+", ql))  # Any well pattern (XX/YY or XX-YY)
        )
        
        # Only match wants_all_wells if NO specific well is mentioned
        wants_all_wells = (
            not has_specific_well  # CRITICAL: Only match if NO specific well is mentioned
            and ("formation" in ql or "formations" in ql)
            and ("well" in ql or "wells" in ql)
            and any(k in ql for k in ["each", "every", "all", "complete", "entire"])
        )
        if wants_all_wells:
            # Create a tool call directly to avoid LLM skipping wells.
            forced = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "lookup_well_picks",
                        "args": {"query": tool_query},
                        "id": "call_lookup_well_picks_all_wells_1",
                    }
                ],
            )
            return {"messages": [forced]}

        # Deterministic routing: formation list for a specific well (e.g. "formations present in 15/9-F-4").
        # We force the structured well-picks tool so the LLM doesn't answer "not in context".
        # Check for "list" queries with formations and a well name - be very permissive
        has_list_intent = "list" in ql or "all" in ql
        has_formations = "formation" in ql or "formations" in ql
        has_formation_in_well = any(k in ql for k in [" in ", " for ", "present", "present in", "present for", "which formations", "formations in", "all formations"])
        
        # Very permissive: if it mentions formations and has a well pattern, route to well picks
        # This catches queries like "formations in Well NO 15/9-F-15 A" even without "all" or "list"
        wants_single_well_formations = (
            has_specific_well  # Use the same check we did above (now includes direct well extraction)
            and has_formations
            and (has_list_intent or has_formation_in_well or "formation" in ql or "formations" in ql)
        )
        if wants_single_well_formations:
            logger.info(f"[ROUTING] Routing to well picks tool for single well formation query: {tool_query}")
            forced = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "lookup_well_picks",
                        "args": {"query": tool_query},
                        "id": "call_lookup_well_picks_single_well_formations_1",
                    }
                ],
            )
            return {"messages": [forced]}
    except Exception as e:
        # Log the exception instead of silently catching it
        logger.error(f"[ROUTING] Exception in deterministic routing: {e}", exc_info=True)
        # Fall back to normal behavior
        pass

    response = _get_response_model().bind_tools(tools).invoke(state["messages"])
    return {"messages": [response]}


def grade_documents(state: MessagesState) -> Literal["generate_answer", "rewrite_question"]:
    """
    Determine whether the retrieved documents are relevant to the question.
    
    Args:
        state: MessagesState containing messages
        
    Returns:
        Next node to route to: "generate_answer" or "rewrite_question"
    """
    from langchain_core.messages import ToolMessage
    
    messages = state["messages"]
    question = messages[0].content
    
    # Extract context from tool messages (retrieved documents)
    context_parts = []
    rewrite_count = 0
    
    for msg in messages:
        # Count rewrite attempts
        if isinstance(msg, HumanMessage) and msg != messages[0]:
            rewrite_count += 1
        
        # Extract tool message content
        if isinstance(msg, ToolMessage):
            context_parts.append(msg.content)
        elif hasattr(msg, 'content') and msg.content:
            # Check if it's a tool message by content length and structure
            if isinstance(msg.content, str) and len(msg.content) > 100:
                # Could be tool message content
                context_parts.append(msg.content)
    
    # Prevent infinite rewrite loops
    if rewrite_count >= 2:
        logger.warning(f"[GRADE] Too many rewrite attempts ({rewrite_count}) - proceeding to generate answer anyway")
        return "generate_answer"
    
    if not context_parts:
        # Fallback: get last message content
        context = messages[-1].content if messages else ""
    else:
        context = "\n\n".join(context_parts)
    
    logger.info(f"[GRADE] Question: {question[:100]}...")
    logger.info(f"[GRADE] Context length: {len(context)} chars, Rewrite count: {rewrite_count}")
    
    # If context is very short or empty, it's probably not relevant
    if len(context) < 50:
        logger.info("[GRADE] Context too short - rewriting question")
        return "rewrite_question"
    
    # Quick heuristic check: if context contains well name and "model" or "evaluation", likely relevant
    question_lower = question.lower()
    context_lower = context.lower()
    
    # Extract well name pattern (e.g., "15/9-F-4", "15_9-F-4", "15-9-F-4")
    import re
    well_patterns = re.findall(r'15[_\s/-]?9[_\s/-]?F[_\s/-]?[0-9A-Z]+', question_lower)
    
    # Check if question appears to be a partial sentence (doesn't end with punctuation and has "is" or similar)
    is_partial_sentence = (
        not question.strip().endswith(('.', '!', '?', ':')) and 
        ('is' in question_lower or 'accordingly' in question_lower or 'reported' in question_lower)
    )
    
    # If it's a partial sentence and context is substantial, likely relevant
    if is_partial_sentence and len(context) > 500:
        logger.info("[GRADE] Quick check: Partial sentence query with substantial context - marking as relevant")
        return "generate_answer"
    
    # Check for well name and related terms
    if well_patterns:
        well_found = any(well in context_lower for well in well_patterns)
        model_terms = ["model", "evaluation", "sleipner", "volve", "hugin", "skagerak", "formation"]
        has_model_term = any(term in context_lower for term in model_terms)
        
        if well_found and (has_model_term or "accordingly" in context_lower or "reported" in context_lower):
            logger.info("[GRADE] Quick check: Well name and model/evaluation found in context - marking as relevant")
            return "generate_answer"
    
    prompt = GRADE_PROMPT.format(question=question, context=context[:3000])  # Limit context length
    response = _get_grader_model().with_structured_output(GradeDocuments).invoke(
        [{"role": "user", "content": prompt}]
    )
    score = response.binary_score
    
    if score == "yes":
        logger.info("[GRADE] Documents are relevant - proceeding to generate answer")
        return "generate_answer"
    else:
        logger.info("[GRADE] Documents are not relevant - rewriting question")
        return "rewrite_question"


def rewrite_question(state: MessagesState):
    """
    Rewrite the original user question to improve retrieval.
    
    Args:
        state: MessagesState containing messages
        
    Returns:
        Dictionary with 'messages' key containing rewritten question
    """
    messages = state["messages"]
    question = _latest_user_question(messages)
    question = _latest_user_question(messages)
    context = messages[-1].content
    
    prompt = REWRITE_PROMPT.format(question=question, context=context)
    response = _get_response_model().invoke([{"role": "user", "content": prompt}])
    
    logger.info(f"[REWRITE] Original: {question}")
    logger.info(f"[REWRITE] Rewritten: {response.content}")
    
    return {"messages": [HumanMessage(content=response.content)]}


def generate_answer(state: MessagesState):
    """
    Generate final answer from relevant documents.
    
    Args:
        state: MessagesState containing messages
        
    Returns:
        Dictionary with 'messages' key containing final answer
    """
    import re
    from langchain_core.messages import ToolMessage, AIMessage
    
    messages = state["messages"]
    question = _latest_user_question(messages)
    
    # Collect all retrieved context from tool messages
    # BUT: If we have well picks formations output, exclude evaluation parameters context
    # to prevent LLM from extracting parameter values instead of formation names
    has_well_picks_formations_in_messages = False
    for msg in messages:
        if isinstance(msg, ToolMessage) and isinstance(msg.content, str):
            if msg.content.startswith("[WELL_PICKS]") and "formations:" in msg.content:
                has_well_picks_formations_in_messages = True
                break
    
    context_parts = []
    for msg in messages:
        # Extract tool message content
        if isinstance(msg, ToolMessage):
            # If we have well picks formations, skip evaluation parameters context
            if has_well_picks_formations_in_messages:
                if isinstance(msg.content, str) and ("Evaluation Parameters" in msg.content or "A — Well" in msg.content):
                    logger.info(f"[ANSWER] Skipping evaluation parameters context to prevent parameter extraction")
                    continue
            context_parts.append(msg.content)
        elif hasattr(msg, 'content') and msg.content:
            # Check if it's a tool message by content length and structure
            if isinstance(msg.content, str) and len(msg.content) > 100:
                # Could be tool message content
                context_parts.append(msg.content)

    # If a structured tool returned an authoritative output, bypass LLM and return as-is
    # CRITICAL: Check for well picks formations FIRST, before any other processing
    question_lower_check = question.lower()
    is_formation_query = "formation" in question_lower_check or "formations" in question_lower_check
    
    for msg in messages:
        if isinstance(msg, ToolMessage) and isinstance(msg.content, str):
            # PRIORITY 1: Well picks formations list - MUST bypass LLM to prevent parameter extraction
            if msg.content.startswith("[WELL_PICKS]") and "formations:" in msg.content:
                if is_formation_query:
                    # Format the well picks formations list directly - NEVER let LLM process this
                    lines = msg.content.split("\n")
                    if len(lines) > 1:
                        # Extract well name and formations
                        header = lines[0]  # e.g., "[WELL_PICKS] Well 15/9-F-15 formations:"
                        formations = [line.strip("- ").strip() for line in lines[1:] if line.strip().startswith("-")]
                        if formations:
                            # Extract well name from header (e.g., "[WELL_PICKS] Well NO 15/9-C-2 AH formations:")
                            # Header format: "[WELL_PICKS] Well <well_name> formations:"
                            well_name = "the well"
                            if "Well" in header:
                                # Extract well name from header: "Well NO 15/9-C-2 AH" or "Well 15/9-C-2 AH"
                                well_match = re.search(r"Well\s+(NO\s+)?(.+?)\s+formations:", header, re.IGNORECASE)
                                if well_match:
                                    well_name = (well_match.group(1) or "") + well_match.group(2)
                                else:
                                    # Fallback: try to extract from question
                                    well_name = extract_well(question) or extract_well(header) or "the well"
                            else:
                                well_name = extract_well(question) or extract_well(header) or "the well"
                            formatted = f"The formations in {well_name} are:\n" + "\n".join([f"- {f}" for f in formations])
                            logger.info(f"[ANSWER] Direct formatting of well picks formations list: {len(formations)} formations - BYPASSING LLM")
                            return {"messages": [AIMessage(content=formatted)]}
            
            # PRIORITY 2: Other structured outputs
            if msg.content.startswith("[WELL_PICKS_ALL]"):
                return {"messages": [AIMessage(content=msg.content)]}
            if msg.content.startswith("[WELL_FORMATION_PROPERTIES]"):
                return {"messages": [AIMessage(content=msg.content)]}
            if msg.content.startswith("[SECTION]"):
                return {"messages": [AIMessage(content=msg.content)]}
            
            # PRIORITY 3: Well picks errors for formation queries
            if msg.content.startswith("[WELL_PICKS]") and ("No rows found" in msg.content or "No well detected" in msg.content):
                if is_formation_query:
                    # Return the error message directly - DO NOT let LLM process evaluation parameters
                    logger.warning(f"[ANSWER] Well picks tool returned error for formation query: {msg.content}")
                    # Format the error nicely but don't let LLM extract parameter values
                    error_msg = f"Could not find formations for the specified well. {msg.content}"
                    return {"messages": [AIMessage(content=error_msg)]}
            
            # PRIORITY 4: Check for well detection errors in structured tools
            if isinstance(msg.content, str):
                # Check for well detection errors from structured facts, eval params, petro params
                if ("no_well_detected" in msg.content or "No well detected" in msg.content) and ("error" in msg.content.lower() or "FACTS_JSON" in msg.content or "EVAL_PARAMS_JSON" in msg.content or "PETRO_PARAMS_JSON" in msg.content):
                    logger.warning(f"[ANSWER] Structured tool returned well detection error: {msg.content}")
                    return {"messages": [AIMessage(content="I couldn't identify which well you're asking about. Please specify the well name clearly (e.g., 15/9-F-5, 19/9-19 bt2).")]}
            # PETRO params now flows through a structured JSON formatter below

    # Special deterministic formatting: Evaluation parameters JSON payload -> one clean technical answer.
    for msg in messages:
        if isinstance(msg, ToolMessage) and isinstance(msg.content, str) and msg.content.startswith("[EVAL_PARAMS_JSON]"):
            import json
            import re

            raw = msg.content[len("[EVAL_PARAMS_JSON]") :].strip()
            try:
                payload = json.loads(raw)
            except Exception:
                return {"messages": [AIMessage(content="Evaluation parameters: failed to parse structured payload.")]}

            if isinstance(payload, dict) and payload.get("error"):
                # Deterministic fallback: if structured lookup fails, do NOT stop.
                # Let the retriever context (if any) drive a best-effort answer downstream.
                payload = None
                break

            well = payload.get("well") or ""
            formations = payload.get("formations") or []
            params = payload.get("params") or {}
            notes = payload.get("notes") or []
            source = payload.get("source") or ""
            ps = payload.get("page_start")
            pe = payload.get("page_end")
            ql = question.lower() if isinstance(question, str) else ""

            def normalize_note_line(s: str) -> str:
                # Conservative normalization: spelling/units/spacing only (numbers unchanged)
                out = s
                out = re.sub(r"\bReservoar\b", "Reservoir", out, flags=re.IGNORECASE)
                out = re.sub(r"\boC\b", "°C", out)
                out = re.sub(r"\bOC\b", "°C", out)
                out = re.sub(r"(\d)m\b", r"\1 m", out)
                out = re.sub(r"\s{2,}", " ", out).strip()
                return out

            def format_value(param_key: str, v: str) -> str:
                vv = str(v).strip()
                if vv == "*":
                    return "Derived*"
                # Keep non-numeric as-is (e.g., "Derived*" or "N/A")
                if not re.fullmatch(r"[-+]?\d+(?:\.\d+)?", vv):
                    return vv
                # Numeric formatting without changing value (only adds trailing zeros)
                f = float(vv)
                pk = str(param_key)
                if pk in {"Rhoma", "Rhofl", "A", "B"}:
                    return f"{f:.2f}"
                if pk in {"Grmin", "Grmax"}:
                    # These are typically integers in the table
                    return f"{int(f)}" if f.is_integer() else f"{f}"
                if pk in {"a", "m", "n"}:
                    return f"{f:.2f}"
                return vv

            def display_param_name(k: str) -> str:
                # Professional labels + units where unambiguous
                if k == "Rhoma":
                    return "ρma (g/cc)"
                if k == "Rhofl":
                    return "ρfl (g/cc)"
                if k == "Grmin":
                    return "GRmin"
                if k == "Grmax":
                    return "GRmax"
                if k == "a":
                    return "Archie a"
                if k == "m":
                    return "Archie m"
                if k == "n":
                    return "Archie n"
                return k

            def sentence_param_label(k: str) -> str:
                # Human-friendly parameter labels for single-value answers (no interpretation)
                if k == "Rhoma":
                    return "Matrix density (ρma)"
                if k == "Rhofl":
                    return "Fluid density (ρfl)"
                if k == "Grmin":
                    return "GRmin"
                if k == "Grmax":
                    return "GRmax"
                if k == "a":
                    return "Archie a"
                if k == "m":
                    return "Archie m"
                if k == "n":
                    return "Archie n"
                return str(k)

            def detect_param_key_from_query() -> Optional[str]:
                """
                Detect requested parameter from the user's question.
                Returns a key that exists in `params` (e.g. 'Rhoma', 'Grmax', 'm').
                """
                if not isinstance(params, dict):
                    return None

                # Direct key mention (e.g., "Grmax", "Rhoma")
                for k in params.keys():
                    if not isinstance(k, str):
                        continue
                    kl = k.lower()
                    # Avoid false positives for single-letter keys (A/B/a/m/n) inside other words (e.g. "grmax")
                    if len(kl) == 1:
                        if re.search(rf"\b{re.escape(k)}\b", question, flags=re.IGNORECASE):
                            return k
                        continue
                    if kl in ql:
                        return k

                # Registry-based fuzzy resolution (typo tolerant)
                try:
                    registry = _get_registry()
                    entry, candidates = resolve_property_deterministic(question, registry)
                    if entry and entry.structured_key and entry.structured_key in params:
                        return entry.structured_key
                    # If ambiguous here, upstream should have asked; keep None.
                except Exception:
                    pass

                # Single-letter A/B explicitly requested
                if re.search(r"\bA\b", question) and "A" in params:
                    return "A"
                if re.search(r"\bB\b", question) and "B" in params:
                    return "B"

                return None

            def detect_formation_from_query() -> Optional[str]:
                if not isinstance(formations, list):
                    return None
                for f in formations:
                    if isinstance(f, str) and f.lower() in ql:
                        return f
                return None

            pages = ""
            if isinstance(ps, int) and isinstance(pe, int):
                pages = f" (pages {ps}-{pe})"

            requested_param = detect_param_key_from_query()
            requested_form = detect_formation_from_query()

            # If the user didn't restate the formation in a follow-up, infer it from chat history.
            if not requested_form:
                try:
                    _, inferred_form = _infer_recent_context(messages)
                    if inferred_form and isinstance(formations, list):
                        for f in formations:
                            if isinstance(f, str) and f.lower() == inferred_form.lower():
                                requested_form = f
                                break
                except Exception:
                    pass

            # If we found a formation but couldn't resolve which parameter the user meant,
            # ask a clarification question rather than dumping the whole table.
            if requested_form and not requested_param:
                try:
                    registry = _get_registry()
                    entry, candidates = resolve_property_deterministic(question, registry)
                    # If we can resolve it now, proceed by setting requested_param.
                    if entry and entry.structured_key and entry.structured_key in params:
                        requested_param = entry.structured_key
                    else:
                        # If we have candidates, let the agent decide whether to ask or pick.
                        if candidates:
                            dis = choose_property_with_agent(question, candidates)
                            if dis.clarification_question:
                                return {"messages": [AIMessage(content=dis.clarification_question)]}
                            if dis.canonical:
                                chosen = next((e for e in registry if e.canonical == dis.canonical), None)
                                if chosen and chosen.structured_key and chosen.structured_key in params:
                                    requested_param = chosen.structured_key
                        # Still unresolved: ask clarification using a bounded menu.
                        if not requested_param:
                            qlow = (question or "").lower()
                            if "dens" in qlow or "rho" in qlow or "ρ" in qlow:
                                options = ["matrix density (ρma)", "fluid density (ρfl)"]
                            else:
                                options = ["matrix density (ρma)", "fluid density (ρfl)", "GRmin", "GRmax", "Archie a", "Archie m", "Archie n"]
                            return {"messages": [AIMessage(content=f"I found formation {requested_form} in well {well}. Which parameter do you want: " + ", ".join(options) + "?")]}
                except Exception:
                    pass

            # If the user asks for a specific value, return a single value (not the whole table)
            if requested_param and requested_form and isinstance(params.get(requested_param), dict):
                row = params.get(requested_param) or {}
                v = row.get(requested_form, "N/A")
                unit = "g/cc" if requested_param in {"Rhoma", "Rhofl"} else None
                val = format_value(requested_param, v)
                out = [
                    f"{sentence_param_label(requested_param)} for {requested_form} in well {well}: {val}" + (f" {unit}" if unit else ""),
                    f"Source: {source}{pages}".strip(),
                ]
                return {"messages": [AIMessage(content="\n".join(out))]}

            # Row request: parameter only (values across formations)
            if requested_param and not requested_form and isinstance(params.get(requested_param), dict) and formations:
                row = params.get(requested_param) or {}
                lines = [f"{sentence_param_label(requested_param)} — Well {well}", ""]
                lines.extend(["| Formation | Value |", "|---|---:|"])
                for f in formations:
                    lines.append(f"| {f} | {format_value(requested_param, row.get(f, 'N/A'))} |")
                lines.append("")
                lines.append(f"Source: {source}{pages}".strip())
                return {"messages": [AIMessage(content="\n".join(lines))]}

            # Column request: formation only (all parameters for that formation)
            if requested_form and not requested_param and isinstance(params, dict):
                lines = [f"Evaluation parameters — Well {well} — {requested_form}", ""]
                lines.extend(["| Parameter | Value |", "|---|---:|"])
                order = ["Rhoma", "Rhofl", "A", "B", "Grmin", "Grmax", "a", "n", "m"]
                keys = [k for k in order if k in params] + [k for k in params.keys() if k not in order]
                for k in keys:
                    row = params.get(k) or {}
                    if not isinstance(row, dict):
                        continue
                    lines.append(f"| {display_param_name(str(k))} | {format_value(str(k), row.get(requested_form, 'N/A'))} |")
                lines.append("")
                lines.append(f"Source: {source}{pages}".strip())
                return {"messages": [AIMessage(content="\n".join(lines))]}

            def normalize_assumption_lines(raw_lines: list[str]) -> list[str]:
                """
                Convert the PDF-style note strings into cleaner, technically-correct lines
                (spelling/units/spacing), without changing numeric values.
                """
                out: list[str] = []
                for s in raw_lines:
                    s2 = normalize_note_line(str(s))

                    # Make the derived m equation more readable if present
                    # Example: "*  m = 1.865 * ( Klogh ** -0.0083)"
                    if re.search(r"\bm\s*=\s*1\.865\b", s2, re.IGNORECASE) and "klogh" in s2.lower():
                        s2 = re.sub(r"^\*\s*", "", s2).strip()
                        s2 = s2.replace("**", "^")
                        # Convert "Klogh ^ -0.0083" variants to "Klogh^(-0.0083)"
                        s2 = re.sub(r"(Klogh)\s*\^\s*([-+]?[\d\.]+)", r"\1^(\2)", s2, flags=re.IGNORECASE)
                        s2 = s2.replace("*", "·")
                        out.append(f"Derived Archie m: {s2}")
                        continue

                    # Split combined Rw/temp line into two fields if possible
                    # Example: "Rw = 0.07 ohmm at 20 °C, Temp Gradient : 2.6 °C"
                    if s2.lower().startswith("rw"):
                        m = re.search(r"rw\s*=\s*([-\d\.]+)\s*([a-z]+)\s+at\s+([-\d\.]+)\s*°c", s2, re.IGNORECASE)
                        g = re.search(r"temp\s*gradient\s*:\s*([-\d\.]+)\s*°c", s2, re.IGNORECASE)
                        if m:
                            out.append(f"Rw (at {m.group(3)} °C): {m.group(1)} {m.group(2)}")
                        else:
                            out.append(s2)
                        if g:
                            out.append(f"Temperature gradient: {g.group(1)} °C")
                        continue

                    # Reservoir temperature line
                    if "reservoir temperature" in s2.lower():
                        m = re.search(r"reservoir temperature\s*:\s*([-\d\.]+)\s*°c\s+at\s+([-\d\.]+)\s*m\s*(tvdss\.?)", s2, re.IGNORECASE)
                        if m:
                            out.append(f"Reservoir temperature (at {m.group(2)} m TVDSS): {m.group(1)} °C")
                        else:
                            out.append(s2)
                        continue

                    out.append(s2)
                return out

            # Build a single clean technical response (no interpretation)
            title = f"Evaluation parameters — Well {well} (Table 1)".strip(" —")
            lines = [title, ""]

            if formations and isinstance(formations, list):
                header = "| Parameter | " + " | ".join([str(f) for f in formations]) + " |"
                sep = "|---|" + "|".join(["---:"] * len(formations)) + "|"
                lines.extend([header, sep])

                order = ["Rhoma", "Rhofl", "A", "B", "Grmin", "Grmax", "a", "n", "m"]
                seen = set()
                keys: list[str] = []
                for k in order:
                    if k in params:
                        keys.append(k)
                        seen.add(k)
                for k in params.keys():
                    if k not in seen:
                        keys.append(k)

                for k in keys:
                    row = params.get(k) or {}
                    vals = []
                    for f in formations:
                        vals.append(format_value(str(k), row.get(f, "N/A")))
                    lines.append("| " + display_param_name(str(k)) + " | " + " | ".join(vals) + " |")

                # Footnote for derived entries
                if any(format_value("m", (params.get("m") or {}).get(f, "")) == "Derived*" for f in formations):
                    lines.append("")
                    lines.append("*Derived*: value is derived per report note (see Assumptions).")

            if notes:
                lines.append("")
                lines.append("Assumptions (Table 1):")
                for n in normalize_assumption_lines([str(n) for n in notes]):
                    lines.append(f"- {n}")

            lines.append("")
            # Keep full path for click-to-page UI, but with clean formatting
            lines.append(f"Source: {source}{pages}".strip())

            return {"messages": [AIMessage(content="\n".join(lines))]}

    # Special deterministic formatting: Petrophysical parameters JSON payload -> cell/row/column/full table
    for msg in messages:
        if isinstance(msg, ToolMessage) and isinstance(msg.content, str) and msg.content.startswith("[PETRO_PARAMS_JSON]"):
            import json
            import re

            raw = msg.content[len("[PETRO_PARAMS_JSON]") :].strip()
            try:
                payload = json.loads(raw)
            except Exception:
                return {"messages": [AIMessage(content="Petrophysical parameters: failed to parse structured payload.")]}

            if isinstance(payload, dict) and payload.get("error"):
                # Deterministic fallback: allow downstream to use retriever context
                break

            well = payload.get("well") or ""
            formations = payload.get("formations") or []
            values = payload.get("values") or {}
            sources = payload.get("sources") or []
            ql = question.lower() if isinstance(question, str) else ""

            def detect_formation() -> Optional[str]:
                """Detect formation from query, with improved matching."""
                if not isinstance(formations, list):
                    return None
                
                # Normalize query for matching (remove common suffixes)
                ql_normalized = ql.replace("fm.", "").replace("fm", "").replace("formation", "").replace("formations", "")
                
                # First pass: exact or substring match on base formation token
                for f in formations:
                    if not isinstance(f, str):
                        continue
                    f_lower = f.lower()
                    f_base = f_lower.replace("fm.", "").replace("fm", "").replace("formation", "").strip()
                    
                    # Check if formation name appears in query
                    if f_lower in ql or f_base in ql_normalized:
                        return f
                
                # Second pass: check if query contains formation name (even if not in formations list)
                # This helps detect when user asks for a formation that's not in the table
                # Common formation names in the dataset
                known_formations = ["draupne", "heather", "hugin", "sleipner", "ekofisk", "hod", "ty", "utsira", 
                                  "skagerrak", "hordaland", "nordland", "shetland", "seabed"]
                for known_f in known_formations:
                    if known_f in ql_normalized:
                        # Check if it's in the formations list
                        for f in formations:
                            if isinstance(f, str) and known_f in f.lower():
                                return f
                        # If not found, return None but we'll handle this case below
                        return None
                
                return None

            def detect_param() -> Optional[str]:
                # Supports both canonical and common synonyms
                syn = {
                    "netgros": "netgros",
                    "net to gross": "netgros",
                    "net-to-gross": "netgros",
                    "net/gross": "netgros",
                    "ntg": "netgros",
                    "n/g": "netgros",
                    "phif": "phif",
                    "porosity": "phif",
                    "phi": "phif",
                    "sw": "sw",
                    "water saturation": "sw",
                    "klogh": "klogh",
                    "permeability": "klogh",
                    "klogh_a": "klogh_a",
                    "klogh_h": "klogh_h",
                    "klogh_g": "klogh_g",
                    "arithmetic": "klogh_a",
                    "harmonic": "klogh_h",
                    "geometric": "klogh_g",
                }
                # Special handling: "klogh harmonic" etc
                if "klogh" in ql and "harmonic" in ql:
                    return "klogh_h"
                if "klogh" in ql and "arithmetic" in ql:
                    return "klogh_a"
                if "klogh" in ql and "geometric" in ql:
                    return "klogh_g"

                for needle, p in syn.items():
                    if needle in ql:
                        return p
                return None

            def fmt_fraction(x) -> str:
                if x is None:
                    return "N/A"
                try:
                    return f"{float(x):.3f}"
                except Exception:
                    return "N/A"

            def fmt_klogh(x) -> str:
                if x is None:
                    return "N/A"
                try:
                    f = float(x)
                    if f.is_integer():
                        return str(int(f))
                    if abs(f) >= 10:
                        return f"{f:.1f}"
                    return f"{f:.3f}"
                except Exception:
                    return "N/A"

            def fmt_value(param: str, x) -> str:
                if param in {"netgros", "phif", "sw"}:
                    return fmt_fraction(x)
                if param in {"klogh_a", "klogh_h", "klogh_g"}:
                    return fmt_klogh(x)
                if param == "klogh":
                    return "N/A"
                return str(x)

            def source_line() -> str:
                # Prefer first source; keep full path for click-to-page UI
                if isinstance(sources, list) and sources:
                    s0 = sources[0]
                    src = s0.get("source")
                    ps = s0.get("page_start")
                    pe = s0.get("page_end")
                    pages = f" (pages {ps}-{pe})" if isinstance(ps, int) and isinstance(pe, int) else ""
                    return f"Source: {src}{pages}".strip()
                return "Source: N/A"

            f = detect_formation()
            p = detect_param()
            
            # Check if query mentions a formation that's not in the formations list
            # This helps us provide a better "not found" message
            ql_normalized = ql.replace("fm.", "").replace("fm", "").replace("formation", "").replace("formations", "")
            known_formations = ["draupne", "heather", "hugin", "sleipner", "ekofisk", "hod", "ty", "utsira", 
                              "skagerrak", "hordaland", "nordland", "shetland", "seabed"]
            requested_formation_name = None
            if not f:  # Formation not detected in the data
                for known_f in known_formations:
                    if known_f in ql_normalized:
                        # Check if it's actually in the formations list
                        found_in_data = False
                        if isinstance(formations, list):
                            for ff in formations:
                                if isinstance(ff, str) and known_f in ff.lower():
                                    found_in_data = True
                                    break
                        if not found_in_data:
                            requested_formation_name = known_f.capitalize()
                            break

            # Cell: parameter + formation
            if f and p and isinstance(values, dict):
                row = values.get(f) or {}
                # Check if the formation has a value for this parameter
                param_value = row.get(p)
                
                # If value is None or N/A, provide a helpful message
                if param_value is None and p != "klogh":
                    # Formation found but no value - check if other formations have values
                    available_formations = [ff for ff in formations 
                                           if isinstance(ff, str) and values.get(ff, {}).get(p) is not None]
                    if available_formations:
                        available_list = ", ".join([f"{ff} ({fmt_value(p, values.get(ff, {}).get(p))})" 
                                                   for ff in available_formations])
                        out = [
                            f"{p.upper()} for {f} in well {well}: N/A",
                            "",
                            f"Note: {f} formation does not have a {p.upper()} value in the petrophysical parameters table for this well.",
                            f"Available formations with {p.upper()} values: {available_list}.",
                            "",
                            source_line(),
                        ]
                    else:
                        out = [
                            f"{p.upper()} for {f} in well {well}: N/A",
                            "",
                            f"Note: {f} formation does not have a {p.upper()} value in the petrophysical parameters table for this well.",
                            "",
                            source_line(),
                        ]
                    return {"messages": [AIMessage(content="\n".join(out))]}
                
                # Formation has a value - return it
                if p == "klogh":
                    a = fmt_klogh(row.get("klogh_a"))
                    h = fmt_klogh(row.get("klogh_h"))
                    g = fmt_klogh(row.get("klogh_g"))
                    out = [
                        f"KLOGH for {f} in well {well}: Arithmetic {a}, Harmonic {h}, Geometric {g}",
                        source_line(),
                    ]
                    return {"messages": [AIMessage(content="\n".join(out))]}
                out = [
                    f"{p.upper()} for {f} in well {well}: {fmt_value(p, param_value)}",
                    source_line(),
                ]
                return {"messages": [AIMessage(content="\n".join(out))]}

            # Row: parameter only (across formations)
            if p and not f and isinstance(values, dict) and isinstance(formations, list):
                # Check if user asked for a specific formation that's not in the data
                if requested_formation_name:
                    available_formations = [ff for ff in formations 
                                         if isinstance(ff, str) and values.get(ff, {}).get(p) is not None]
                    if available_formations:
                        available_list = ", ".join([f"{ff} ({fmt_value(p, values.get(ff, {}).get(p))})" 
                                                   for ff in available_formations])
                        out = [
                            f"{p.upper()} for {requested_formation_name} formation in well {well}: N/A",
                            "",
                            f"Note: {requested_formation_name} formation does not have a {p.upper()} value in the petrophysical parameters table for this well.",
                            f"Available formations with {p.upper()} values: {available_list}.",
                            "",
                            source_line(),
                        ]
                    else:
                        out = [
                            f"{p.upper()} for {requested_formation_name} formation in well {well}: N/A",
                            "",
                            f"Note: {requested_formation_name} formation does not have a {p.upper()} value in the petrophysical parameters table for this well.",
                            "",
                            source_line(),
                        ]
                    return {"messages": [AIMessage(content="\n".join(out))]}
                
                # Otherwise, show all formations (original behavior)
                title = f"{p.upper()} — Well {well}"
                lines = [title, "", "| Formation | Value |", "|---|---:|"]
                for ff in formations:
                    row = values.get(ff) or {}
                    if p == "klogh":
                        a = fmt_klogh(row.get("klogh_a"))
                        h = fmt_klogh(row.get("klogh_h"))
                        g = fmt_klogh(row.get("klogh_g"))
                        lines.append(f"| {ff} | A/H/G = {a}/{h}/{g} |")
                    else:
                        lines.append(f"| {ff} | {fmt_value(p, row.get(p))} |")
                lines.append("")
                lines.append(source_line())
                return {"messages": [AIMessage(content="\n".join(lines))]}

            # Column: formation only (all parameters for that formation)
            if f and not p and isinstance(values, dict):
                row = values.get(f) or {}
                lines = [f"Petrophysical parameters — Well {well} — {f}", "", "| Parameter | Value |", "|---|---:|"]
                lines.append(f"| Net/Gross | {fmt_fraction(row.get('netgros'))} |")
                lines.append(f"| PHIF | {fmt_fraction(row.get('phif'))} |")
                lines.append(f"| SW | {fmt_fraction(row.get('sw'))} |")
                lines.append(f"| KLOGH (A/H/G) | {fmt_klogh(row.get('klogh_a'))}/{fmt_klogh(row.get('klogh_h'))}/{fmt_klogh(row.get('klogh_g'))} |")
                lines.append("")
                lines.append(source_line())
                return {"messages": [AIMessage(content="\n".join(lines))]}

            # Full table (well only / ambiguous)
            lines = [f"Petrophysical parameters — Well {well}", ""]
            lines.append("| Formation | Net/Gross | PHIF | SW | KLOGH (A/H/G) |")
            lines.append("|---|---:|---:|---:|---|")
            if isinstance(formations, list) and isinstance(values, dict):
                for ff in formations:
                    row = values.get(ff) or {}
                    lines.append(
                        f"| {ff} | {fmt_fraction(row.get('netgros'))} | {fmt_fraction(row.get('phif'))} | {fmt_fraction(row.get('sw'))} | "
                        f"{fmt_klogh(row.get('klogh_a'))}/{fmt_klogh(row.get('klogh_h'))}/{fmt_klogh(row.get('klogh_g'))} |"
                    )
            lines.append("")
            lines.append(source_line())
            return {"messages": [AIMessage(content="\n".join(lines))]}

    # Special deterministic formatting: Structured facts JSON payload -> single value or list (no interpretation)
    for msg in messages:
        if isinstance(msg, ToolMessage) and isinstance(msg.content, str) and msg.content.startswith("[FACTS_JSON]"):
            import json

            raw = msg.content[len("[FACTS_JSON]") :].strip()
            try:
                payload = json.loads(raw)
            except Exception:
                return {"messages": [AIMessage(content="Facts: failed to parse structured payload.")]}

            if isinstance(payload, dict) and payload.get("error"):
                # Deterministic fallback: allow downstream to use retriever context
                break

            well = payload.get("well") or ""
            matches = payload.get("matches") or []
            message = payload.get("message")

            if message and not matches:
                return {"messages": [AIMessage(content=f"{message} (well {well})")]}

            # If exactly one match, return a single line + citation
            if isinstance(matches, list) and len(matches) == 1:
                r = matches[0]
                unit = r.get("unit")
                pages = ""
                ps = r.get("page_start")
                pe = r.get("page_end")
                if isinstance(ps, int) and isinstance(pe, int):
                    pages = f" (pages {ps}-{pe})"
                line = f"{r.get('parameter')}: {r.get('value')}" + (f" {unit}" if unit else "")
                src = f"Source: {r.get('source')}{pages}".strip()
                return {"messages": [AIMessage(content=line + "\n" + src)]}

            # Otherwise return a compact table (top 25)
            lines = [f"Facts — Well {well}", "", "| Parameter | Value | Source (page) |", "|---|---:|---|"]
            for r in (matches[:25] if isinstance(matches, list) else []):
                unit = r.get("unit")
                val = str(r.get("value")) + (f" {unit}" if unit else "")
                ps = r.get("page_start")
                pe = r.get("page_end")
                page = ""
                if isinstance(ps, int) and isinstance(pe, int):
                    page = f"p.{ps}" if ps == pe else f"pp.{ps}-{pe}"
                lines.append(f"| {r.get('parameter')} | {val} | {Path(str(r.get('source'))).name} {page} |")
            lines.append("")
            # Keep full path source line for click-to-page for first match
            if isinstance(matches, list) and matches:
                r0 = matches[0]
                ps = r0.get("page_start")
                pe = r0.get("page_end")
                pages = f" (pages {ps}-{pe})" if isinstance(ps, int) and isinstance(pe, int) else ""
                lines.append(f"Source: {r0.get('source')}{pages}".strip())
            return {"messages": [AIMessage(content="\n".join(lines))]}
    
    if not context_parts:
        # Fallback: get last message content
        context = messages[-1].content if messages else ""
    else:
        context = "\n\n".join(context_parts)
    
    logger.info(f"[ANSWER] Question: {question[:100]}...")
    logger.info(f"[ANSWER] Using context of {len(context)} chars")
    if context:
        logger.info(f"[ANSWER] Context preview: {context[:200]}...")
    
    # Check if question needs information from multiple sources (e.g., "all wells", "all formations")
    question_lower = question.lower().strip()
    needs_multiple_sources = any(term in question_lower for term in ["all", "every", "each", "list", "summary", "overview"])
    
    # Adjust context limit based on query type
    # With document-level retrieval, we now retrieve ALL chunks from relevant documents
    # So we should use the full context for all queries, not just comprehensive ones
    is_list_query = "list" in question_lower or "each" in question_lower or "all" in question_lower
    if needs_multiple_sources or is_list_query:
        # For comprehensive/list queries, use ALL context (no truncation)
        context_limit = len(context)  # Use full context, no truncation
        logger.info(f"[ANSWER] Comprehensive/list query detected - using FULL context ({context_limit} chars)")
    else:
        # For regular queries, also use full context since we retrieve full documents
        context_limit = len(context)  # Use full context for document-level retrieval
        logger.info(f"[ANSWER] Using FULL document context ({context_limit} chars) - document-level retrieval enabled")
    
    # Check if question is a partial sentence - if so, find exact completion
    if not question_lower.endswith(('.', '!', '?', ':')) and len(question.split()) > 5:
        # Likely a partial sentence - try to find exact completion in context
        import re
        # Escape special regex characters in question
        question_escaped = re.escape(question)
        # Look for text that starts with the question
        matches = re.findall(rf'{question_escaped}[^.]*\.', context, re.IGNORECASE | re.DOTALL)
        if matches:
            # Found exact completion
            exact_completion = matches[0]
            logger.info(f"[ANSWER] Found exact completion: {exact_completion[:200]}...")
            # Use a simpler prompt for exact extraction
            prompt = f"The user asked: {question}\n\nIn the context, I found this exact completion:\n{exact_completion}\n\nProvide this exact text as the answer, or a slightly cleaned version if needed."
            response = _get_response_model().invoke([{"role": "user", "content": prompt}])
            logger.info("[ANSWER] Generated answer from exact completion")
            return {"messages": [response]}
    
    # Detect if context contains well picks data
    is_well_picks_context = False
    for msg in messages:
        if isinstance(msg, ToolMessage) and isinstance(msg.content, str):
            # Check for well picks tool output (but not [WELL_PICKS_ALL] which already bypasses LLM)
            if msg.content.startswith("[WELL_PICKS]") and not msg.content.startswith("[WELL_PICKS_ALL]"):
                is_well_picks_context = True
                logger.info("[ANSWER] Detected well picks structured tool output")
                break
    
    # Also check context content for well picks patterns (from vector retrieval)
    if not is_well_picks_context and context:
        # Very specific indicators that alone indicate well picks
        strong_indicators = ["Well_picks", "NO 15/9", "Well NO "]
        # Less specific indicators that need multiple matches
        weak_indicators = [
            "well picks", "formation picks", "Pick:", "Quality:",
            "Top: MD", "Base: MD", "Top: TVD", "Base: TVD"
        ]
        # Check for strong indicators (one is enough) or multiple weak indicators
        strong_matches = sum(1 for indicator in strong_indicators if indicator in context)
        weak_matches = sum(1 for indicator in weak_indicators if indicator in context)
        if strong_matches >= 1 or weak_matches >= 2:
            is_well_picks_context = True
            logger.info(f"[ANSWER] Detected well picks patterns in context (strong: {strong_matches}, weak: {weak_matches})")
    
    # Check if we have well picks formation list output
    has_well_picks_formations = False
    well_picks_formations_content = None
    for msg in messages:
        if isinstance(msg, ToolMessage) and isinstance(msg.content, str):
            if msg.content.startswith("[WELL_PICKS]") and "formations:" in msg.content:
                has_well_picks_formations = True
                well_picks_formations_content = msg.content
                break
    
    # For ANY formation query, add explicit instructions to prevent parameter extraction
    if "formation" in question_lower:
        # CRITICAL: Check if we have well picks formations output - if so, use it directly and NEVER call LLM
        # This must happen BEFORE any context is passed to LLM to prevent parameter extraction
        if has_well_picks_formations and well_picks_formations_content:
            # Format it directly - don't let LLM process it AT ALL
            lines = well_picks_formations_content.split("\n")
            if len(lines) > 1:
                formations = [line.strip("- ").strip() for line in lines[1:] if line.strip().startswith("-")]
                if formations:
                    # Extract well name from well_picks_formations_content header
                    well_name = "the well"
                    if "Well" in well_picks_formations_content:
                        well_match = re.search(r"Well\s+(NO\s+)?(.+?)\s+formations:", well_picks_formations_content, re.IGNORECASE)
                        if well_match:
                            well_name = (well_match.group(1) or "") + well_match.group(2)
                        else:
                            well_name = extract_well(question) or "the well"
                    else:
                        well_name = extract_well(question) or "the well"
                    formatted = f"The formations in {well_name} are:\n" + "\n".join([f"- {f}" for f in formations])
                    logger.info(f"[ANSWER] Direct formatting of well picks formations (bypass LLM): {len(formations)} formations - RETURNING IMMEDIATELY")
                    return {"messages": [AIMessage(content=formatted)]}
        
        # If we reach here, well picks tool wasn't called or didn't return formations
        # This should NOT happen for formation queries, but if it does, add strong warnings
        
        # If no well picks output but asking for formations, add strong warning
        enhanced_prompt = GENERATE_PROMPT.format(question=question, context=context[:context_limit])
        if "list" in question_lower or "each" in question_lower or "all" in question_lower:
            enhanced_prompt += "\n\nCRITICAL: This is a LISTING query. You MUST include EVERY item mentioned in the context. Count the number of wells/items in the context and ensure you list ALL of them. Do not skip any. Do NOT say 'the list may be incomplete' - you have the full context."
        
        # CRITICAL: Prevent parameter extraction for formation queries
        enhanced_prompt += "\n\nCRITICAL INSTRUCTION: The user is asking for FORMATION NAMES (like Draupne, Heather, Hugin, Sleipner). Do NOT extract parameter values (like 'A', 'B', 'n', 'm', 'Rhoma', 'Rhofl', '0.00', '0.40') from evaluation parameters tables. If you see a table with columns like 'Formation' and 'Value', the user wants the FORMATION NAMES from the 'Formation' column, NOT the values. Do NOT create a table with 'Formation' and 'Value' columns - just list the formation names."
        
        if is_well_picks_context or has_well_picks_formations:
            enhanced_prompt += "\n\nIMPORTANT: The context contains well picks data. Use the [WELL_PICKS] formations list if present. Do not use evaluation parameters tables."
            if well_picks_formations_content:
                enhanced_prompt += f"\n\nHere is the well picks formations list you MUST use:\n{well_picks_formations_content}"
        
        prompt = enhanced_prompt
    elif "list" in question_lower or "each" in question_lower:
        enhanced_prompt = GENERATE_PROMPT.format(question=question, context=context[:context_limit])
        enhanced_prompt += "\n\nCRITICAL: This is a LISTING query. You MUST include EVERY item mentioned in the context. Count the number of wells/items in the context and ensure you list ALL of them. Do not skip any. Do NOT say 'the list may be incomplete' - you have the full context."
        if is_well_picks_context:
            enhanced_prompt += "\n\nIMPORTANT: The context provided is from the Well_picks_Volve_v1 document. You MUST ONLY use information from this document. Do not use information from other sources or make assumptions. If the answer is not in the well picks document, state that clearly."
        prompt = enhanced_prompt
    else:
        prompt = GENERATE_PROMPT.format(question=question, context=context[:context_limit])
        if is_well_picks_context:
            prompt += "\n\nIMPORTANT: The context provided is from the Well_picks_Volve_v1 document. You MUST ONLY use information from this document. Do not use information from other sources or make assumptions. If the answer is not in the well picks document, state that clearly."
    
    response = _get_response_model().invoke([{"role": "user", "content": prompt}])
    
    # Phase 1.5: Extract source citations with page numbers from context
    # Parse citations from context (format: [Source: path (page X)] or [Source: path (pages X-Y)])
    import re
    citations = []
    citation_pattern = r'\[Source:\s*([^\]]+?)\s*(?:\(page\s+(\d+)\)|\(pages\s+(\d+)\s*-\s*(\d+)\))?\]'
    
    for msg in messages:
        if isinstance(msg, ToolMessage) and isinstance(msg.content, str):
            # Extract citations from tool message content
            matches = re.findall(citation_pattern, msg.content)
            for match in matches:
                source_path = match[0].strip()
                page_single = match[1] if match[1] else None
                page_start = match[2] if match[2] else None
                page_end = match[3] if match[3] else None
                
                # Build citation
                if page_start and page_end:
                    if page_start == page_end:
                        citation = f"Source: {source_path} (page {page_start})"
                    else:
                        citation = f"Source: {source_path} (pages {page_start}-{page_end})"
                elif page_single:
                    citation = f"Source: {source_path} (page {page_single})"
                else:
                    citation = f"Source: {source_path}"
                
                # Avoid duplicates
                if citation not in citations:
                    citations.append(citation)
    
    # Also extract from context string directly
    if context:
        matches = re.findall(citation_pattern, context)
        for match in matches:
            source_path = match[0].strip()
            page_single = match[1] if match[1] else None
            page_start = match[2] if match[2] else None
            page_end = match[3] if match[3] else None
            
            if page_start and page_end:
                if page_start == page_end:
                    citation = f"Source: {source_path} (page {page_start})"
                else:
                    citation = f"Source: {source_path} (pages {page_start}-{page_end})"
            elif page_single:
                citation = f"Source: {source_path} (page {page_single})"
            else:
                citation = f"Source: {source_path}"
            
            if citation not in citations:
                citations.append(citation)
    
    # Append citations to response if found
    answer_content = response.content
    if citations:
        answer_content += "\n\n" + "\n".join(citations[:5])  # Limit to first 5 citations
        logger.info(f"[ANSWER] Added {len(citations)} source citation(s) to answer")
    
    logger.info("[ANSWER] Generated final answer")
    return {"messages": [AIMessage(content=answer_content)]}

