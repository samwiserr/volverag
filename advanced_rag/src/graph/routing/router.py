"""
Query router that orchestrates routing strategies.
"""
import os
import re
import logging
from pathlib import Path
from typing import List
from langgraph.graph import MessagesState
from langchain_core.messages import AIMessage
from src.core.result import Result, AppError, ErrorType
from src.core.logging import get_logger
from src.normalize.query_normalizer import normalize_query, extract_well, normalize_formation
from src.normalize.property_registry import resolve_property_deterministic, PropertyEntry
from src.normalize.agent_disambiguator import choose_property_with_agent
from src.normalize.entity_resolver import resolve_with_bounded_agent
from src.query.incomplete_query_handler import is_incomplete_query, complete_incomplete_query
from src.query.query_decomposer import decompose_query, expand_query_synonyms
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
    # Import helper functions from nodes.py inside function to avoid circular imports
    from ..nodes import _get_response_model, _get_registry
    
    # Deterministic routing: for "complete list ... formations ... each/all wells" queries,
    # force the structured well-picks tool to run (no embeddings / no LLM summarization).
    try:
        logger.info(f"[ROUTING] Entering routing logic. State messages: {len(state.get('messages', []))}")
        question = _latest_user_question(state.get("messages"))
        
        # Edge case: Empty or None question
        if not question or not isinstance(question, str) or not question.strip():
            logger.warning(f"[ROUTING] Empty or invalid question: '{question}'")
            return {"messages": [AIMessage(content="I didn't receive a valid question. Please ask a question about the Volve petrophysical reports.")]}
        
        # Edge case: Very long queries (potential abuse or copy-paste errors)
        from ...core.thresholds import get_retrieval_thresholds
        thresholds = get_retrieval_thresholds()
        if len(question) > thresholds.max_query_length:
            logger.warning(f"[ROUTING] Query too long: {len(question)} characters")
            return {"messages": [AIMessage(content=f"Your question is too long ({len(question)} characters). Please keep questions under {thresholds.max_query_length} characters. You can break complex questions into smaller parts.")]}
        
        ql = question.lower() if isinstance(question, str) else ""
        logger.info(f"[ROUTING] Initial question: '{question[:100] if isinstance(question, str) else question}', ql: '{ql[:100]}'")
        
        # Store original question for "all wells" detection (before any modifications)
        original_question = question if isinstance(question, str) else ""
        # CRITICAL: Extract well name from ORIGINAL question before any rewriting/decomposition
        # This ensures filtering uses the user's intended well, not wells added by rewrite
        original_well = extract_well(original_question) if original_question else None
        if original_well:
            logger.info(f"[ROUTING] Extracted well from original question: {original_well}")
        
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
        # Use the same detection logic as the routing check below
        decomposed_sub_queries = []
        has_formation_keyword = "formation" in ql or "formations" in ql
        has_all_keyword = any(k in ql for k in ["all", "each", "every", "complete", "entire", "list all"])
        has_list_all = "list" in ql and "all" in ql
        has_properties = "properties" in ql or "petrophysical" in ql
        # Check if a specific well is mentioned (this takes priority)
        well_in_query = extract_well(original_question) if original_question else None
        
        is_all_wells_query = (
            not well_in_query  # No specific well mentioned
            and has_formation_keyword
            and (
                # Pattern 1: Has "properties" or "petrophysical" + "all" keyword
                (has_properties and has_all_keyword)
                # Pattern 2: "list all" + "formation" (even without properties)
                or (has_list_all and has_formation_keyword)
                # Pattern 3: "all" + "formation" + "available" (e.g., "list all available formation")
                or (has_all_keyword and has_formation_keyword and "available" in ql)
            )
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
        
        # Update ql after query decomposition/rewriting
        ql = question.lower() if isinstance(question, str) else ""

        # Normalize -> Resolve (deterministic, runs before any retrieval)
        # CRITICAL: Normalize the REWRITTEN question (after decomposition) to pick up corrected entity names
        # This ensures typos fixed by query decomposition (e.g., "Hugim" -> "Hugin") are captured
        # Use absolute path for persist_dir to ensure formation vocab loads correctly
        vectorstore_dir = Path(__file__).resolve().parents[3] / "data" / "vectorstore"
        persist_dir_str = str(vectorstore_dir) if vectorstore_dir.exists() else "./data/vectorstore"
        nq = normalize_query(question if isinstance(question, str) else "", persist_dir=persist_dir_str)
        if (not nq.well) or (not nq.formation):
            cw, cf = _infer_recent_context(state.get("messages"))
            if not nq.well and cw:
                nq = nq.__class__(raw=nq.raw, well=cw, formation=nq.formation, property=nq.property, tool=nq.tool, intent=nq.intent)
            if not nq.formation and cf:
                nq = nq.__class__(raw=nq.raw, well=nq.well, formation=cf, property=nq.property, tool=nq.tool, intent=nq.intent)

        # Smart next-step: bounded agentic resolver (well+formation+property) for heavy-typo inputs.
        # Only invoke when deterministic parse is missing key entities for a fact-like query.
        # BUT: Skip entity resolution for param queries - let routing handle it directly
        if isinstance(question, str):
            looks_facty = any(t in ql for t in ["dens", "rho", "archie", "gr", "net", "gross", "phif", "sw", "klogh", "rw", "temperature"])
            # Check if this is a param query that should route directly to tool
            is_likely_param_query = any(k in ql for k in ["water saturation", "sw", "phif", "porosity", "net to gross", "klogh", "permeability", "petrophysical parameter"])
            
            # Only do entity resolution if it's NOT a param query (param queries should route directly)
            if looks_facty and (nq.well is None or nq.property is None or nq.formation is None) and not is_likely_param_query:
                try:
                    # Enhance question with inferred context before passing to entity resolver
                    # This ensures follow-up queries like "matrix density" include well/formation from previous messages
                    enhanced_question = question
                    if nq.well and extract_well(question) is None:
                        enhanced_question = f"{question} in well {nq.well}"
                    if nq.formation and nq.formation.lower() not in question.lower():
                        enhanced_question = f"{enhanced_question} formation {nq.formation}"
                    er = resolve_with_bounded_agent(enhanced_question, persist_dir="./data/vectorstore")
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
                    # BUT: Skip this for param queries - let routing handle it
                    if looks_facty and not nq.well and not is_likely_param_query:
                        return {"messages": [AIMessage(content="I couldn't identify which well you're asking about. Please specify the well name clearly (e.g., 15/9-F-5, 19/9-19 bt2).")]}
                    pass

        # Build an "effective query" for tools so follow-ups like "matrix density"
        # still include well/formation context inferred from chat history.
        # CRITICAL: Use original_well if available (from original question) to prevent
        # rewritten queries from adding wrong well names
        tool_query = question if isinstance(question, str) else ""
        tool_query = tool_query.strip()
        
        # CRITICAL FIX: If rewrite added a different well, remove it and use original well
        # Prefer original_well over nq.well to avoid using wells added by rewrite
        well_for_tool = original_well or nq.well
        if original_well and well_for_tool:
            # Remove any well mentions from rewritten query that don't match original
            # Find all well mentions in tool_query (re is imported at module level)
            well_mentions = re.findall(r'15[_\s/-]?9[_\s/-]?[Ff]?[_\s/-]*-?\s*\d+[A-Z]?', tool_query, re.IGNORECASE)
            # Remove well mentions that don't match original_well
            for mention in well_mentions:
                normalized_mention = re.sub(r'[\s_/-]', '', mention.upper())
                normalized_original = re.sub(r'[\s_/-]', '', original_well.upper())
                if normalized_mention != normalized_original:
                    # Remove this well mention from tool_query
                    tool_query = re.sub(re.escape(mention), '', tool_query, flags=re.IGNORECASE).strip()
                    tool_query = re.sub(r'\s+', ' ', tool_query)  # Clean up extra spaces
                    logger.info(f"[ROUTING] Removed well mention '{mention}' from rewritten query (doesn't match original '{original_well}')")
        
        # Ensure original well is in tool_query for filtering
        if well_for_tool and extract_well(tool_query) is None:
            tool_query = (tool_query + f" in well {well_for_tool}").strip()
        elif well_for_tool:
            # Verify the well in tool_query matches original_well
            extracted = extract_well(tool_query)
            if extracted:
                normalized_extracted = re.sub(r'[\s_/-]', '', extracted.upper())
                normalized_original = re.sub(r'[\s_/-]', '', well_for_tool.upper())
                if normalized_extracted != normalized_original:
                    # Replace with original well
                    tool_query = re.sub(re.escape(extracted), well_for_tool, tool_query, flags=re.IGNORECASE)
                    logger.info(f"[ROUTING] Replaced well '{extracted}' with original well '{well_for_tool}' in tool_query")
        
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

        # Deterministic routing: Petrophysical parameters table (Net/Gross, PHIF, SW, KLOGH) by well/formation/parameter
        # IMPORTANT: This must happen BEFORE fact-like query routing to catch param queries
        # Prefer structured petrophysical lookup when a local cache exists (no hardcoded well-number requirement)
        logger.info(f"[ROUTING] Starting petro params routing check. ql='{ql[:100]}', question='{question[:100] if isinstance(question, str) else question}'")
        try:
            vectorstore_dir = Path(__file__).resolve().parents[3] / "data" / "vectorstore"
            cache_path = vectorstore_dir / "petro_params_cache.json"
            has_petro_cache = cache_path.exists()
            logger.info(f"[ROUTING] Checking petro cache: path='{cache_path}', exists={has_petro_cache}")

            # Check for parameter keywords (including variations)
            param_keywords = ["petrophysical parameters", "petrophysical parameter", "net to gross", "net-to-gross", "netgros", "net/gross", "ntg", "n/g", "phif", "phi", "poro", "porosity", "water saturation", "sw", "klogh", "permeability", "permeab", "perm"]
            # Use re module (imported at top) - check for standalone 'sw' or 'k' as parameter keywords
            has_param_keyword = any(k in ql for k in param_keywords) or bool(re.search(r'\bsw\b', ql, re.IGNORECASE)) or bool(re.search(r'\bk\b', ql, re.IGNORECASE))
            
            # Check for well pattern (15/9 or similar)
            extracted_well = extract_well(question)
            has_well_pattern = ("15" in ql and "9" in ql) or extracted_well is not None or nq.well is not None
            logger.info(f"[ROUTING] Well detection: extracted_well='{extracted_well}', nq.well='{nq.well}', has_well_pattern={has_well_pattern}")
            logger.info(f"[ROUTING] Query text (ql): '{ql[:100]}'")
            logger.info(f"[ROUTING] has_param_keyword={has_param_keyword}")

            is_param_query = has_param_keyword and (has_petro_cache or has_well_pattern)
            
            if is_param_query:
                logger.info(f"[ROUTING] ✅ Detected param query - routing to lookup_petrophysical_params. Query: '{question[:100]}'")
                logger.info(f"[ROUTING] has_param_keyword={has_param_keyword}, has_petro_cache={has_petro_cache}, has_well_pattern={has_well_pattern}")
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
            else:
                logger.warning(f"[ROUTING] ❌ NOT routing to petro params tool. has_param_keyword={has_param_keyword}, has_petro_cache={has_petro_cache}, has_well_pattern={has_well_pattern}")
        except Exception as e:
            logger.error(f"[ROUTING] Exception in petro params routing: {e}", exc_info=True)

        # Deterministic routing: "Evaluation parameters" tables (Rhoma/Rhofl/GRmin/GRmax/Archie a,n,m, etc.)
        # IMPORTANT: This must happen BEFORE fact-like query routing to catch density/rho queries
        # Users often ask for these values without saying "evaluation parameters", so we key off parameter terms too.
        # Use same well detection logic as petro params routing (not just "15" and "9")
        extracted_well_eval = extract_well(question)
        has_well_pattern_eval = ("15" in ql and "9" in ql) or extracted_well_eval is not None or nq.well is not None
        
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
        is_eval_params = has_well_pattern_eval and any(t in ql for t in eval_params_terms)
        if is_eval_params:
            logger.info(f"[ROUTING] ✅ Detected eval params query - routing to lookup_evaluation_parameters. Query: '{question[:100] if isinstance(question, str) else question}', well_pattern={has_well_pattern_eval}, extracted_well='{extracted_well_eval}'")
            # Enhance retriever query with evaluation parameter synonyms for better retrieval
            # Map query terms to evaluation parameter synonyms
            eval_param_synonyms = {
                "matrix density": ["rhoma", "ρma", "matrix density", "evaluation parameters", "density matrix"],
                "fluid density": ["rhofl", "ρfl", "fluid density", "evaluation parameters", "density fluid"],
                "density": ["rhoma", "rhofl", "ρma", "ρfl", "matrix density", "fluid density", "evaluation parameters"],
                "grmax": ["gr max", "gamma ray max", "gr maximum", "evaluation parameters"],
                "grmin": ["gr min", "gamma ray min", "gr minimum", "evaluation parameters"],
                "archie": ["archie a", "archie m", "archie n", "tortuosity", "cementation", "saturation exponent", "evaluation parameters"],
            }
            
            # Build enhanced query with synonyms
            enhanced_retriever_query = tool_query
            ql_lower = ql.lower()
            for term, synonyms in eval_param_synonyms.items():
                if term in ql_lower:
                    # Add synonyms to query to improve retrieval
                    enhanced_retriever_query = f"{tool_query} {' '.join(synonyms)}"
                    logger.info(f"[ROUTING] Enhanced retriever query with eval param synonyms: '{enhanced_retriever_query[:150]}'")
                    break
            
            # Add retriever as fallback in case eval params tool doesn't have data for this well
            # This ensures we can still answer from documents even if structured lookup fails
            forced = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "lookup_evaluation_parameters",
                        "args": {"query": tool_query},
                        "id": "call_lookup_evaluation_parameters_1",
                    },
                    {
                        "name": "retrieve_petrophysical_docs",
                        "args": {"query": enhanced_retriever_query},  # Use enhanced query
                        "id": "call_retrieve_petrophysical_docs_eval_fallback_1",
                    }
                ],
            )
            return {"messages": [forced]}

        # For fact-like queries, do typo-tolerant property resolution:
        # - fuzzy match over registry
        # - if ambiguous, ask a clarification question (agent)
        # NOTE: "matrix density" and "fluid density" are handled by evaluation parameters routing above
        # Exclude eval params terms from fact-like routing to avoid conflicts
        eval_params_in_query = any(t in ql for t in ["matrix density", "fluid density", "rhoma", "rhofl", "grmax", "grmin", "archie a", "archie m", "archie n", "tortuosity factor", "cementation exponent", "saturation exponent", "ρma", "ρfl"])
        if nq.well and not eval_params_in_query and (("density" in ql) or ("rho" in ql) or ("archie" in ql) or ("gr" in ql) or ("dens" in ql) or ("permeability" in ql) or ("permeab" in ql) or ("perm" in ql) or ("porosity" in ql) or ("poro" in ql) or ("phi" in ql) or re.search(r'\bk\b', ql, re.IGNORECASE) or nq.intent == "fact"):
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
        extracted_well_section = extract_well(question)
        has_well_pattern_section = ("15" in ql and "9" in ql) or extracted_well_section is not None or nq.well is not None
        is_section_like = (
            any(k in ql for k in ["summary", "introduction", "conclusion", "results", "discussion", "abstract"])
            and has_well_pattern_section
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

        # Check for specific well name FIRST before checking wants_all_wells
        # This prevents "all formations in Well NO 15/9-F-15 A" from matching wants_all_wells
        # Import well extraction function for direct check
        try:
            from ...tools.well_picks_tool import _extract_query_well
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
        # Enhanced detection for petrophysicist query patterns
        # Must distinguish between:
        # - "all formations in X well" → single-well lookup
        # - "all available formations" → all-wells lookup
        
        has_formation_keyword = "formation" in ql or "formations" in ql
        has_all_keyword = any(k in ql for k in ["all", "each", "every", "complete", "entire", "list all"])
        has_list_all = "list" in ql and "all" in ql
        # Enhanced property detection - more flexible for petrophysicist language
        has_properties = any(k in ql for k in [
            "properties", "petrophysical", "petro", "parameter", "parameters", 
            "reported", "values", "data", "net", "gross", "phif", "sw", "klogh"
        ])
        has_in_well = "in" in ql and ("well" in ql or has_specific_well)  # "all formations in 15/9-F-5"
        has_for_well = "for" in ql and ("well" in ql or has_specific_well)  # "all formations for 15/9-F-5"
        
        # Check for "all formations in [specific well]" pattern FIRST
        # This should route to single-well lookup, NOT all-wells
        is_single_well_all_formations = (
            has_specific_well  # A specific well is mentioned
            and has_formation_keyword
            and has_all_keyword
            and (has_in_well or has_for_well)  # "all formations in X" or "all formations for X"
        )
        
        # Check for "all formations across all wells" pattern
        # This should route to all-wells lookup
        is_all_wells_formations = (
            not has_specific_well  # No specific well mentioned
            and has_formation_keyword
            and (
                # Pattern 1: "all" + "formation" + properties-related keywords
                (has_all_keyword and has_properties)
                # Pattern 2: "list all" + "formation" (even without properties)
                or (has_list_all and has_formation_keyword)
                # Pattern 3: "all" + "formation" + "available" (e.g., "list all available formation")
                or (has_all_keyword and has_formation_keyword and "available" in ql)
                # Pattern 4: "all formations" with properties keywords anywhere (e.g., "with petrophysical properties reported")
                or (has_all_keyword and has_formation_keyword and has_properties)
            )
        )
        
        # Route single-well "all formations" queries first (priority)
        if is_single_well_all_formations:
            # Use original question to preserve the query structure
            single_well_query = original_question if original_question else tool_query
            logger.info(f"[ROUTING] Routing to formation properties tool for SINGLE well: {nq.well or extract_well(question)}")
            logger.info(f"[ROUTING] Using query for single-well lookup: '{single_well_query}'")
            forced = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "lookup_formation_properties",
                        "args": {"query": single_well_query},
                        "id": "call_lookup_formation_properties_single_well_1",
                    }
                ],
            )
            return {"messages": [forced]}
        
        # Route all-wells "all formations" queries
        if is_all_wells_formations:
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

        # Deterministic routing: general numeric facts in notes/narrative (Rw, temperature gradient, reservoir temperature, cutoffs, etc.)
        # Note: "matrix density" and "fluid density" are handled by evaluation parameters routing above
        # Check if ANY well is detected (not just hardcoded "15" and "9")
        extracted_well_fact = extract_well(question)
        has_well_pattern_fact = (nq.well is not None or extracted_well_fact is not None)
        is_fact_query = (
            has_well_pattern_fact
            and any(k in ql for k in ["rw", "temperature gradient", "reservoir temperature", "cutoff", "cut-offs", "cut off"])
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
            from ...tools.well_picks_tool import _extract_query_well
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
        
        # EXCLUDE queries asking about formation properties/characteristics/pressure/etc.
        # These should route to RetrieverTool, not WellPicksTool
        is_formation_property_query = any(k in ql for k in [
            "characteristic", "characteristics", "property", "properties", 
            "pressure", "permeability", "porosity", "saturation", "density",
            "explain", "describe", "what are", "how", "why", "interpretation",
            "evaluation", "analysis", "data", "measurement", "test"
        ])
        
        # Very permissive: if it mentions formations and has a well pattern, route to well picks
        # BUT: Exclude queries asking about formation properties/characteristics
        # This catches queries like "formations in Well NO 15/9-F-15 A" even without "all" or "list"
        wants_single_well_formations = (
            has_specific_well  # Use the same check we did above (now includes direct well extraction)
            and has_formations
            and (has_list_intent or has_formation_in_well or "formation" in ql or "formations" in ql)
            and not is_formation_property_query  # EXCLUDE property/characteristic queries
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

