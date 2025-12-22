"""
Answer generation for RAG workflow.

This module contains the generate_answer function extracted from nodes.py
for better modularity while maintaining backward compatibility.
"""
import re
import logging
from pathlib import Path
from typing import Optional
from langgraph.graph import MessagesState
from langchain_core.messages import ToolMessage, AIMessage

# Import utilities from message_utils
from ..utils.message_utils import _latest_user_question, _infer_recent_context

# Import from normalize modules
from ...normalize.query_normalizer import extract_well
from ...normalize.property_registry import resolve_property_deterministic
from ...normalize.agent_disambiguator import choose_property_with_agent

# Create logger for this module
logger = logging.getLogger(__name__)

# Copy GENERATE_PROMPT from nodes.py to avoid circular import
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


def _extract_tool_content(content: str) -> str:
    """
    Extract actual content from ToolMessage, handling Result.ok() string representation.
    
    Sometimes ToolMessage.content is the string representation of a Result object
    like "Result.ok('[PETRO_PARAMS_JSON] ...')" instead of the actual JSON string.
    This function extracts the inner value.
    
    Args:
        content: ToolMessage content (may be Result.ok('...') format)
        
    Returns:
        Extracted content string
    """
    if content.startswith("Result.ok(") and content.endswith(")"):
        try:
            # Remove "Result.ok(" and trailing ")"
            inner = content[10:-1]
            # Handle quoted strings - remove outer quotes if present
            if (inner.startswith("'") and inner.endswith("'")) or (inner.startswith('"') and inner.endswith('"')):
                return inner[1:-1]
            return inner
        except Exception:
            pass
    return content


def generate_answer(state: MessagesState):
    """
    Generate final answer from relevant documents.
    
    Args:
        state: MessagesState containing messages
        
    Returns:
        Dictionary with 'messages' key containing final answer
    """
    # Import from nodes.py at function level to avoid circular import
    from ..nodes import _get_response_model, _get_registry
    
    messages = state["messages"]
    question = _latest_user_question(messages)
    
    # Collect all retrieved context from tool messages
    # BUT: If we have well picks formations output, exclude evaluation parameters context
    # to prevent LLM from extracting parameter values instead of formation names
    has_well_picks_formations_in_messages = False
    for msg in messages:
        if isinstance(msg, ToolMessage) and isinstance(msg.content, str):
            content = _extract_tool_content(msg.content)
            if content and content.startswith("[WELL_PICKS]") and "formations:" in content:
                has_well_picks_formations_in_messages = True
                break
    
    # Check if eval params tool returned an error but we have retriever context
    eval_params_error_with_retriever = False
    for msg in messages:
        if isinstance(msg, ToolMessage) and isinstance(msg.content, str):
            content = _extract_tool_content(msg.content)
            if content and content.startswith("[EVAL_PARAMS_JSON]"):
                try:
                    import json
                    raw = content[len("[EVAL_PARAMS_JSON]"):].strip()
                    payload = json.loads(raw)
                    if isinstance(payload, dict) and payload.get("error") == "no_table_for_well":
                        # Check if we have retriever context
                        has_retriever = any(
                            isinstance(m, ToolMessage) and 
                            isinstance(m.content, str) and 
                            (hasattr(m, 'name') and "retrieve" in str(m.name).lower() or len(m.content) > 500)
                            for m in messages
                        )
                        if has_retriever:
                            eval_params_error_with_retriever = True
                            logger.info(f"[ANSWER] Eval params error detected with retriever context - will exclude error message from context")
                            break
                except Exception:
                    pass
    
    context_parts = []
    for msg in messages:
        # Extract tool message content
        if isinstance(msg, ToolMessage):
            # Extract actual content (handle Result.ok() string representation)
            content = _extract_tool_content(msg.content) if isinstance(msg.content, str) else msg.content
            
            # If we have well picks formations, skip evaluation parameters context
            if has_well_picks_formations_in_messages:
                if isinstance(content, str) and ("Evaluation Parameters" in content or "A — Well" in content):
                    logger.info(f"[ANSWER] Skipping evaluation parameters context to prevent parameter extraction")
                    continue
            # If eval params returned error but we have retriever context, exclude the error message
            if eval_params_error_with_retriever:
                if isinstance(content, str) and content.startswith("[EVAL_PARAMS_JSON]"):
                    try:
                        import json
                        raw = content[len("[EVAL_PARAMS_JSON]"):].strip()
                        payload = json.loads(raw)
                        if isinstance(payload, dict) and payload.get("error") == "no_table_for_well":
                            logger.info(f"[ANSWER] Excluding eval params error message from context - using retriever context only")
                            continue
                    except Exception:
                        pass
            context_parts.append(content if isinstance(content, str) else str(msg.content))
        elif hasattr(msg, 'content') and msg.content:
            # Check if it's a tool message by content length and structure
            if isinstance(msg.content, str) and len(msg.content) > 100:
                # Could be tool message content - extract if it's Result.ok() format
                content = _extract_tool_content(msg.content)
                context_parts.append(content if isinstance(content, str) else str(msg.content))

    # If a structured tool returned an authoritative output, bypass LLM and return as-is
    # CRITICAL: Check for well picks formations FIRST, before any other processing
    question_lower_check = question.lower()
    is_formation_query = "formation" in question_lower_check or "formations" in question_lower_check
    
    for msg in messages:
        if isinstance(msg, ToolMessage) and isinstance(msg.content, str):
            content = _extract_tool_content(msg.content)
            # PRIORITY 1: Well picks formations list - MUST bypass LLM to prevent parameter extraction
            if content and content.startswith("[WELL_PICKS]") and "formations:" in content:
                if is_formation_query:
                    # Format the well picks formations list directly - NEVER let LLM process this
                    lines = content.split("\n")
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
            if content and content.startswith("[WELL_PICKS_ALL]"):
                return {"messages": [AIMessage(content=content)]}
            if content and content.startswith("[WELL_FORMATION_PROPERTIES]"):
                return {"messages": [AIMessage(content=content)]}
            if content and content.startswith("[SECTION]"):
                return {"messages": [AIMessage(content=content)]}
            
            # PRIORITY 3: Well picks errors for formation queries
            if content and content.startswith("[WELL_PICKS]") and ("No rows found" in content or "No well detected" in content):
                if is_formation_query:
                    # Return the error message directly - DO NOT let LLM process evaluation parameters
                    logger.warning(f"[ANSWER] Well picks tool returned error for formation query: {content}")
                    # Format the error nicely but don't let LLM extract parameter values
                    error_msg = f"Could not find formations for the specified well. {content}"
                    return {"messages": [AIMessage(content=error_msg)]}
            
            # PRIORITY 4: Check for well detection errors in structured tools
            if isinstance(content, str):
                # Check for well detection errors from structured facts, eval params, petro params
                if ("no_well_detected" in content or "No well detected" in content) and ("error" in content.lower() or "FACTS_JSON" in content or "EVAL_PARAMS_JSON" in content or "PETRO_PARAMS_JSON" in content):
                    logger.warning(f"[ANSWER] Structured tool returned well detection error: {content}")
                    return {"messages": [AIMessage(content="I couldn't identify which well you're asking about. Please specify the well name clearly (e.g., 15/9-F-5, 19/9-19 bt2).")]}
            # PETRO params now flows through a structured JSON formatter below

    # Special deterministic formatting: Evaluation parameters JSON payload -> one clean technical answer.
    for msg in messages:
        if isinstance(msg, ToolMessage) and isinstance(msg.content, str):
            content = _extract_tool_content(msg.content)
            if not content or not content.startswith("[EVAL_PARAMS_JSON]"):
                continue
            
            import json
            raw = content[len("[EVAL_PARAMS_JSON]") :].strip()
            try:
                payload = json.loads(raw)
            except Exception:
                return {"messages": [AIMessage(content="Evaluation parameters: failed to parse structured payload.")]}

            if isinstance(payload, dict) and payload.get("error"):
                error_type = payload.get("error", "")
                error_message = payload.get("message", "")
                well = payload.get("well", "")
                
                # If eval params tool doesn't have data for this well, check if we have retriever context
                # If retriever context exists, use it to generate answer. Otherwise, provide helpful error.
                has_retriever_context = any(
                    isinstance(m, ToolMessage) and 
                    isinstance(m.content, str) and 
                    (hasattr(m, 'name') and "retrieve" in str(m.name).lower() or len(m.content) > 500)
                    for m in messages
                )
                
                if error_type == "no_table_for_well" and has_retriever_context:
                    # Eval params tool doesn't have data, but we have retriever context - use it
                    logger.info(f"[ANSWER] Eval params tool returned 'no_table_for_well' for well {well}, but retriever context available - using retriever context")
                    payload = None
                    break  # Continue to answer generation with retriever context
                elif error_type == "no_table_for_well":
                    # No eval params data AND no retriever context - provide helpful error
                    logger.warning(f"[ANSWER] Eval params tool returned 'no_table_for_well' for well {well}, and no retriever context available")
                    return {"messages": [AIMessage(
                        content=f"No evaluation parameters table was found for well {well} in the structured data. "
                               f"The information may be available in the document text. Please try rephrasing your query or asking about a different well."
                    )]}
                else:
                    # Other error types - let retriever context (if any) drive answer
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

                # Map common query terms to parameter keys (check this FIRST before direct key matching)
                param_synonyms = {
                    "matrix density": "Rhoma",
                    "rhoma": "Rhoma",
                    "ρma": "Rhoma",
                    "rho ma": "Rhoma",
                    "pma": "Rhoma",  # Common shorthand
                    "fluid density": "Rhofl",
                    "rhofl": "Rhofl",
                    "ρfl": "Rhofl",
                    "rho fl": "Rhofl",
                    "pfl": "Rhofl",  # Common shorthand
                    "grmax": "GRmax",
                    "gr max": "GRmax",
                    "gamma ray max": "GRmax",
                    "grmin": "GRmin",
                    "gr min": "GRmin",
                    "gamma ray min": "GRmin",
                    "archie a": "a",
                    "tortuosity factor": "a",
                    "archie m": "m",
                    "cementation exponent": "m",
                    "archie n": "n",
                    "saturation exponent": "n",
                }
                for term, param_key in param_synonyms.items():
                    if term in ql and param_key in params:
                        return param_key

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
        if isinstance(msg, ToolMessage) and isinstance(msg.content, str):
            # Extract actual content (handle Result.ok() string representation)
            content = _extract_tool_content(msg.content)
            if not content or not content.startswith("[PETRO_PARAMS_JSON]"):
                continue
            
            if content.startswith("[PETRO_PARAMS_JSON]"):
                import json

                raw = content[len("[PETRO_PARAMS_JSON]") :].strip()
                try:
                    payload = json.loads(raw)
                except Exception:
                    return {"messages": [AIMessage(content="Petrophysical parameters: failed to parse structured payload.")]}

            if isinstance(payload, dict) and payload.get("error"):
                # Provide helpful deterministic error message instead of breaking
                error_type = payload.get("error", "")
                error_message = payload.get("message", "Unknown error")
                well = payload.get("well", "")
                
                # If well not found, try to suggest available wells
                if error_type == "no_rows_for_well" or "no rows found" in error_message.lower():
                    # Try to get available wells from the cache
                    try:
                        vectorstore_dir = Path(__file__).resolve().parents[3] / "data" / "vectorstore"
                        cache_path = vectorstore_dir / "petro_params_cache.json"
                        if cache_path.exists():
                            with open(cache_path, 'r', encoding='utf-8') as f:
                                cache_data = json.load(f)
                            # Extract unique well names from cache
                            available_wells = set()
                            if isinstance(cache_data, dict) and "rows" in cache_data:
                                for row in cache_data.get("rows", []):
                                    if isinstance(row, dict) and "well" in row:
                                        available_wells.add(row["well"])
                            available_wells_list = sorted(list(available_wells))[:10]
                            
                            if available_wells_list:
                                out = [
                                    f"No petrophysical parameter data found for well {well}.",
                                    "",
                                    f"Available wells with petrophysical data: {', '.join(available_wells_list)}.",
                                    "",
                                    "Please try querying one of the available wells, or check if the well name is correct."
                                ]
                            else:
                                out = [
                                    f"No petrophysical parameter data found for well {well}.",
                                    "",
                                    error_message
                                ]
                        else:
                            out = [
                                f"No petrophysical parameter data found for well {well}.",
                                "",
                                error_message
                            ]
                    except Exception as e:
                        logger.warning(f"[ANSWER] Failed to get available wells: {e}")
                        out = [
                            f"No petrophysical parameter data found for well {well}.",
                            "",
                            error_message
                        ]
                elif error_type == "no_well_detected":
                    out = [
                        "No well detected in your query.",
                        "",
                        "Please specify a well name, for example: 'What is the water saturation for Hugin formation in 15/9-F-5?'"
                    ]
                else:
                    out = [
                        f"Error retrieving petrophysical parameters: {error_message}",
                    ]
                
                return {"messages": [AIMessage(content="\n".join(out))]}

            well = payload.get("well") or ""
            formations = payload.get("formations") or []
            values = payload.get("values") or {}
            sources = payload.get("sources") or []
            ql = question.lower() if isinstance(question, str) else ""

            # Edge case: Handle malformed payload (not a dict, missing required fields)
            if not isinstance(payload, dict):
                return {"messages": [AIMessage(content="Petrophysical parameters: invalid payload format.")]}
            
            # Edge case: Check for additional error types
            if payload.get("error") in ["formation_no_data", "no_data_after_filtering", "no_valid_formations"]:
                error_message = payload.get("message", "Unknown error")
                return {"messages": [AIMessage(content=error_message)]}

            def detect_formation() -> Optional[str]:
                """Detect formation from query, with improved matching and fuzzy matching for typos."""
                if not isinstance(formations, list) or not formations:
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
                
                # Second pass: fuzzy matching for typos (e.g., "sleipmer" -> "Sleipner")
                try:
                    from rapidfuzz import fuzz, process
                    
                    # Extract potential formation names from query
                    # Common formation names in the dataset
                    known_formations = ["draupne", "heather", "hugin", "sleipner", "ekofisk", "hod", "ty", "utsira", 
                                      "skagerrak", "hordaland", "nordland", "shetland", "seabed"]
                    
                    # Check if any known formation appears in query (even with typos)
                    for known_f in known_formations:
                        if known_f in ql_normalized or any(char in ql_normalized for char in known_f[:4]):  # Partial match for typos
                            # Try fuzzy matching against available formations
                            formation_lower_map = {f.lower(): f for f in formations if isinstance(f, str) and f.strip()}
                            if formation_lower_map:
                                hits = process.extract(known_f, list(formation_lower_map.keys()), scorer=fuzz.partial_ratio, limit=3)
                                if hits:
                                    best_match, score, _ = hits[0]
                                    if score >= 80.0:  # 80% match threshold
                                        return formation_lower_map[best_match]
                    
                    # Third pass: direct fuzzy match of query text against formations
                    formation_lower_map = {f.lower(): f for f in formations if isinstance(f, str) and f.strip()}
                    if formation_lower_map:
                        # Extract words from query that might be formation names
                        query_words = [w for w in ql_normalized.split() if len(w) >= 4 and w.isalpha()]
                        for word in query_words:
                            hits = process.extract(word, list(formation_lower_map.keys()), scorer=fuzz.partial_ratio, limit=3)
                            if hits:
                                best_match, score, _ = hits[0]
                                second_score = hits[1][1] if len(hits) > 1 else 0.0
                                if score >= 80.0 and (score - second_score) >= 10.0:  # 80% match with 10% margin
                                    return formation_lower_map[best_match]
                except Exception:
                    pass
                
                # Fourth pass: check if query contains formation name (even if not in formations list)
                # This helps detect when user asks for a formation that's not in the table
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
                # Petro params (from petro_params_cache)
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
                    # Eval params (from eval_params_cache)
                    "matrix density": "Rhoma",
                    "rhoma": "Rhoma",
                    "ρma": "Rhoma",
                    "rho ma": "Rhoma",
                    "pma": "Rhoma",  # Common shorthand
                    "fluid density": "Rhofl",
                    "rhofl": "Rhofl",
                    "ρfl": "Rhofl",
                    "rho fl": "Rhofl",
                    "grmax": "GRmax",
                    "gr max": "GRmax",
                    "gamma ray max": "GRmax",
                    "grmin": "GRmin",
                    "gr min": "GRmin",
                    "gamma ray min": "GRmin",
                    "archie a": "a",
                    "tortuosity factor": "a",
                    "archie m": "m",
                    "cementation exponent": "m",
                    "archie n": "n",
                    "saturation exponent": "n",
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
                # Edge case: row is not a dict (malformed data)
                if not isinstance(row, dict):
                    logger.warning(f"[ANSWER] Row for formation '{f}' is not a dict: {type(row)}")
                    row = {}
                
                # Check if the formation has a value for this parameter
                param_value = row.get(p)
                
                # Edge case: param_value is a string "N/A" or empty string
                if isinstance(param_value, str) and param_value.strip().upper() in ["N/A", "NA", "NONE", ""]:
                    param_value = None
                
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

    # Special deterministic formatting: Evaluation parameters JSON payload -> cell/row/column/full table
    for msg in messages:
        if isinstance(msg, ToolMessage) and isinstance(msg.content, str):
            content = _extract_tool_content(msg.content)
            if not content or not content.startswith("[EVAL_PARAMS_JSON]"):
                continue
            
            if content.startswith("[EVAL_PARAMS_JSON]"):
                import json

                raw = content[len("[EVAL_PARAMS_JSON]") :].strip()
                try:
                    payload = json.loads(raw)
                except Exception:
                    # If parsing fails, continue to LLM answer generation
                    break

            if isinstance(payload, dict) and payload.get("error"):
                # Error handling - let retriever fallback handle it (already implemented above)
                break

            well = payload.get("well") or ""
            formations = payload.get("formations") or []
            params = payload.get("params") or {}  # param -> formation -> value (e.g., {"Rhoma": {"Hugin": "2.65"}})
            notes = payload.get("notes") or []
            source = payload.get("source") or ""
            page_start = payload.get("page_start")
            page_end = payload.get("page_end")
            ql = question.lower() if isinstance(question, str) else ""

            def detect_formation_eval() -> Optional[str]:
                """Detect formation from query for eval params."""
                if not isinstance(formations, list) or not formations:
                    return None
                ql_normalized = ql.replace("fm.", "").replace("fm", "").replace("formation", "").replace("formations", "")
                # Exact match first
                for f in formations:
                    if isinstance(f, str) and f.lower() in ql_normalized:
                        return f
                # Fuzzy match
                try:
                    from rapidfuzz import fuzz, process
                    formation_lower_map = {f.lower(): f for f in formations if isinstance(f, str) and f.strip()}
                    if formation_lower_map:
                        hits = process.extract(ql_normalized, list(formation_lower_map.keys()), scorer=fuzz.partial_ratio, limit=3)
                        if hits:
                            best_match, score, _ = hits[0]
                            if score >= 80.0:
                                return formation_lower_map[best_match]
                except Exception:
                    pass
                return None

            def detect_param_eval() -> Optional[str]:
                """Detect evaluation parameter from query."""
                # Map query terms to eval param names (as stored in cache)
                syn = {
                    "matrix density": "Rhoma",
                    "rhoma": "Rhoma",
                    "ρma": "Rhoma",
                    "rho ma": "Rhoma",
                    "fluid density": "Rhofl",
                    "rhofl": "Rhofl",
                    "ρfl": "Rhofl",
                    "rho fl": "Rhofl",
                    "grmax": "GRmax",
                    "gr max": "GRmax",
                    "gamma ray max": "GRmax",
                    "grmin": "GRmin",
                    "gr min": "GRmin",
                    "gamma ray min": "GRmin",
                    "archie a": "a",
                    "tortuosity factor": "a",
                    "archie m": "m",
                    "cementation exponent": "m",
                    "archie n": "n",
                    "saturation exponent": "n",
                }
                for needle, p in syn.items():
                    if needle in ql:
                        return p
                return None

            def source_line_eval() -> str:
                pages = ""
                if isinstance(page_start, int) and isinstance(page_end, int):
                    pages = f" (pages {page_start}-{page_end})" if page_start != page_end else f" (page {page_start})"
                return f"Source: {source}{pages}".strip()

            f = detect_formation_eval()
            p = detect_param_eval()

            # Cell: parameter + formation
            if f and p and isinstance(params, dict):
                param_data = params.get(p)
                if isinstance(param_data, dict):
                    value = param_data.get(f)
                    if value:
                        out = [
                            f"{p} for {f} formation in well {well}: {value}",
                            "",
                            source_line_eval(),
                        ]
                        if notes:
                            out.insert(-1, f"Notes: {'; '.join(notes[:3])}")
                        return {"messages": [AIMessage(content="\n".join(out))]}
                    else:
                        # Formation found but no value for this parameter
                        available_formations = [ff for ff in formations 
                                               if isinstance(ff, str) and param_data.get(ff)]
                        if available_formations:
                            available_list = ", ".join([f"{ff} ({param_data.get(ff)})" for ff in available_formations])
                            out = [
                                f"{p} for {f} formation in well {well}: N/A",
                                "",
                                f"Note: {f} formation does not have a {p} value in the evaluation parameters table.",
                                f"Available formations with {p} values: {available_list}.",
                                "",
                                source_line_eval(),
                            ]
                        else:
                            out = [
                                f"{p} for {f} formation in well {well}: N/A",
                                "",
                                f"Note: {f} formation does not have a {p} value in the evaluation parameters table for this well.",
                                "",
                                source_line_eval(),
                            ]
                        return {"messages": [AIMessage(content="\n".join(out))]}

            # Row: parameter only (across formations)
            if p and not f and isinstance(params, dict):
                param_data = params.get(p)
                if isinstance(param_data, dict):
                    title = f"{p} — Well {well}"
                    lines = [title, "", "| Formation | Value |", "|---|---:|"]
                    for ff in formations:
                        if isinstance(ff, str):
                            val = param_data.get(ff, "N/A")
                            lines.append(f"| {ff} | {val} |")
                    lines.append("")
                    lines.append(source_line_eval())
                    if notes:
                        lines.insert(-1, f"Notes: {'; '.join(notes[:3])}")
                    return {"messages": [AIMessage(content="\n".join(lines))]}

            # Column: formation only (all parameters for that formation)
            if f and not p and isinstance(params, dict):
                lines = [f"Evaluation parameters — Well {well} — {f}", "", "| Parameter | Value |", "|---|---:|"]
                for param_name, param_data in params.items():
                    if isinstance(param_data, dict):
                        val = param_data.get(f, "N/A")
                        lines.append(f"| {param_name} | {val} |")
                lines.append("")
                lines.append(source_line_eval())
                if notes:
                    lines.insert(-1, f"Notes: {'; '.join(notes[:3])}")
                return {"messages": [AIMessage(content="\n".join(lines))]}

            # Full table (well only / ambiguous)
            if isinstance(params, dict) and isinstance(formations, list):
                # Build a table with all parameters and formations
                param_names = sorted([k for k in params.keys() if isinstance(params.get(k), dict)])
                if param_names and formations:
                    lines = [f"Evaluation parameters — Well {well}", ""]
                    # Header: Formation names
                    header = "| Parameter | " + " | ".join([str(f) for f in formations]) + " |"
                    lines.append(header)
                    lines.append("|" + "---|" * (len(formations) + 1))
                    # Rows: one per parameter
                    for param_name in param_names:
                        param_data = params.get(param_name)
                        if isinstance(param_data, dict):
                            row = "| " + param_name + " | " + " | ".join([str(param_data.get(f, "N/A")) for f in formations]) + " |"
                            lines.append(row)
                    lines.append("")
                    lines.append(source_line_eval())
                    if notes:
                        lines.insert(-1, f"Notes: {'; '.join(notes[:3])}")
                    return {"messages": [AIMessage(content="\n".join(lines))]}

    # Special deterministic formatting: Structured facts JSON payload -> single value or list (no interpretation)
    for msg in messages:
        if isinstance(msg, ToolMessage) and isinstance(msg.content, str):
            content = _extract_tool_content(msg.content)
            if not content or not content.startswith("[FACTS_JSON]"):
                continue
            
            import json
            raw = content[len("[FACTS_JSON]") :].strip()
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
            content = _extract_tool_content(msg.content)
            if content and content.startswith("[WELL_PICKS]") and "formations:" in content:
                has_well_picks_formations = True
                well_picks_formations_content = content
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
        
        # If we're using retriever fallback for eval params, add explicit instructions
        if eval_params_error_with_retriever:
            prompt += "\n\nCRITICAL: The structured evaluation parameters lookup did not find data for this well, but document context is available. You MUST carefully search the document context for the requested information (e.g., matrix density, fluid density, Rhoma, Rhofl, GRmax, GRmin, Archie parameters). Look for tables, evaluation parameters sections, or any mention of these values in the context. If you find the information, extract it precisely. If the information is truly not in the context, state that clearly. Do NOT say the information is unavailable just because the structured lookup failed - the document context may contain it."
    
    response = _get_response_model().invoke([{"role": "user", "content": prompt}])
    
    # Phase 1.5: Extract source citations with page numbers from context
    # FIRST: Extract from structured JSON payloads (PETRO_PARAMS_JSON, EVAL_PARAMS_JSON, etc.)
    citations = []
    
    # Extract citations from structured JSON payloads
    for msg in messages:
        if isinstance(msg, ToolMessage) and isinstance(msg.content, str):
            # Extract actual content (handle Result.ok() string representation)
            content = _extract_tool_content(msg.content)
            if not content:
                continue
            
            # Extract from PETRO_PARAMS_JSON
            if content.startswith("[PETRO_PARAMS_JSON]"):
                import json
                try:
                    raw = content[len("[PETRO_PARAMS_JSON]"):].strip()
                    payload = json.loads(raw)
                    sources = payload.get("sources", [])
                    if isinstance(sources, list) and sources:
                        for s in sources:
                            if isinstance(s, dict):
                                src = s.get("source", "")
                                ps = s.get("page_start")
                                pe = s.get("page_end")
                                if src and src != "N/A":
                                    # Normalize path (handle backslashes)
                                    src_normalized = src.replace('\\', '/')
                                    if isinstance(ps, int) and isinstance(pe, int):
                                        if ps == pe:
                                            citation = f"Source: {src_normalized} (page {ps})"
                                        else:
                                            citation = f"Source: {src_normalized} (pages {ps}-{pe})"
                                    elif isinstance(ps, int):
                                        citation = f"Source: {src_normalized} (page {ps})"
                                    else:
                                        citation = f"Source: {src_normalized}"
                                    if citation not in citations:
                                        citations.append(citation)
                                        logger.debug(f"[ANSWER] Extracted citation from PETRO_PARAMS_JSON: {citation}")
                except Exception as e:
                    logger.warning(f"[ANSWER] Failed to extract citations from PETRO_PARAMS_JSON: {e}")
            
            # Extract from EVAL_PARAMS_JSON
            elif content.startswith("[EVAL_PARAMS_JSON]"):
                import json
                try:
                    raw = content[len("[EVAL_PARAMS_JSON]"):].strip()
                    payload = json.loads(raw)
                    source = payload.get("source", "")
                    ps = payload.get("page_start")
                    pe = payload.get("page_end")
                    if source and source != "N/A":
                        source_normalized = source.replace('\\', '/')
                        if isinstance(ps, int) and isinstance(pe, int):
                            if ps == pe:
                                citation = f"Source: {source_normalized} (page {ps})"
                            else:
                                citation = f"Source: {source_normalized} (pages {ps}-{pe})"
                        elif isinstance(ps, int):
                            citation = f"Source: {source_normalized} (page {ps})"
                        else:
                            citation = f"Source: {source_normalized}"
                        if citation not in citations:
                            citations.append(citation)
                            logger.debug(f"[ANSWER] Extracted citation from EVAL_PARAMS_JSON: {citation}")
                except Exception as e:
                    logger.warning(f"[ANSWER] Failed to extract citations from EVAL_PARAMS_JSON: {e}")
            
            # Extract from FACTS_JSON
            elif content.startswith("[FACTS_JSON]"):
                import json
                try:
                    raw = content[len("[FACTS_JSON]"):].strip()
                    payload = json.loads(raw)
                    matches = payload.get("matches", [])
                    if isinstance(matches, list):
                        for match in matches:
                            if isinstance(match, dict):
                                src = match.get("source", "")
                                ps = match.get("page_start")
                                pe = match.get("page_end")
                                if src and src != "N/A":
                                    src_normalized = src.replace('\\', '/')
                                    if isinstance(ps, int) and isinstance(pe, int):
                                        if ps == pe:
                                            citation = f"Source: {src_normalized} (page {ps})"
                                        else:
                                            citation = f"Source: {src_normalized} (pages {ps}-{pe})"
                                    elif isinstance(ps, int):
                                        citation = f"Source: {src_normalized} (page {ps})"
                                    else:
                                        citation = f"Source: {src_normalized}"
                                    if citation not in citations:
                                        citations.append(citation)
                                        logger.debug(f"[ANSWER] Extracted citation from FACTS_JSON: {citation}")
                except Exception as e:
                    logger.warning(f"[ANSWER] Failed to extract citations from FACTS_JSON: {e}")
    
    # THEN: Extract citations from [Source: ...] format in tool messages and context
    citation_pattern = r'\[Source:\s*([^\]]+?)\s*(?:\(page\s+(\d+)\)|\(pages\s+(\d+)\s*-\s*(\d+)\))?\]'
    
    for msg in messages:
        if isinstance(msg, ToolMessage) and isinstance(msg.content, str):
            content = _extract_tool_content(msg.content)
            # Skip JSON payloads (already processed above)
            if content and content.startswith(("[PETRO_PARAMS_JSON]", "[EVAL_PARAMS_JSON]", "[FACTS_JSON]")):
                continue
            # Extract citations from tool message content
            if content:
                matches = re.findall(citation_pattern, content)
            else:
                matches = []
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

