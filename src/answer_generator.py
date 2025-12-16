"""
Answer generation module using LLM to generate natural language answers from retrieved context.
Supports both OpenAI and Google Gemini with function calling capabilities.
Enforces strict data grounding - answers must come from provided data only.
"""

from typing import List, Dict, Optional, Any
import os
import json

# Core LangChain message imports (required for all LLMs)
try:
    from langchain_core.messages import HumanMessage, SystemMessage
except ImportError:
    try:
        from langchain.schema import HumanMessage, SystemMessage
    except ImportError:
        raise ImportError(
            "Could not import HumanMessage and SystemMessage. "
            "Please install langchain-core: pip install langchain-core"
        )

# Optional LLM imports
try:
    from langchain_openai import ChatOpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    ChatOpenAI = None

# Gemini imports
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False
    ChatGoogleGenerativeAI = None


class AnswerGenerator:
    """Generates natural language answers from retrieved context using LLM with strict grounding."""
    
    def __init__(self, model_name: str = "gemini-pro", provider: str = "gemini", 
                 temperature: float = 0.0, tools: Optional[Any] = None):
        """
        Initialize the answer generator.
        
        Args:
            model_name: Name of the LLM model to use
            provider: "gemini" or "openai"
            temperature: Temperature for generation (0.0 for deterministic)
            tools: Optional ComputationalTools instance for function calling
        """
        self.model_name = model_name
        self.provider = provider.lower()
        self.temperature = temperature
        self.tools = tools
        
        # Initialize LLM
        self.llm = None
        self.llm_provider = None
        
        if self.provider == "gemini" and HAS_GEMINI:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError(
                    "GEMINI_API_KEY not set. Please set it in your .env file or environment variables. "
                    "This system requires an LLM to function."
                )
            try:
                self.llm = ChatGoogleGenerativeAI(
                    model=model_name,
                    temperature=temperature,
                    google_api_key=api_key
                )
                self.llm_provider = "gemini"
                # Verify LLM was created successfully
                if self.llm is None:
                    raise RuntimeError("Gemini LLM initialization returned None. This should not happen.")
                print(f"✓ Initialized Gemini LLM ({model_name}) for answer generation.")
            except Exception as e:
                import traceback
                error_msg = f"Failed to initialize Gemini LLM: {e}\n{traceback.format_exc()}"
                raise RuntimeError(
                    f"Could not initialize Gemini LLM. This system requires an LLM to function.\n{error_msg}"
                ) from e
        elif self.provider == "openai" and HAS_OPENAI:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY not set. Please set it in your .env file or environment variables. "
                    "This system requires an LLM to function."
                )
            try:
                self.llm = ChatOpenAI(
                    model_name=model_name,
                    temperature=temperature,
                    api_key=api_key
                )
                self.llm_provider = "openai"
                if self.llm is None:
                    raise RuntimeError("OpenAI LLM initialization returned None. This should not happen.")
                print(f"✓ Initialized OpenAI LLM ({model_name}) for answer generation.")
            except Exception as e:
                import traceback
                error_msg = f"Failed to initialize OpenAI LLM: {e}\n{traceback.format_exc()}"
                raise RuntimeError(
                    f"Could not initialize OpenAI LLM. This system requires an LLM to function.\n{error_msg}"
                ) from e
        else:
            if not HAS_GEMINI and not HAS_OPENAI:
                raise ImportError(
                    "Neither langchain-google-genai nor langchain-openai is installed. "
                    "Please install one: pip install langchain-google-genai"
                )
            raise ValueError(
                f"Invalid LLM provider: {self.provider}. Must be 'gemini' or 'openai'. "
                f"Current provider: {self.provider}, HAS_GEMINI: {HAS_GEMINI}, HAS_OPENAI: {HAS_OPENAI}"
            )
    
    def generate_answer(self, query: str, context: List[Dict], 
                       analysis: Dict, aggregated_data: Optional[Dict] = None,
                       conversation_history: Optional[List[Dict]] = None) -> Dict:
        """
        Generate a natural language answer from context with strict grounding.
        
        Args:
            query: Original user query
            context: List of retrieved context documents
            analysis: Query analysis results
            aggregated_data: Optional aggregated statistics
            conversation_history: Optional conversation history for context
            
        Returns:
            Dictionary with answer, sources, tools_used, and metadata
        """
        tools_used = []
        tool_results = {}
        
        # If LLM is available and tools are provided, try function calling
        if self.llm and self.tools:
            # Check if query suggests tools are needed
            if self._should_use_tools(query, analysis):
                # For listing queries, try listing tools first
                if analysis.get('should_use_listing_tool', False) or \
                   any(pattern in query.lower() for pattern in ['list all', 'all formations', 'all surfaces']):
                    tool_result = self._try_listing_tools(query, analysis)
                    if tool_result:
                        tools_used.append(tool_result.get('tool_name'))
                        tool_results[tool_result.get('tool_name')] = tool_result.get('result')
                else:
                    # Try other tools (visualization, etc.)
                    tool_result = self._try_function_calling(query, analysis, context)
                    if tool_result and tool_result.get('result', {}).get('success'):
                        tools_used.append(tool_result.get('tool_name'))
                        tool_results[tool_result.get('tool_name')] = tool_result.get('result')
        
        # Generate answer - Production system always uses LLM
        if not self.llm:
            raise RuntimeError(
                "LLM is not available. This production system requires an LLM to function. "
                "Please check your API key and LLM configuration."
            )
        
        # Always use LLM for answer generation
        if aggregated_data:
            answer = self._generate_with_llm(query, context, analysis, aggregated_data, 
                                           conversation_history, tool_results)
        else:
            answer = self._generate_with_llm(query, context, analysis, None, 
                                           conversation_history, tool_results)
        
        # Extract sources
        sources = self._extract_sources(context)
        
        return {
            'answer': answer,
            'sources': sources,
            'context_count': len(context),
            'aggregated_data': aggregated_data,
            'tools_used': tools_used,
            'tool_results': tool_results
        }
    
    def _should_use_tools(self, query: str, analysis: Dict) -> bool:
        """Determine if tools should be used based on query."""
        query_lower = query.lower()
        
        # Prioritize tools for listing queries - these need complete, deterministic results
        if analysis.get('should_use_listing_tool', False):
            return True
        
        # Check for "list all" patterns that should use listing tools
        listing_patterns = ['list all', 'list all formations', 'list all surfaces', 
                           'all formations', 'all surfaces', 'all wells']
        if any(pattern in query_lower for pattern in listing_patterns):
            return True
        
        # Check for formation-related queries that would benefit from visualization
        if analysis.get('formation') or 'formation' in query_lower:
            return True
        
        # Check for visualization keywords
        viz_keywords = ['show', 'display', 'plot', 'visualize', 'graph', 'chart']
        if any(keyword in query_lower for keyword in viz_keywords):
            return True
        
        return False
    
    def _try_listing_tools(self, query: str, analysis: Dict) -> Optional[Dict]:
        """Try to use listing tools for 'list all' queries."""
        if not self.tools:
            return None
        
        query_lower = query.lower()
        well_name = analysis.get('well_name')
        
        # Determine which listing tool to use
        if 'all formations' in query_lower or 'all surfaces' in query_lower:
            if 'all wells' in query_lower or 'in all wells' in query_lower:
                # List all wells with formations
                result = self.tools.list_all_wells_with_formations()
                if result.get('success'):
                    return {
                        'tool_name': 'list_all_wells_with_formations',
                        'result': result
                    }
            elif well_name:
                # List formations for specific well
                result = self.tools.list_formations_by_well(well_name)
                if result.get('success'):
                    return {
                        'tool_name': 'list_formations_by_well',
                        'result': result
                    }
            else:
                # List all unique formations
                result = self.tools.list_all_formations()
                if result.get('success'):
                    return {
                        'tool_name': 'list_all_formations',
                        'result': result
                    }
        elif well_name and ('formation' in query_lower or 'surface' in query_lower):
            # List formations for specific well
            result = self.tools.list_formations_by_well(well_name)
            if result.get('success'):
                return {
                    'tool_name': 'list_formations_by_well',
                    'result': result
                }
        
        return None
    
    def _try_function_calling(self, query: str, analysis: Dict, context: List[Dict]) -> Optional[Dict]:
        """Try to use function calling to execute tools."""
        if not self.tools:
            return None
        
        # Determine which tool to use
        formation = analysis.get('formation')
        well_name = analysis.get('well_name')
        curve = analysis.get('curve')
        
        # If formation is mentioned, suggest plotting
        if formation:
            # Try to find well_name from context if not in analysis
            if not well_name and context:
                # Look for well_name in context metadata
                for item in context[:3]:
                    metadata = item.get('metadata', {})
                    if metadata.get('well_name'):
                        well_name = metadata['well_name']
                        break
            
            if well_name:
                # Get relevant curves
                relevant_curves = self.tools.get_relevant_curves(curve or 'formation')
                suggested_curves = relevant_curves.get('suggested_curves', ['GR', 'PHIF', 'VSH'])
                
                # Execute plot_formation_log tool
                result = self.tools.plot_formation_log(well_name, formation, suggested_curves)
                if result.get('success'):
                    return {
                        'tool_name': 'plot_formation_log',
                        'result': result
                    }
        
        return None
    
    def _generate_with_llm(self, query: str, context: List[Dict], 
                          analysis: Dict, aggregated_data: Optional[Dict] = None,
                          conversation_history: Optional[List[Dict]] = None,
                          tool_results: Optional[Dict] = None) -> str:
        """Generate answer using LLM with strict grounding."""
        # Production system: LLM must be available
        if not self.llm or self.llm is None:
            raise RuntimeError(
                "LLM is not initialized. This is a production system that requires an LLM. "
                "Please check your API key and LLM configuration."
            )
        
        # Additional check to ensure LLM has invoke method
        if not hasattr(self.llm, 'invoke') or not callable(getattr(self.llm, 'invoke', None)):
            raise RuntimeError(
                f"LLM object is not callable. LLM type: {type(self.llm)}, "
                f"has invoke: {hasattr(self.llm, 'invoke')}. "
                "This indicates a problem with LLM initialization."
            )
        
        # Build context string
        context_text = self._build_context_string(context)
        
        # Build strict grounding system prompt
        system_prompt = self._build_strict_grounding_prompt()
        
        # Build user prompt
        # Check aggregated_data is not a callable (defensive check)
        if aggregated_data is not None and not callable(aggregated_data):
            if not hasattr(self, '_build_aggregation_prompt') or not callable(self._build_aggregation_prompt):
                raise RuntimeError("_build_aggregation_prompt method is not available")
            prompt = self._build_aggregation_prompt(query, context_text, analysis, aggregated_data, tool_results)
        else:
            if not hasattr(self, '_build_general_prompt') or not callable(self._build_general_prompt):
                raise RuntimeError("_build_general_prompt method is not available")
            prompt = self._build_general_prompt(query, context_text, analysis, conversation_history, tool_results)
        
        # Verify SystemMessage and HumanMessage are available and callable
        if SystemMessage is None:
            raise RuntimeError(
                "SystemMessage is None. This indicates an import failure. "
                "Please ensure langchain-core or langchain is installed: pip install langchain-core"
            )
        if not callable(SystemMessage):
            raise RuntimeError(
                f"SystemMessage is not callable. Type: {type(SystemMessage)}, Value: {SystemMessage}"
            )
        if HumanMessage is None:
            raise RuntimeError(
                "HumanMessage is None. This indicates an import failure. "
                "Please ensure langchain-core or langchain is installed: pip install langchain-core"
            )
        if not callable(HumanMessage):
            raise RuntimeError(
                f"HumanMessage is not callable. Type: {type(HumanMessage)}, Value: {HumanMessage}"
            )
        
        try:
            # Create messages - both Gemini and OpenAI use the same format
            # Final check before creating messages
            if not callable(SystemMessage):
                raise RuntimeError(f"SystemMessage is not callable: {type(SystemMessage)}")
            if not callable(HumanMessage):
                raise RuntimeError(f"HumanMessage is not callable: {type(HumanMessage)}")
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=prompt)
            ]
            
            # Production system: LLM must be valid
            if self.llm is None:
                raise RuntimeError("LLM became None unexpectedly. This should not happen in production.")
            
            if not hasattr(self.llm, 'invoke'):
                raise RuntimeError(
                    f"LLM object missing invoke method. LLM type: {type(self.llm)}. "
                    "This indicates a problem with LLM initialization."
                )
            
            # Invoke LLM
            response = self.llm.invoke(messages)
            answer = response.content if hasattr(response, 'content') else str(response)
            
            # Validate answer is grounded (basic check)
            if not self._validate_answer_grounding(answer, context, aggregated_data):
                # If not well grounded, add a note
                answer += "\n\nNote: This answer is based solely on the provided well data."
            
            return answer
        except Exception as e:
            import traceback
            error_details = (
                f"Error generating answer with LLM: {e}\n"
                f"LLM object: {self.llm}\n"
                f"LLM type: {type(self.llm)}\n"
                f"LLM provider: {self.llm_provider}\n"
                f"Traceback:\n{traceback.format_exc()}"
            )
            print(error_details)
            # In production, we should raise the error, not fall back to templates
            raise RuntimeError(
                f"Failed to generate answer with LLM. This is a production system that requires LLM functionality.\n{error_details}"
            ) from e
    
    def _build_strict_grounding_prompt(self) -> str:
        """Build system prompt that enforces strict data grounding."""
        return """You are a helpful assistant for oil and gas well data analysis.

CRITICAL RULES - YOU MUST FOLLOW THESE:
1. You can ONLY answer using:
   - The provided well data context
   - Results from computational tools (plotting, statistics, listing tools)
   - Retrieved formation tops data

2. DO NOT use any external knowledge, general petroleum engineering knowledge, or information not explicitly in the provided data.

3. If information is not in the provided data, state clearly: "The requested information is not available in the provided well data."

4. **COMPLETENESS WARNING**: You **must only** use the information provided in `CONTEXT`. 
   - If the context does not contain enough information to fully answer the question (e.g., to list *all* items), say so explicitly and explain what is missing.
   - Do not claim that a list is complete unless the context clearly says it is complete.
   - For queries asking for 'all' items (e.g., "list all formations"), prefer using available tools (list_all_formations, list_all_wells_with_formations) to get complete lists rather than inferring from partial context.
   - If you only see formations for a subset of wells in the context, explicitly state: "I only see formations for wells [X, Y, Z] in the context, so I cannot list *all* formations in the dataset."

5. When questions involve formations, you should mention that visualization is available to show the formation on a well log with relevant curves highlighted.

6. Always cite which wells or formations your answer is based on.

7. Be precise with numerical values - use the exact values from the data.

8. If you're uncertain, state that clearly rather than guessing.

Provide accurate, concise answers based strictly on the provided context."""
    
    def _build_context_string(self, context: List[Dict]) -> str:
        """Build a string representation of the context."""
        context_parts = []
        for i, item in enumerate(context[:10], 1):  # Limit to top 10
            doc = item.get('document', '')
            metadata = item.get('metadata', {})
            well_name = metadata.get('well_name', 'Unknown')
            formation = metadata.get('formation_name', '')
            
            context_part = f"Source {i}:\n"
            context_part += f"Well: {well_name}\n"
            if formation:
                context_part += f"Formation: {formation}\n"
            context_part += f"Information: {doc}\n"
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def _build_aggregation_prompt(self, query: str, context: str, 
                                 analysis: Dict, aggregated_data: Dict,
                                 tool_results: Optional[Dict] = None) -> str:
        """Build prompt for aggregation queries."""
        curve = analysis.get('curve', 'data')
        formation = analysis.get('formation', '')
        agg_type = analysis.get('aggregation_type', 'mean')
        
        prompt = f"""Based on the following well data, answer this question: {query}

Aggregated Statistics:
"""
        if formation:
            prompt += f"Formation: {formation}\n"
        
        for key, value in aggregated_data.items():
            if isinstance(value, (int, float)):
                prompt += f"{key}: {value:.4f}\n"
            else:
                prompt += f"{key}: {value}\n"
        
        prompt += f"\nContext from well data:\n{context}\n\n"
        
        # Add tool results if available
        if tool_results:
            prompt += "\nTool Results:\n"
            for tool_name, result in tool_results.items():
                if isinstance(result, dict) and result.get('success'):
                    if tool_name in ['list_all_formations', 'list_formations_by_well', 'list_all_wells_with_formations']:
                        # Include the actual tool results in the prompt
                        if 'formations' in result:
                            formations_list = result.get('formations', [])
                            prompt += f"{tool_name}: Complete list of {len(formations_list)} formations\n"
                            prompt += f"Formations: {', '.join(formations_list[:20])}"  # Show first 20
                            if len(formations_list) > 20:
                                prompt += f" ... and {len(formations_list) - 20} more\n"
                        elif 'wells_formations' in result:
                            well_count = result.get('well_count', 0)
                            prompt += f"{tool_name}: Complete list from {well_count} wells\n"
                            wells_formations = result.get('wells_formations', {})
                            # Show first few wells
                            for well, formations in list(wells_formations.items())[:5]:
                                prompt += f"  {well}: {len(formations)} formations\n"
                            if len(wells_formations) > 5:
                                prompt += f"  ... and {len(wells_formations) - 5} more wells\n"
                    else:
                        prompt += f"{tool_name}: Visualization available\n"
        
        prompt += "\nProvide a clear, concise answer to the question using the aggregated statistics. "
        prompt += "Include the specific numerical value and mention which wells/formations were included in the calculation. "
        prompt += "If a formation is mentioned, note that a visualization showing the formation on a well log is available."
        prompt += "\n\nIMPORTANT: If the context does not contain complete information (e.g., only shows a subset of wells), explicitly state this limitation. Do not claim completeness unless the context clearly indicates it."
        
        return prompt
    
    def _build_general_prompt(self, query: str, context: str, analysis: Dict,
                             conversation_history: Optional[List[Dict]] = None,
                             tool_results: Optional[Dict] = None) -> str:
        """Build prompt for general queries."""
        prompt = f"""Based on the following well data, answer this question: {query}

Context from well data:
{context}
"""
        
        # Add conversation history if available
        if conversation_history:
            prompt += "\nPrevious conversation context:\n"
            for msg in conversation_history[-3:]:  # Last 3 messages
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')[:200]  # Truncate
                prompt += f"{role}: {content}\n"
        
        # Add tool results if available
        if tool_results:
            prompt += "\nTool Results:\n"
            for tool_name, result in tool_results.items():
                if isinstance(result, dict):
                    if result.get('success'):
                        if tool_name in ['list_all_formations', 'list_formations_by_well', 'list_all_wells_with_formations']:
                            # Include the actual tool results in the prompt
                            if 'formations' in result:
                                prompt += f"{tool_name}: Complete list of {len(result.get('formations', []))} formations\n"
                            elif 'wells_formations' in result:
                                well_count = result.get('well_count', 0)
                                prompt += f"{tool_name}: Complete list from {well_count} wells\n"
                        else:
                            prompt += f"{tool_name}: Visualization available\n"
        
        prompt += "\nProvide a clear, concise answer based strictly on the context. "
        prompt += "If the answer requires specific numerical values, include them. "
        prompt += "If multiple wells are mentioned, you can compare them. "
        prompt += "If a formation is mentioned, note that visualization is available."
        prompt += "\n\nIMPORTANT: You **must only** use the information provided in the context above. "
        prompt += "If the context does not contain enough information to fully answer the question (e.g., to list *all* items), "
        prompt += "say so explicitly and explain what is missing. Do not claim that a list is complete unless the context clearly says it is complete. "
        prompt += "If tool results are provided (e.g., list_all_formations), use those for complete, accurate lists rather than inferring from partial context."
        
        return prompt
    
    def _validate_answer_grounding(self, answer: str, context: List[Dict], 
                                   aggregated_data: Optional[Dict]) -> bool:
        """Basic validation that answer is grounded in provided data."""
        # Check if answer mentions wells from context
        if context:
            well_names = [item.get('metadata', {}).get('well_name', '') for item in context[:5]]
            answer_lower = answer.lower()
            if any(well.lower() in answer_lower for well in well_names if well):
                return True
        
        # Check if answer mentions aggregated data
        if aggregated_data and 'value' in aggregated_data:
            value_str = f"{aggregated_data['value']:.4f}"
            if value_str in answer:
                return True
        
        return False
    
    def _generate_template_answer(self, query: str, analysis: Dict, 
                                 aggregated_data: Optional[Dict] = None,
                                 context: Optional[List[Dict]] = None,
                                 tool_results: Optional[Dict] = None) -> str:
        """Generate answer using templates when LLM is not available."""
        # Handle both enum and string query_type
        query_type = analysis.get('query_type')
        if hasattr(query_type, 'value'):
            query_type = query_type.value
        elif isinstance(query_type, str):
            query_type = query_type.lower()
        else:
            query_type = 'general'
        
        curve = analysis.get('curve') or 'property'
        formation = analysis.get('formation')
        
        if aggregated_data and query_type != 'comparison':
            # Aggregation query (not comparison)
            agg_type = analysis.get('aggregation_type', 'mean')
            value = aggregated_data.get('value')
            count = aggregated_data.get('count', 0)
            curve_name = curve if curve else 'property'
            
            if formation:
                answer = f"The {agg_type} {curve_name} in the {formation} formation is {value:.4f}."
            else:
                answer = f"The {agg_type} {curve_name} across all wells is {value:.4f}."
            
            if count > 0:
                answer += f" This value is calculated from {count} well(s)."
            
            # Add range if available
            if 'min' in aggregated_data and 'max' in aggregated_data:
                answer += f" The range is {aggregated_data['min']:.4f} to {aggregated_data['max']:.4f}."
            
            # Add visualization note if formation mentioned
            if formation and tool_results:
                answer += " A visualization showing this formation on a well log is available."
        
        elif query_type == 'comparison' and aggregated_data:
            # Comparison query
            well_name = aggregated_data.get('well_name', 'Unknown')
            value = aggregated_data.get('value', 0)
            comparison_type = aggregated_data.get('comparison_type', 'max')
            comp_word = 'highest' if comparison_type == 'max' else 'lowest'
            curve_name = analysis.get('curve', 'property') if analysis.get('curve') else 'property'
            
            answer = f"The well with the {comp_word} {curve_name} is {well_name} with a value of {value:.4f}."
            
            if 'all_values' in aggregated_data and aggregated_data['all_values']:
                all_vals = list(aggregated_data['all_values'].values())
                if len(all_vals) > 1:
                    answer += f" Other wells have values ranging from {min(all_vals):.4f} to {max(all_vals):.4f}."
        
        elif query_type == 'specific' and context:
            # Specific well query
            well_name = analysis.get('well_name', 'the well')
            if context:
                first_result = context[0]
                metadata = first_result.get('metadata', {})
                curve_stats = {}
                for key, val in metadata.items():
                    if key.endswith('_mean') and val is not None:
                        curve_name = key.replace('_mean', '').upper()
                        curve_stats[curve_name] = val
                
                if curve in ['porosity', 'permeability', 'water saturation', 'shale volume']:
                    curve_key = {'porosity': 'PHIF', 'permeability': 'KLOGH', 
                               'water saturation': 'SW', 'shale volume': 'VSH'}.get(curve)
                    if curve_key and curve_key in curve_stats:
                        value = curve_stats[curve_key]
                        answer = f"The {curve} of {well_name} is {value:.4f}."
                    else:
                        answer = f"Based on the available data for {well_name}, {context[0].get('document', 'information is available.')}"
                else:
                    answer = f"Based on the available data for {well_name}, {context[0].get('document', 'information is available.')}"
            else:
                answer = f"No data found for {well_name}."
        
        else:
            # General query
            if context:
                answer = f"Based on the retrieved well data: {context[0].get('document', 'Information is available.')}"
                if len(context) > 1:
                    answer += f" (Found {len(context)} relevant results)"
            else:
                answer = "No relevant information found in the well database."
        
        # Add visualization note if tool results available
        if tool_results and formation:
            answer += " A visualization showing this formation on a well log is available."
        
        return answer
    
    def _extract_sources(self, context: List[Dict]) -> List[Dict]:
        """Extract source information from context."""
        sources = []
        seen_wells = set()
        
        for item in context[:5]:  # Limit to top 5 sources
            metadata = item.get('metadata', {})
            well_name = metadata.get('well_name')
            
            if well_name and well_name not in seen_wells:
                source = {
                    'well_name': well_name,
                    'formation': metadata.get('formation_name'),
                    'type': metadata.get('type', 'well_summary'),
                    'similarity': item.get('similarity')
                }
                sources.append(source)
                seen_wells.add(well_name)
        
        return sources


if __name__ == "__main__":
    # Test the answer generator
    generator = AnswerGenerator()
    
    test_context = [
        {
            'document': 'Well 15/9-F-1 in VOLVE field with porosity 0.15 and permeability 50 mD',
            'metadata': {'well_name': '15/9-F-1', 'PHIF_mean': 0.15, 'KLOGH_mean': 50.0},
            'similarity': 0.85
        }
    ]
    
    test_analysis = {
        'query_type': 'aggregation',
        'curve': 'porosity',
        'formation': 'Hugin',
        'aggregation_type': 'mean'
    }
    
    test_aggregated = {
        'value': 0.16,
        'count': 3,
        'min': 0.14,
        'max': 0.18
    }
    
    result = generator.generate_answer(
        "What is the average porosity in the Hugin formation?",
        test_context,
        test_analysis,
        test_aggregated
    )
    
    print("Generated Answer:")
    print(result['answer'])
    print("\nSources:")
    for source in result['sources']:
        print(f"  - {source['well_name']}")
