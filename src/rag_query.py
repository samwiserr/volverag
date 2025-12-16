"""
RAG query module for processing natural language queries and semantic search.
Enhanced with answer generation, aggregation capabilities, tools, and conversation context.
"""

import re
from typing import List, Dict, Optional, Tuple
from src.embeddings import EmbeddingGenerator
from src.vector_store import VectorStore
from src.query_analyzer import QueryAnalyzer, QueryType
from src.aggregator import DataAggregator
from src.answer_generator import AnswerGenerator
from src.tools import ComputationalTools
from src.data_access import DataAccess
from src.conversation_manager import ConversationManager
from src.answer_generator import AnswerGenerator


class RAGQueryProcessor:
    """Processes queries and performs retrieval-augmented generation."""
    
    def __init__(self, vector_store: VectorStore, embedding_generator: EmbeddingGenerator,
                 formation_tops_data: Optional[Dict] = None, 
                 llm_provider: str = "gemini", llm_model: str = "gemini-pro"):
        """
        Initialize the RAG query processor.
        
        Args:
            vector_store: VectorStore instance
            embedding_generator: EmbeddingGenerator instance
            formation_tops_data: Optional formation tops data for filtering
            llm_provider: LLM provider ("gemini" or "openai")
            llm_model: LLM model name
        """
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
        self.query_analyzer = QueryAnalyzer()
        self.aggregator = DataAggregator()
        self.formation_tops_data = formation_tops_data or {}
        
        # Initialize data access and tools
        self.data_access = DataAccess()
        self.tools = ComputationalTools(self.data_access, formation_tops_data=self.formation_tops_data)
        
        # Initialize answer generator with tools
        self.answer_generator = AnswerGenerator(
            model_name=llm_model,
            provider=llm_provider,
            tools=self.tools
        )
        
        # Initialize conversation manager
        self.conversation_manager = ConversationManager()
    
    def _safe_get_list_length(self, results: Dict, key: str, default: List = None) -> int:
        """
        Safely get the length of a nested list in results.
        
        Args:
            results: Results dictionary
            key: Key to look up (e.g., 'ids', 'metadatas')
            default: Default value if key doesn't exist
            
        Returns:
            Length of the list (0 if empty or doesn't exist)
        """
        if default is None:
            default = [[]]
        
        value = results.get(key, default)
        if not value:
            return 0
        
        # Handle nested structure (from query())
        if isinstance(value, list) and len(value) > 0:
            if isinstance(value[0], list):
                return len(value[0])
            else:
                return len(value)
        
        return 0
    
    def parse_structured_query(self, query: str) -> Optional[Dict]:
        """
        Parse structured queries (e.g., "porosity > 0.2", "well name contains F-1").
        
        Args:
            query: Query string
            
        Returns:
            Dictionary with parsed query components or None if not a structured query
        """
        query_lower = query.lower()
        
        # Pattern for numeric comparisons: "porosity > 0.2", "permeability < 100"
        numeric_patterns = {
            'phif': r'porosity\s*(>|<|>=|<=|==|=)\s*([\d.]+)',
            'klogh': r'permeability\s*(>|<|>=|<=|==|=)\s*([\d.]+)',
            'sw': r'(?:water\s+)?saturation\s*(>|<|>=|<=|==|=)\s*([\d.]+)',
            'vsh': r'shale\s*(?:volume)?\s*(>|<|>=|<=|==|=)\s*([\d.]+)',
        }
        
        parsed = {}
        
        for curve_key, pattern in numeric_patterns.items():
            match = re.search(pattern, query_lower)
            if match:
                operator = match.group(1).replace('=', '==')
                value = float(match.group(2))
                parsed[curve_key] = {'operator': operator, 'value': value}
        
        # Pattern for well name: "well name contains X", "wells like X"
        well_pattern = r'well\s*(?:name\s+)?(?:contains|like|is)\s+([A-Z0-9/\-\s]+)'
        well_match = re.search(well_pattern, query_lower)
        if well_match:
            parsed['well_name'] = well_match.group(1).strip()
        
        # Pattern for formation: "formation X", "in formation X"
        formation_pattern = r'(?:formation|in)\s+([A-Za-z\s]+?)(?:\s+formation|$)'
        formation_match = re.search(formation_pattern, query_lower)
        if formation_match:
            parsed['formation'] = formation_match.group(1).strip()
        
        return parsed if parsed else None
    
    def execute_structured_query(self, parsed_query: Dict, n_results: int = 20) -> Dict:
        """
        Execute a structured query using metadata filters.
        
        Args:
            parsed_query: Parsed query dictionary
            n_results: Number of results to return
            
        Returns:
            Dictionary with search results
        """
        # Build metadata filter
        where_clause = {}
        
        # Filter by well name
        if 'well_name' in parsed_query:
            # ChromaDB supports partial matching with $contains
            where_clause['well_name'] = {'$contains': parsed_query['well_name']}
        
        # Filter by formation
        if 'formation' in parsed_query:
            where_clause['formation_name'] = {'$contains': parsed_query['formation']}
        
        # Filter by curve values (these need to be applied after retrieval)
        curve_filters = {}
        for curve_key, filter_info in parsed_query.items():
            if curve_key in ['phif', 'klogh', 'sw', 'vsh']:
                curve_filters[curve_key] = filter_info
        
        # Perform search with metadata filters
        if where_clause:
            results = self.vector_store.filter_by_metadata(where_clause, n_results=n_results * 2)
        else:
            # Get all results if no metadata filters
            results = self.vector_store.filter_by_metadata({}, n_results=n_results * 2)
        
        # Apply curve value filters
        if curve_filters and results.get('metadatas'):
            filtered_results = self._apply_curve_filters(results, curve_filters)
            # Limit to n_results
            if self._safe_get_list_length(filtered_results, 'ids') > n_results:
                for key in filtered_results:
                    if isinstance(filtered_results[key], list) and len(filtered_results[key]) > 0:
                        if isinstance(filtered_results[key][0], list):
                            filtered_results[key][0] = filtered_results[key][0][:n_results]
            return filtered_results
        
        # Limit results
        if results.get('ids') and len(results['ids']) > n_results:
            for key in results:
                if isinstance(results[key], list) and len(results[key]) > 0:
                    results[key][0] = results[key][0][:n_results]
        
        return results
    
    def _apply_curve_filters(self, results: Dict, curve_filters: Dict) -> Dict:
        """
        Apply curve value filters to results.
        
        Args:
            results: Search results dictionary
            curve_filters: Dictionary of curve filters
            
        Returns:
            Filtered results dictionary
        """
        if not results.get('metadatas') or not results['metadatas']:
            return results
        
        filtered_indices = []
        metadatas = results['metadatas']
        
        for i, metadata in enumerate(metadatas):
            if not isinstance(metadata, dict):
                continue
            
            matches = True
            for curve_key, filter_info in curve_filters.items():
                operator = filter_info['operator']
                value = filter_info['value']
                
                # Look for curve mean value in metadata
                metadata_key = f'{curve_key}_mean'
                if metadata_key in metadata:
                    curve_value = metadata[metadata_key]
                    if curve_value is not None:
                        # Apply filter
                        if operator == '>':
                            if not (curve_value > value):
                                matches = False
                                break
                        elif operator == '<':
                            if not (curve_value < value):
                                matches = False
                                break
                        elif operator == '>=':
                            if not (curve_value >= value):
                                matches = False
                                break
                        elif operator == '<=':
                            if not (curve_value <= value):
                                matches = False
                                break
                        elif operator in ['==', '=']:
                            if not (abs(curve_value - value) < 0.001):
                                matches = False
                                break
            
            if matches:
                filtered_indices.append(i)
        
        # Filter all result arrays
        filtered_results = {}
        for key, value in results.items():
            if isinstance(value, list) and len(value) > 0:
                filtered_results[key] = [[value[0][i] for i in filtered_indices]]
            else:
                filtered_results[key] = value
        
        return filtered_results
    
    def semantic_search(self, query: str, n_results: int = 10, 
                       where: Optional[Dict] = None) -> Dict:
        """
        Perform semantic search using embeddings.
        
        Args:
            query: Query text
            n_results: Number of results to return
            where: Optional metadata filter
            
        Returns:
            Dictionary with search results
        """
        # Generate query embedding
        query_embedding = self.embedding_generator.embed_query(query)
        
        # Search vector store
        results = self.vector_store.search(
            query_embedding=query_embedding.tolist(),
            n_results=n_results,
            where=where
        )
        
        return results
    
    def hybrid_search(self, query: str, n_results: int = 10) -> Dict:
        """
        Perform hybrid search combining structured and semantic search.
        
        Args:
            query: Query text
            n_results: Number of results to return
            
        Returns:
            Dictionary with search results
        """
        # Try to parse as structured query
        parsed = self.parse_structured_query(query)
        
        if parsed:
            # Execute structured query
            structured_results = self.execute_structured_query(parsed, n_results=n_results)
            
            # Also perform semantic search
            semantic_results = self.semantic_search(query, n_results=n_results)
            
            # Combine results (prioritize structured if available)
            # For now, return structured results if they exist
            if structured_results.get('ids') and structured_results['ids'][0]:
                return structured_results
            else:
                return semantic_results
        else:
            # Pure semantic search
            return self.semantic_search(query, n_results=n_results)
    
    def format_results(self, results: Dict, query: str = "") -> List[Dict]:
        """
        Format search results for display.
        
        Args:
            results: Search results dictionary from vector store
            query: Original query string
            
        Returns:
            List of formatted result dictionaries
        """
        formatted = []
        
        if not results or not results.get('ids'):
            return formatted
        
        # Handle both flat structure (from get()) and nested structure (from query())
        ids = results['ids']
        if isinstance(ids, list) and len(ids) > 0:
            # Check if nested structure
            if isinstance(ids[0], list):
                ids = ids[0]
            # If flat structure, use as-is
        else:
            return formatted
        
        if len(ids) == 0:
            return formatted
        
        # Extract documents, metadatas, distances (handle both structures)
        documents = results.get('documents', [])
        if documents:
            if isinstance(documents, list) and len(documents) > 0:
                if isinstance(documents[0], list):
                    documents = documents[0]
        else:
            documents = []
        
        metadatas = results.get('metadatas', [])
        if metadatas:
            if isinstance(metadatas, list) and len(metadatas) > 0:
                if isinstance(metadatas[0], list):
                    metadatas = metadatas[0]
        else:
            metadatas = []
        
        distances = results.get('distances', [])
        if distances:
            if isinstance(distances, list) and len(distances) > 0:
                if isinstance(distances[0], list):
                    distances = distances[0]
        else:
            distances = []
        
        # Format results
        for i, result_id in enumerate(ids):
            result_dict = {
                'id': result_id,
                'document': documents[i] if i < len(documents) else '',
                'metadata': metadatas[i] if i < len(metadatas) else {},
                'distance': distances[i] if i < len(distances) else None,
                'similarity': 1 - distances[i] if i < len(distances) and distances[i] is not None else None
            }
            formatted.append(result_dict)
        
        return formatted
    
    def query(self, query_text: str, n_results: int = 20, 
              conversation_context: Optional[List[Dict]] = None) -> Dict:
        """
        Main query method with full RAG capabilities including answer generation, tools, and conversation.
        
        Args:
            query_text: Query text
            n_results: Number of results to retrieve for context
            conversation_context: Optional conversation history
            
        Returns:
            Dictionary with answer, sources, tools_used, and metadata
        """
        # Step 1: Resolve references in follow-up questions
        resolved_refs = self.conversation_manager.resolve_reference(query_text)
        if resolved_refs:
            # Update query with resolved references
            if 'formation' in resolved_refs and 'formation' not in query_text.lower():
                query_text = f"{query_text} (formation: {resolved_refs['formation']})"
            if 'well' in resolved_refs and resolved_refs['well'] not in query_text:
                query_text = f"{query_text} (well: {resolved_refs['well']})"
        
        # Step 2: Analyze the query
        analysis = self.query_analyzer.analyze(query_text)
        
        # Update analysis with resolved references
        if resolved_refs:
            if 'formation' in resolved_refs and not analysis.get('formation'):
                analysis['formation'] = resolved_refs['formation']
            if 'well' in resolved_refs and not analysis.get('well_name'):
                analysis['well_name'] = resolved_refs['well']
        
        # Step 3: Determine if this is an analytical query (requires tools) or document-style (RAG only)
        is_analytical = analysis.get('is_analytical', False)
        is_document_style = analysis.get('is_document_style', False)
        
        # For analytical queries, we may still need RAG context, but tools will do the computation
        # For document-style queries, RAG context is sufficient
        
        # Step 4: Retrieve relevant context (always retrieve for context, even for analytical queries)
        try:
            results = self._retrieve_context(query_text, analysis, n_results)
            formatted_results = self.format_results(results, query_text)
        except Exception as e:
            # Log the error for debugging
            import traceback
            error_details = traceback.format_exc()
            print(f"Error in _retrieve_context or format_results: {str(e)}")
            print(f"Traceback: {error_details}")
            # Return empty results instead of crashing
            formatted_results = []
            results = {'ids': [], 'documents': [], 'metadatas': [], 'distances': []}
        
        # Step 5: Perform aggregation if needed (for analytical queries)
        aggregated_data = None
        if analysis['query_type'] == QueryType.AGGREGATION:
            aggregated_data = self._perform_aggregation(formatted_results, analysis)
        elif analysis['query_type'] == QueryType.COMPARISON:
            aggregated_data = self._perform_comparison(formatted_results, analysis)
        
        # Step 6: Get conversation history
        if conversation_context is None:
            conversation_history = self.conversation_manager.get_conversation_history()
        else:
            conversation_history = conversation_context
        
        # Step 7: Generate answer with tools and conversation context
        # The answer generator will decide whether to use tools based on the query type
        answer_result = self.answer_generator.generate_answer(
            query_text, formatted_results, analysis, aggregated_data, conversation_history
        )
        
        # Step 8: Add user message to conversation
        self.conversation_manager.add_message(
            'user', query_text, {'analysis': analysis}
        )
        
        # Step 9: Add assistant response to conversation
        self.conversation_manager.add_message(
            'assistant', answer_result['answer'],
            {
                'sources': answer_result['sources'],
                'tools_used': answer_result.get('tools_used', [])
            }
        )
        
        # Step 10: Return comprehensive result
        result = {
            'query': query_text,
            'answer': answer_result['answer'],
            'sources': answer_result['sources'],
            'context': formatted_results[:5],  # Top 5 for display
            'context_count': answer_result['context_count'],
            'aggregated_data': aggregated_data,
            'tools_used': answer_result.get('tools_used', []),
            'tool_results': answer_result.get('tool_results', {}),
            'analysis': {
                'query_type': analysis['query_type'].value if hasattr(analysis['query_type'], 'value') else str(analysis['query_type']),
                'curve': analysis.get('curve'),
                'formation': analysis.get('formation'),
                'well_name': analysis.get('well_name'),
                'is_analytical': analysis.get('is_analytical', False),
                'is_document_style': analysis.get('is_document_style', False)
            }
        }
        
        return result
    
    def _log_retrieval(self, query_text: str, analysis: Dict, results: Dict, formatted_results: List[Dict]) -> None:
        """
        Log retrieval details for debugging.
        
        Args:
            query_text: Original query
            analysis: Query analysis results
            results: Raw retrieval results
            formatted_results: Formatted results
        """
        import logging
        logger = logging.getLogger(__name__)
        
        # Only log if debug mode is enabled (can be controlled via environment variable)
        debug_mode = os.getenv('RAG_DEBUG', 'false').lower() == 'true'
        
        if not debug_mode:
            return
        
        logger.info(f"\n{'='*80}")
        logger.info(f"RETRIEVAL DEBUG: {query_text}")
        logger.info(f"{'='*80}")
        logger.info(f"Query Type: {analysis.get('query_type')}")
        logger.info(f"Number of results retrieved: {len(formatted_results)}")
        
        # Log retrieval method
        if analysis.get('query_type') == QueryType.LIST and 'all' in query_text.lower():
            logger.info("Retrieval Method: Metadata filter (LIST query)")
        else:
            logger.info("Retrieval Method: Semantic search")
        
        # Log top-k chunk IDs and previews
        if formatted_results:
            logger.info(f"\nTop {min(10, len(formatted_results))} retrieved chunks:")
            for i, result in enumerate(formatted_results[:10], 1):
                chunk_id = result.get('id', 'unknown')
                metadata = result.get('metadata', {})
                doc_preview = result.get('document', '')[:100] if result.get('document') else ''
                well_name = metadata.get('well_name', 'Unknown')
                doc_type = metadata.get('doc_type', metadata.get('type', 'unknown'))
                
                logger.info(f"  {i}. ID: {chunk_id}")
                logger.info(f"     Well: {well_name}")
                logger.info(f"     Type: {doc_type}")
                logger.info(f"     Preview: {doc_preview}...")
                if metadata:
                    logger.info(f"     Metadata keys: {list(metadata.keys())[:5]}")
        else:
            logger.info("No results retrieved")
        
        logger.info(f"{'='*80}\n")
    
    def _retrieve_context(self, query_text: str, analysis: Dict, n_results: int) -> Dict:
        """Retrieve relevant context based on query analysis."""
        query_lower = query_text.lower()
        
        # Special handling for LIST queries asking for "all" items
        if analysis.get('query_type') == QueryType.LIST:
            # Check if query asks for "all" formations/surfaces
            if 'all' in query_lower and ('formation' in query_lower or 'surface' in query_lower):
                # Use metadata filtering to get ALL formation records
                # This bypasses semantic search and gets comprehensive results
                # Use a very high limit to ensure we get all formations from all wells
                where_clause = {'type': 'formation'}
                # Try alternative metadata key if 'type' doesn't work
                results = self.vector_store.filter_by_metadata(where_clause, n_results=10000)
                
                # Normalize results structure first
                if results.get('ids'):
                    ids = results['ids']
                    if isinstance(ids, list) and len(ids) > 0 and not isinstance(ids[0], list):
                        # Flat structure - normalize to nested
                        results = {
                            'ids': [ids],
                            'documents': [results.get('documents', [])],
                            'metadatas': [results.get('metadatas', [])],
                            'distances': [[0.0] * len(ids)]
                        }
                
                # Check if we got results
                ids_check = results.get('ids', [])
                has_results = False
                if ids_check:
                    if isinstance(ids_check, list):
                        if len(ids_check) > 0:
                            if isinstance(ids_check[0], list):
                                has_results = len(ids_check[0]) > 0
                            else:
                                has_results = len(ids_check) > 0
                
                # If no results with 'type', try 'doc_type'
                if not has_results:
                    where_clause = {'doc_type': 'formation'}
                    results = self.vector_store.filter_by_metadata(where_clause, n_results=10000)
                    
                    # Normalize this result too
                    if results.get('ids'):
                        ids = results['ids']
                        if isinstance(ids, list) and len(ids) > 0 and not isinstance(ids[0], list):
                            results = {
                                'ids': [ids],
                                'documents': [results.get('documents', [])],
                                'metadatas': [results.get('metadatas', [])],
                                'distances': [[0.0] * len(ids)]
                            }
                    
                    # Re-check if we have results now
                    ids_check = results.get('ids', [])
                    if ids_check:
                        if isinstance(ids_check, list) and len(ids_check) > 0:
                            if isinstance(ids_check[0], list):
                                has_results = len(ids_check[0]) > 0
                            else:
                                has_results = len(ids_check) > 0
                
                # Normalize results structure: collection.get() returns flat structure, 
                # but format_results expects nested structure like collection.query()
                if results.get('ids'):
                    # Check if structure is flat (from get()) or nested (from query())
                    ids = results['ids']
                    if isinstance(ids, list) and len(ids) > 0 and not isinstance(ids[0], list):
                        # Flat structure from get() - convert to nested structure
                        normalized_results = {
                            'ids': [ids],
                            'documents': [results.get('documents', [])],
                            'metadatas': [results.get('metadatas', [])],
                            'distances': [[0.0] * len(ids)]  # No distances for metadata-filtered results
                        }
                        results = normalized_results
                        has_results = len(ids) > 0
                
                # If still no results, try getting all items and filter in Python
                if not has_results:
                    # Get all items and filter for formation type
                    # Use very high limit to ensure we get all records
                    all_results = self.vector_store.filter_by_metadata({}, n_results=10000)
                    
                    # Normalize all_results structure
                    if all_results.get('ids'):
                        all_ids = all_results['ids']
                        if isinstance(all_ids, list) and len(all_ids) > 0 and not isinstance(all_ids[0], list):
                            # Normalize to nested structure
                            all_results = {
                                'ids': [all_ids],
                                'documents': [all_results.get('documents', [])],
                                'metadatas': [all_results.get('metadatas', [])]
                            }
                    
                    # Extract metadatas list (handle both flat and nested)
                    metadatas_list = []
                    if all_results.get('metadatas'):
                        if isinstance(all_results['metadatas'], list) and len(all_results['metadatas']) > 0:
                            if isinstance(all_results['metadatas'][0], list):
                                metadatas_list = all_results['metadatas'][0]
                            else:
                                metadatas_list = all_results['metadatas']
                    
                    if metadatas_list:
                        filtered_indices = []
                        for i, metadata in enumerate(metadatas_list):
                            if isinstance(metadata, dict):
                                # Check if this is a formation record
                                if (metadata.get('type') == 'formation' or 
                                    metadata.get('doc_type') == 'formation' or
                                    'formation_name' in metadata):
                                    filtered_indices.append(i)
                        
                        if filtered_indices:
                            # Filter results to only formation records
                            # all_results is already normalized to nested structure
                            filtered = {}
                            for key, value in all_results.items():
                                if key in ['metadatas', 'documents', 'ids']:
                                    if isinstance(value, list) and len(value) > 0 and isinstance(value[0], list):
                                        filtered[key] = [[value[0][i] for i in filtered_indices if i < len(value[0])]]
                                    else:
                                        filtered[key] = value
                                else:
                                    filtered[key] = value
                            # Add distances (all 0.0 for metadata-filtered results)
                            filtered['distances'] = [[0.0] * len(filtered_indices)]
                            results = filtered
                
                # Ensure results are in correct format (nested structure)
                if results.get('ids'):
                    ids = results['ids']
                    if isinstance(ids, list) and len(ids) > 0:
                        if not isinstance(ids[0], list):
                            # Still flat, normalize
                            results = {
                                'ids': [ids],
                                'documents': [results.get('documents', [])],
                                'metadatas': [results.get('metadatas', [])],
                                'distances': [results.get('distances', [[0.0] * len(ids)])[0] if results.get('distances') else [0.0] * len(ids)]
                            }
                else:
                    # No results found - return empty structure in correct format
                    results = {
                        'ids': [[]],
                        'documents': [[]],
                        'metadatas': [[]],
                        'distances': [[]]
                    }
                
                return results
        
        # Build metadata filter if we have specific requirements
        where_clause = None
        
        if analysis.get('formation'):
            # Try to filter by formation in metadata
            # Note: ChromaDB where clause would need formation_name
            pass  # Will filter after retrieval
        
        # Note: ChromaDB doesn't support $contains, so we'll filter after retrieval
        # if analysis.get('well_name'):
        #     where_clause = {'well_name': {'$contains': analysis['well_name']}}
        
        # Perform semantic search
        results = self.semantic_search(query_text, n_results=n_results, where=where_clause)
        
        # Post-filter by well name first (for specific queries)
        if analysis.get('well_name') and self._safe_get_list_length(results, 'metadatas') > 0:
            filtered_results = self._filter_by_well_name(results, analysis['well_name'])
            if filtered_results.get('ids') and self._safe_get_list_length(filtered_results, 'ids') > 0:
                return filtered_results
        
        # Post-filter by formation if needed (only if formation is specified and no well name)
        if analysis.get('formation') and not analysis.get('well_name') and self._safe_get_list_length(results, 'metadatas') > 0:
            filtered_results = self._filter_by_formation(results, analysis['formation'])
            if filtered_results.get('ids') and self._safe_get_list_length(filtered_results, 'ids') > 0:
                return filtered_results
        
        return results
    
    def _filter_by_formation(self, results: Dict, formation: str) -> Dict:
        """Filter results by formation name."""
        if not results.get('metadatas') or not results['metadatas'] or len(results['metadatas']) == 0:
            return results
        
        metadatas_list = results['metadatas'][0] if isinstance(results['metadatas'][0], list) else results['metadatas']
        if not metadatas_list or len(metadatas_list) == 0:
            return results
        
        filtered_indices = []
        documents_list = []
        if results.get('documents') and len(results['documents']) > 0:
            documents_list = results['documents'][0] if isinstance(results['documents'][0], list) else results['documents']
        
        for i, metadata in enumerate(metadatas_list):
            if i >= len(metadatas_list):
                break
            formation_name = metadata.get('formation_name', '').lower() if isinstance(metadata, dict) else ''
            doc = documents_list[i].lower() if i < len(documents_list) and isinstance(documents_list[i], str) else ''
            
            if formation.lower() in formation_name or formation.lower() in doc:
                filtered_indices.append(i)
        
        if not filtered_indices:
            return results
        
        # Filter all arrays - handle nested structure from ChromaDB
        filtered = {}
        for key, value in results.items():
            if key == 'metadatas' and isinstance(value, list) and len(value) > 0:
                if isinstance(value[0], list):
                    filtered[key] = [[value[0][i] for i in filtered_indices if i < len(value[0])]]
                else:
                    filtered[key] = [[value[i] for i in filtered_indices if i < len(value)]]
            elif key == 'documents' and isinstance(value, list) and len(value) > 0:
                if isinstance(value[0], list):
                    filtered[key] = [[value[0][i] for i in filtered_indices if i < len(value[0])]]
                else:
                    filtered[key] = [[value[i] for i in filtered_indices if i < len(value)]]
            elif key == 'ids' and isinstance(value, list) and len(value) > 0:
                if isinstance(value[0], list):
                    filtered[key] = [[value[0][i] for i in filtered_indices if i < len(value[0])]]
                else:
                    filtered[key] = [[value[i] for i in filtered_indices if i < len(value)]]
            elif key == 'distances' and isinstance(value, list) and len(value) > 0:
                if isinstance(value[0], list):
                    filtered[key] = [[value[0][i] for i in filtered_indices if i < len(value[0])]]
                else:
                    filtered[key] = [[value[i] for i in filtered_indices if i < len(value)]]
            else:
                filtered[key] = value
        
        return filtered
    
    def _filter_by_well_name(self, results: Dict, well_name: str) -> Dict:
        """Filter results by well name."""
        if not results.get('metadatas') or not results['metadatas'] or len(results['metadatas']) == 0:
            return results
        
        metadatas_list = results['metadatas'][0] if isinstance(results['metadatas'][0], list) else results['metadatas']
        if not metadatas_list or len(metadatas_list) == 0:
            return results
        
        filtered_indices = []
        for i, metadata in enumerate(metadatas_list):
            if i >= len(metadatas_list):
                break
            result_well_name = metadata.get('well_name', '').lower() if isinstance(metadata, dict) else ''
            if well_name.lower() in result_well_name or result_well_name in well_name.lower():
                filtered_indices.append(i)
        
        if not filtered_indices:
            return results
        
        # Filter all arrays - handle nested structure from ChromaDB
        filtered = {}
        for key, value in results.items():
            if key == 'metadatas' and isinstance(value, list) and len(value) > 0:
                if isinstance(value[0], list):
                    filtered[key] = [[value[0][i] for i in filtered_indices if i < len(value[0])]]
                else:
                    filtered[key] = [[value[i] for i in filtered_indices if i < len(value)]]
            elif key == 'documents' and isinstance(value, list) and len(value) > 0:
                if isinstance(value[0], list):
                    filtered[key] = [[value[0][i] for i in filtered_indices if i < len(value[0])]]
                else:
                    filtered[key] = [[value[i] for i in filtered_indices if i < len(value)]]
            elif key == 'ids' and isinstance(value, list) and len(value) > 0:
                if isinstance(value[0], list):
                    filtered[key] = [[value[0][i] for i in filtered_indices if i < len(value[0])]]
                else:
                    filtered[key] = [[value[i] for i in filtered_indices if i < len(value)]]
            elif key == 'distances' and isinstance(value, list) and len(value) > 0:
                if isinstance(value[0], list):
                    filtered[key] = [[value[0][i] for i in filtered_indices if i < len(value[0])]]
                else:
                    filtered[key] = [[value[i] for i in filtered_indices if i < len(value)]]
            else:
                filtered[key] = value
        
        return filtered
    
    def _perform_aggregation(self, results: List[Dict], analysis: Dict) -> Optional[Dict]:
        """Perform aggregation on results."""
        curve = analysis.get('curve')
        if not curve:
            return None
        
        formation = analysis.get('formation')
        aggregation_type = analysis.get('aggregation_type', 'mean')
        
        if formation:
            aggregated = self.aggregator.aggregate_by_formation(
                results, formation, curve, aggregation_type
            )
        else:
            aggregated = self.aggregator.aggregate_all(results, curve, aggregation_type)
        
        return aggregated
    
    def _perform_comparison(self, results: List[Dict], analysis: Dict) -> Optional[Dict]:
        """Perform comparison on results."""
        curve = analysis.get('curve')
        if not curve:
            return None
        
        comparison_type = analysis.get('comparison_type', 'max')
        comparison = self.aggregator.compare_wells(results, curve, comparison_type)
        
        return comparison


if __name__ == "__main__":
    # Test the RAG query processor
    from vector_store import VectorStore
    from embeddings import EmbeddingGenerator
    
    # Initialize components
    store = VectorStore()
    generator = EmbeddingGenerator()
    
    # Create processor
    processor = RAGQueryProcessor(store, generator)
    
    # Test queries
    test_queries = [
        "Which wells have high porosity?",
        "Find wells with permeability > 100 mD",
        "Show me wells in the Hugin formation"
    ]
    
    print("RAG Query Processor initialized")
    print("Test queries prepared (will work once data is indexed)")

