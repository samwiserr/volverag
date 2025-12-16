"""
Main application for the Volve Wells RAG System.
Integrates all components and provides interface for querying well data.
"""

import argparse
import sys
import os
from pathlib import Path
from typing import List, Dict, Optional
import json

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

# Add project root to path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data_ingestion import LASFileReader
from src.formation_tops_parser import FormationTopsParser
from src.data_processor import WellDataProcessor
from src.embeddings import EmbeddingGenerator
from src.vector_store import VectorStore
from src.corpus_generator import generate_well_documents, generate_master_formations_document
from src.rag_query import RAGQueryProcessor


class VolveRAGSystem:
    """Main RAG system for Volve well data."""
    
    def __init__(self, base_path: str = "spwla_volve-main", 
                 vector_db_path: str = "vector_db",
                 rebuild_index: bool = False):
        """
        Initialize the RAG system.
        
        Args:
            base_path: Path to Volve dataset directory
            vector_db_path: Path to vector database directory
            rebuild_index: Whether to rebuild the index from scratch
        """
        self.base_path = base_path
        self.vector_db_path = vector_db_path
        
        # Initialize components
        print("Initializing RAG system components...")
        self.reader = LASFileReader(base_path)
        formation_tops_path = Path(base_path) / "Well_picks_Volve_v1.dat"
        self.formation_parser = FormationTopsParser(str(formation_tops_path))
        self.processor = WellDataProcessor()
        self.embedding_generator = EmbeddingGenerator()
        self.vector_store = VectorStore(persist_directory=vector_db_path)
        
        # Load formation tops data for RAG processor
        formation_tops_data = self.formation_parser.parse_file()
        
        # Get LLM provider from environment (default: gemini)
        llm_provider = os.getenv("LLM_PROVIDER", "gemini").lower()
        llm_model = os.getenv("LLM_MODEL", "gemini-pro")
        
        self.query_processor = RAGQueryProcessor(
            self.vector_store, 
            self.embedding_generator,
            formation_tops_data=formation_tops_data,
            llm_provider=llm_provider,
            llm_model=llm_model
        )
        
        # Check if index needs to be built
        collection_info = self.vector_store.get_collection_info()
        if rebuild_index or collection_info['count'] == 0:
            print("Building index from LAS files...")
            self.build_index()
        else:
            print(f"Using existing index with {collection_info['count']} items")
    
    def build_index(self, save_processed_data: bool = True, use_corpus_generator: bool = True) -> None:
        """
        Build the vector index from LAS files.
        
        Args:
            save_processed_data: Whether to save processed data to disk
            use_corpus_generator: If True, use new corpus generator (recommended). 
                                 If False, use legacy data processor.
        """
        print("\n=== Building Vector Index ===")
        
        if use_corpus_generator:
            # Use new corpus generator approach
            self._build_index_with_corpus_generator(save_processed_data)
        else:
            # Use legacy approach
            self._build_index_legacy(save_processed_data)
    
    def _build_index_with_corpus_generator(self, save_processed_data: bool = True) -> None:
        """Build index using new corpus generator."""
        from pathlib import Path
        from tqdm import tqdm
        
        # Step 1: Parse formation tops
        print("\n1. Parsing formation tops...")
        formation_tops_data = self.formation_parser.parse_file()
        print(f"   Found formation tops for {len(formation_tops_data)} wells")
        
        # Step 2: Generate documents from all wells
        print("\n2. Generating documents from LAS files...")
        base_path = Path(self.base_path)
        well_folders = []
        for item in base_path.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                las_files = list(item.glob("*.las")) + list(item.glob("*.LAS"))
                if las_files:
                    well_folders.append(item)
        
        all_documents = []
        for well_folder in tqdm(well_folders, desc="Processing wells"):
            try:
                well_formations = None
                well_name = well_folder.name
                for key, formations in formation_tops_data.items():
                    if well_name.upper() in key.upper() or key.upper() in well_name.upper():
                        well_formations = formations
                        break
                
                well_docs = generate_well_documents(well_folder, well_formations)
                all_documents.extend(well_docs)
            except Exception as e:
                print(f"Error processing {well_folder.name}: {e}")
                continue
        
        print(f"   Generated {len(all_documents)} documents")
        
        # Step 3: Generate embeddings
        print("\n3. Generating embeddings...")
        texts = [doc['text'] for doc in all_documents]
        embeddings = self.embedding_generator.embed_texts(texts)
        
        # Add embeddings to documents
        for i, doc in enumerate(all_documents):
            doc['embedding'] = embeddings[i].tolist()
        
        # Step 4: Clear existing index and add to vector store
        print("\n4. Adding to vector store...")
        self.vector_store.clear_collection()
        
        # Add documents in batches
        batch_size = 50
        for i in tqdm(range(0, len(all_documents), batch_size), desc="Indexing"):
            batch = all_documents[i:i + batch_size]
            self.vector_store.add_documents(batch)
        
        # Step 5: Also add formation data (for backward compatibility)
        formation_embeddings = self.embedding_generator.embed_formation_data(formation_tops_data)
        self.vector_store.add_formation_data(formation_embeddings)
        
        print("\n=== Index Build Complete ===")
        self.vector_store.refresh_collection()
        collection_info = self.vector_store.get_collection_info()
        print(f"Total items in vector store: {collection_info['count']}")
    
    def _build_index_legacy(self, save_processed_data: bool = True) -> None:
        """Build index using legacy data processor (for backward compatibility)."""
        # Step 1: Read LAS files
        print("\n1. Reading LAS files...")
        all_well_data = self.reader.process_all_wells()
        
        if not all_well_data:
            print("Error: No LAS files found. Please check the data path.")
            return
        
        # Step 2: Parse formation tops
        print("\n2. Parsing formation tops...")
        formation_tops_data = self.formation_parser.parse_file()
        print(f"   Found formation tops for {len(formation_tops_data)} wells")
        
        # Step 3: Process well data
        print("\n3. Processing well data and creating summaries...")
        well_summaries = self.processor.process_all_wells(all_well_data, formation_tops_data)
        print(f"   Created {len(well_summaries)} well summaries")
        
        # Step 4: Create depth intervals
        print("\n4. Creating depth intervals...")
        intervals = self.processor.create_interval_chunks(all_well_data, interval_size=50.0)
        print(f"   Created {len(intervals)} depth intervals")
        
        # Step 5: Generate embeddings
        print("\n5. Generating embeddings...")
        well_summaries = self.embedding_generator.embed_well_summaries(well_summaries)
        intervals = self.embedding_generator.embed_interval_chunks(intervals)
        formation_embeddings = self.embedding_generator.embed_formation_data(formation_tops_data)
        print(f"   Generated embeddings for {len(well_summaries)} summaries, "
              f"{len(intervals)} intervals, {len(formation_embeddings)} formations")
        
        # Step 6: Clear existing index and add to vector store
        print("\n6. Adding to vector store...")
        self.vector_store.clear_collection()
        self.vector_store.add_well_summaries(well_summaries)
        self.vector_store.add_interval_chunks(intervals)
        self.vector_store.add_formation_data(formation_embeddings)
        
        # Step 7: Save processed data if requested
        if save_processed_data:
            print("\n7. Saving processed data...")
            self._save_processed_data(well_summaries, intervals, formation_embeddings)
        
        print("\n=== Index Build Complete ===")
        # Refresh collection reference after rebuild to ensure it's current
        self.vector_store.refresh_collection()
        collection_info = self.vector_store.get_collection_info()
        print(f"Total items in vector store: {collection_info['count']}")
    
    def _save_processed_data(self, well_summaries: List[Dict], 
                            intervals: List[Dict],
                            formation_embeddings: List[Dict]) -> None:
        """Save processed data to disk for caching."""
        import pickle
        
        data_dir = Path("data/processed")
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Remove embeddings before saving (they're in vector store)
        summaries_no_emb = [{k: v for k, v in s.items() if k != 'embedding'} 
                            for s in well_summaries]
        intervals_no_emb = [{k: v for k, v in i.items() if k != 'embedding' and k != 'data'} 
                            for i in intervals]
        
        data = {
            'well_summaries': summaries_no_emb,
            'intervals': intervals_no_emb,
            'formation_embeddings': [{k: v for k, v in f.items() if k != 'embedding'} 
                                     for f in formation_embeddings]
        }
        
        with open(data_dir / "processed_data.pkl", "wb") as f:
            pickle.dump(data, f)
        
        print(f"   Saved processed data to {data_dir / 'processed_data.pkl'}")
    
    def query(self, query_text: str, n_results: int = 20, 
              conversation_context: Optional[List[Dict]] = None) -> Dict:
        """
        Query the RAG system and get an answer.
        
        Args:
            query_text: Query string
            n_results: Number of context results to retrieve
            conversation_context: Optional conversation history for context retention
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        return self.query_processor.query(query_text, n_results=n_results,
                                         conversation_context=conversation_context)
    
    def print_results(self, result: Dict, query: str = "") -> None:
        """
        Print query results in a formatted way.
        
        Args:
            result: Result dictionary with answer and sources
            query: Original query string
        """
        if query:
            print(f"\nQuery: {query}")
        else:
            print(f"\nQuery: {result.get('query', 'Unknown')}")
        print("=" * 80)
        
        # Print answer
        print(f"\n📝 Answer:")
        print(result.get('answer', 'No answer generated.'))
        
        # Print aggregated data if available
        if result.get('aggregated_data'):
            agg = result['aggregated_data']
            print(f"\n📊 Aggregated Statistics:")
            if 'value' in agg:
                print(f"  Value: {agg['value']:.4f}")
            if 'count' in agg:
                print(f"  Based on: {agg['count']} well(s)")
            if 'wells' in agg:
                print(f"  Wells: {', '.join(agg['wells'][:5])}")
        
        # Print sources
        if result.get('sources'):
            print(f"\n📚 Sources ({len(result['sources'])}):")
            for i, source in enumerate(result['sources'], 1):
                print(f"  {i}. {source.get('well_name', 'Unknown')}")
                if source.get('formation'):
                    print(f"     Formation: {source['formation']}")
                if source.get('similarity'):
                    print(f"     Relevance: {source['similarity']:.3f}")
        
        print()


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description="Volve Wells RAG System")
    parser.add_argument("--query", "-q", type=str, help="Query to execute")
    parser.add_argument("--rebuild", "-r", action="store_true", 
                       help="Rebuild the vector index")
    parser.add_argument("--results", "-n", type=int, default=20,
                       help="Number of context results to retrieve (default: 20)")
    parser.add_argument("--interactive", "-i", action="store_true",
                       help="Run in interactive mode")
    
    args = parser.parse_args()
    
    # Initialize system
    system = VolveRAGSystem(rebuild_index=args.rebuild)
    
    # Interactive mode
    if args.interactive:
        print("\n=== Volve Wells RAG System - Interactive Mode ===")
        print("Enter queries (type 'exit' to quit):\n")
        
        while True:
            try:
                query = input("Query: ").strip()
                if not query or query.lower() in ['exit', 'quit', 'q']:
                    break
                
                result = system.query(query, n_results=args.results)
                system.print_results(result, query)
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    # Single query mode
    elif args.query:
        result = system.query(args.query, n_results=args.results)
        system.print_results(result, args.query)
    
    # Rebuild only
    elif args.rebuild:
        print("Index rebuilt successfully.")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

