"""
Standalone script to generate document corpus from LAS files.
Can be run independently to rebuild the corpus.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.corpus_generator import generate_well_documents, generate_header_document
from src.formation_tops_parser import FormationTopsParser
from src.las_io import load_las
from src.embeddings import EmbeddingGenerator
from src.vector_store import VectorStore
import json
from tqdm import tqdm


def generate_corpus(base_path: str = "spwla_volve-main", 
                   output_dir: str = "data/corpus",
                   save_json: bool = True):
    """
    Generate document corpus from all LAS files.
    
    Args:
        base_path: Path to Volve dataset directory
        output_dir: Directory to save corpus documents
        save_json: Whether to save documents as JSON files
    """
    base_path = Path(base_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating corpus from {base_path}...")
    
    # Load formation tops
    formation_tops_path = base_path / "Well_picks_Volve_v1.dat"
    formation_tops = {}
    if formation_tops_path.exists():
        parser = FormationTopsParser(str(formation_tops_path))
        formation_tops = parser.parse_file()
        print(f"Loaded formation tops for {len(formation_tops)} wells")
    else:
        print(f"Warning: Formation tops file not found at {formation_tops_path}")
    
    # Find all well folders
    well_folders = []
    for item in base_path.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            las_files = list(item.glob("*.las")) + list(item.glob("*.LAS"))
            if las_files:
                well_folders.append(item)
    
    print(f"Found {len(well_folders)} well folders")
    
    # Generate documents for each well
    all_documents = []
    
    for well_folder in tqdm(well_folders, desc="Processing wells"):
        try:
            # Get formation tops for this well
            well_formations = None
            well_name = well_folder.name
            for key, formations in formation_tops.items():
                if well_name.upper() in key.upper() or key.upper() in well_name.upper():
                    well_formations = formations
                    break
            
            # Generate documents
            well_docs = generate_well_documents(well_folder, well_formations)
            all_documents.extend(well_docs)
            
            # Save individual well corpus if requested
            if save_json and well_docs:
                well_output_file = output_dir / f"{well_name}_corpus.json"
                with open(well_output_file, 'w', encoding='utf-8') as f:
                    json.dump(well_docs, f, indent=2, ensure_ascii=False)
        
        except Exception as e:
            print(f"Error processing {well_folder.name}: {e}")
            continue
    
    # Save complete corpus
    if save_json:
        corpus_file = output_dir / "complete_corpus.json"
        with open(corpus_file, 'w', encoding='utf-8') as f:
            json.dump(all_documents, f, indent=2, ensure_ascii=False)
        print(f"\nSaved {len(all_documents)} documents to {corpus_file}")
    
    # Print summary
    doc_types = {}
    for doc in all_documents:
        doc_type = doc['metadata'].get('doc_type', 'unknown')
        doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
    
    print("\nCorpus Summary:")
    print(f"Total documents: {len(all_documents)}")
    for doc_type, count in doc_types.items():
        print(f"  {doc_type}: {count}")
    
    return all_documents


def index_corpus(documents: list, vector_db_path: str = "vector_db"):
    """
    Index the corpus documents in the vector store.
    
    Args:
        documents: List of document dictionaries
        vector_db_path: Path to vector database directory
    """
    print(f"\nIndexing {len(documents)} documents in vector store...")
    
    # Initialize components
    embedding_generator = EmbeddingGenerator()
    vector_store = VectorStore(persist_directory=vector_db_path)
    
    # Clear existing collection
    vector_store.clear_collection()
    
    # Generate embeddings and add to vector store
    batch_size = 50
    for i in tqdm(range(0, len(documents), batch_size), desc="Indexing documents"):
        batch = documents[i:i + batch_size]
        
        # Generate embeddings
        texts = [doc['text'] for doc in batch]
        embeddings = embedding_generator.embed_texts(texts)
        
        # Add embeddings to documents
        for j, doc in enumerate(batch):
            doc['embedding'] = embeddings[j].tolist()
        
        # Add to vector store
        vector_store.add_documents(batch)
    
    # Refresh collection
    vector_store.refresh_collection()
    
    # Get collection info
    info = vector_store.get_collection_info()
    print(f"\nIndexing complete!")
    print(f"Collection: {info['collection_name']}")
    print(f"Total documents: {info['count']}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate document corpus from LAS files")
    parser.add_argument("--base-path", default="spwla_volve-main", 
                       help="Path to Volve dataset directory")
    parser.add_argument("--output-dir", default="data/corpus",
                       help="Directory to save corpus documents")
    parser.add_argument("--vector-db", default="vector_db",
                       help="Path to vector database directory")
    parser.add_argument("--no-json", action="store_true",
                       help="Don't save corpus as JSON files")
    parser.add_argument("--index", action="store_true",
                       help="Index corpus in vector store after generation")
    
    args = parser.parse_args()
    
    # Generate corpus
    documents = generate_corpus(
        base_path=args.base_path,
        output_dir=args.output_dir,
        save_json=not args.no_json
    )
    
    # Index if requested
    if args.index:
        index_corpus(documents, vector_db_path=args.vector_db)

