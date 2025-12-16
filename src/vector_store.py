"""
Vector store module for storing and retrieving embeddings using ChromaDB.
"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional, Tuple
import os
from pathlib import Path


class VectorStore:
    """Manages vector storage and retrieval using ChromaDB."""
    
    def __init__(self, persist_directory: str = "vector_db", collection_name: str = "volve_wells"):
        """
        Initialize the vector store.
        
        Args:
            persist_directory: Directory to persist the database
            collection_name: Name of the ChromaDB collection
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Volve well data embeddings"}
        )
        
        self.collection_name = collection_name
        print(f"Initialized vector store: {collection_name}")
        try:
            count = self.collection.count()
            print(f"Current collection size: {count}")
        except Exception as e:
            print(f"Warning: Could not get collection count: {e}")
            print("Collection may need to be rebuilt.")
    
    def add_documents(self, documents: List[Dict]) -> None:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of document dictionaries with 'text', 'metadata', and 'embedding' keys
        """
        if not documents:
            return
        
        ids = []
        embeddings = []
        texts = []
        metadatas = []
        
        for i, doc in enumerate(documents):
            if 'embedding' not in doc or 'text' not in doc:
                continue
            
            # Create unique ID
            doc_id = doc.get('id', f"doc_{i}_{doc['metadata'].get('well_name', 'unknown')}_{doc['metadata'].get('doc_type', 'unknown')}")
            
            ids.append(doc_id)
            embeddings.append(doc['embedding'])
            texts.append(doc['text'])
            
            # Create metadata (filter out None values as ChromaDB doesn't accept them)
            metadata = {}
            for key, value in doc['metadata'].items():
                if value is not None:
                    # ChromaDB metadata must be str, int, or float
                    if isinstance(value, (str, int, float, bool)):
                        metadata[key] = value
                    elif isinstance(value, list):
                        # Convert list to comma-separated string
                        metadata[key] = ','.join(str(v) for v in value)
                    else:
                        metadata[key] = str(value)
            
            metadatas.append(metadata)
        
        if ids:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
            )
            print(f"Added {len(ids)} documents to vector store")
    
    def add_well_summaries(self, well_summaries: List[Dict]) -> None:
        """
        Add well summaries to the vector store (legacy method for backward compatibility).
        
        Args:
            well_summaries: List of well summary dictionaries with embeddings
        """
        if not well_summaries:
            return
        
        ids = []
        embeddings = []
        documents = []
        metadatas = []
        
        for summary in well_summaries:
            if 'embedding' not in summary:
                continue
            
            # Create unique ID
            well_name = summary.get('well_name', 'unknown')
            file_type = summary.get('file_type', 'unknown')
            embedding_id = summary.get('embedding_id', f"well_{well_name}_{file_type}")
            
            ids.append(embedding_id)
            embeddings.append(summary['embedding'])
            documents.append(summary.get('summary_text', ''))
            
            # Create metadata (filter out None values as ChromaDB doesn't accept them)
            metadata = {
                'type': 'well_summary',
                'well_name': well_name,
                'file_type': file_type,
                'doc_type': 'well_summary',  # Add doc_type for consistency
            }
            
            # Add optional fields only if they have values
            if summary.get('field'):
                metadata['field'] = summary.get('field')
            if summary.get('start_depth') is not None:
                metadata['depth_top'] = summary.get('start_depth')
            if summary.get('stop_depth') is not None:
                metadata['depth_base'] = summary.get('stop_depth')
            
            # Add curve list if available
            if summary.get('available_curves'):
                curve_list = summary.get('available_curves', [])
                if isinstance(curve_list, list):
                    metadata['curve_list'] = ','.join(str(c) for c in curve_list)
                else:
                    metadata['curve_list'] = str(curve_list)
            
            # Add curve statistics to metadata (only non-None values)
            if summary.get('curve_statistics'):
                for curve, stats in summary['curve_statistics'].items():
                    if stats:
                        if stats.get('mean') is not None:
                            metadata[f'{curve}_mean'] = stats.get('mean')
                        if stats.get('min') is not None:
                            metadata[f'{curve}_min'] = stats.get('min')
                        if stats.get('max') is not None:
                            metadata[f'{curve}_max'] = stats.get('max')
            
            metadatas.append(metadata)
        
        if ids:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            print(f"Added {len(ids)} well summaries to vector store")
    
    def add_interval_chunks(self, intervals: List[Dict]) -> None:
        """
        Add depth interval chunks to the vector store.
        
        Args:
            intervals: List of interval dictionaries with embeddings
        """
        if not intervals:
            return
        
        ids = []
        embeddings = []
        documents = []
        metadatas = []
        
        for interval in intervals:
            if 'embedding' not in interval:
                continue
            
            embedding_id = interval.get('embedding_id', 
                f"interval_{interval.get('well_name', 'unknown')}_{interval.get('start_depth', 0)}")
            
            ids.append(embedding_id)
            embeddings.append(interval['embedding'])
            documents.append(interval.get('description', ''))
            
            metadata = {
                'type': 'interval',
                'doc_type': 'interval_summary',  # Add doc_type for consistency
                'well_name': interval.get('well_name', ''),
            }
            
            # Add optional fields only if they have values
            if interval.get('start_depth') is not None:
                metadata['depth_top'] = interval.get('start_depth')
            if interval.get('end_depth') is not None:
                metadata['depth_base'] = interval.get('end_depth')
            if interval.get('formation'):
                metadata['formation'] = interval.get('formation')
            if interval.get('field'):
                metadata['field'] = interval.get('field')
            if interval.get('curve_list'):
                curve_list = interval.get('curve_list', [])
                if isinstance(curve_list, list):
                    metadata['curve_list'] = ','.join(str(c) for c in curve_list)
                else:
                    metadata['curve_list'] = str(curve_list)
            
            # Add statistics to metadata (only non-None values)
            if interval.get('statistics'):
                for curve, stats in interval['statistics'].items():
                    if stats and stats.get('mean') is not None:
                        metadata[f'{curve}_mean'] = stats.get('mean')
            
            metadatas.append(metadata)
        
        if ids:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            print(f"Added {len(ids)} interval chunks to vector store")
    
    def add_formation_data(self, formation_embeddings: List[Dict]) -> None:
        """
        Add formation top data to the vector store.
        
        Args:
            formation_embeddings: List of formation dictionaries with embeddings
        """
        if not formation_embeddings:
            return
        
        ids = []
        embeddings = []
        documents = []
        metadatas = []
        
        for formation in formation_embeddings:
            if 'embedding' not in formation:
                continue
            
            embedding_id = formation.get('embedding_id', 
                f"formation_{formation.get('well_name', 'unknown')}_{formation.get('formation_name', 'unknown')}")
            
            ids.append(embedding_id)
            embeddings.append(formation['embedding'])
            documents.append(formation.get('description', ''))
            
            metadata = {
                'type': 'formation',
                'doc_type': 'formation',  # Add doc_type for consistency
                'well_name': formation.get('well_name', ''),
                'formation_name': formation.get('formation_name', ''),
            }
            
            # Add optional fields only if they have values
            if formation.get('md') is not None:
                metadata['md'] = formation.get('md')
                metadata['depth_top'] = formation.get('md')  # Also add as depth_top for consistency
            if formation.get('tvd') is not None:
                metadata['tvd'] = formation.get('tvd')
            if formation.get('tvdss') is not None:
                metadata['tvdss'] = formation.get('tvdss')
            if formation.get('field'):
                metadata['field'] = formation.get('field')
            
            metadatas.append(metadata)
        
        if ids:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            print(f"Added {len(ids)} formation records to vector store")
    
    def search(self, query_embedding: List[float], n_results: int = 10, 
               where: Optional[Dict] = None) -> Dict:
        """
        Search the vector store for similar items.
        
        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            where: Optional metadata filter dictionary
            
        Returns:
            Dictionary with search results
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where
        )
        
        return results
    
    def search_by_text(self, query_text: str, n_results: int = 10,
                      where: Optional[Dict] = None) -> Dict:
        """
        Search using text query (will be embedded automatically by ChromaDB).
        
        Args:
            query_text: Query text string
            n_results: Number of results to return
            where: Optional metadata filter dictionary
            
        Returns:
            Dictionary with search results
        """
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where
        )
        
        return results
    
    def get_by_ids(self, ids: List[str]) -> Dict:
        """
        Retrieve items by their IDs.
        
        Args:
            ids: List of item IDs
            
        Returns:
            Dictionary with retrieved items
        """
        results = self.collection.get(ids=ids)
        return results
    
    def filter_by_metadata(self, where: Dict, n_results: int = 100) -> Dict:
        """
        Filter items by metadata without similarity search.
        
        Args:
            where: Metadata filter dictionary
            n_results: Maximum number of results
            
        Returns:
            Dictionary with filtered items
        """
        results = self.collection.get(
            where=where,
            limit=n_results
        )
        return results
    
    def get_collection_info(self) -> Dict:
        """
        Get information about the collection.
        
        Returns:
            Dictionary with collection information
        """
        try:
            count = self.collection.count()
            
            # Get sample items to understand structure
            try:
                sample = self.collection.get(limit=1)
                sample_metadata_keys = list(sample.get('metadatas', [{}])[0].keys()) if sample.get('metadatas') else []
            except Exception:
                sample_metadata_keys = []
            
            return {
                'collection_name': self.collection_name,
                'count': count,
                'sample_metadata_keys': sample_metadata_keys
            }
        except Exception as e:
            # Collection doesn't exist or is corrupted
            return {
                'collection_name': self.collection_name,
                'count': 0,
                'sample_metadata_keys': [],
                'error': str(e)
            }
    
    def clear_collection(self) -> None:
        """Clear all items from the collection."""
        # Delete and recreate collection
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            # Collection might not exist, that's okay
            pass
        
        # Recreate collection with fresh reference
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Volve well data embeddings"}
        )
        print(f"Cleared collection: {self.collection_name}")
    
    def refresh_collection(self) -> None:
        """Refresh the collection reference (useful after rebuild)."""
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
        except Exception:
            # If collection doesn't exist, create it
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Volve well data embeddings"}
            )


if __name__ == "__main__":
    # Test the vector store
    store = VectorStore()
    
    # Test with sample data
    sample_summary = {
        'well_name': 'Test Well',
        'summary_text': 'Test well with porosity 0.15 and permeability 50 mD',
        'embedding': [0.1] * 384,  # Dummy embedding
        'embedding_id': 'test_well_1',
        'field': 'VOLVE',
        'file_type': 'OUTPUT'
    }
    
    # Note: This will fail without actual embeddings, but tests the structure
    print("Vector store initialized successfully")
    print(f"Collection info: {store.get_collection_info()}")

