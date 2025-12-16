"""
Embedding generation module for creating vector embeddings of well data.
Uses sentence-transformers for generating embeddings.
"""

from typing import List, Dict, Optional
import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingGenerator:
    """Generates embeddings for well data using sentence transformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: Name of the sentence transformer model to use
                       Options: "all-MiniLM-L6-v2" (fast, good quality)
                               "all-mpnet-base-v2" (slower, better quality)
        """
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.embedding_dimension = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded. Embedding dimension: {self.embedding_dimension}")
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Numpy array with embedding vector
        """
        if not text or not text.strip():
            # Return zero vector if text is empty
            return np.zeros(self.embedding_dimension)
        
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding
    
    def generate_embeddings_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            
        Returns:
            Numpy array with embeddings (n_texts, embedding_dim)
        """
        if not texts:
            return np.array([])
        
        # Filter out empty texts
        valid_texts = [text if text and text.strip() else "empty" for text in texts]
        
        embeddings = self.model.encode(
            valid_texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 100
        )
        
        return embeddings
    
    def embed_well_summaries(self, well_summaries: List[Dict]) -> List[Dict]:
        """
        Generate embeddings for well summaries.
        
        Args:
            well_summaries: List of well summary dictionaries
            
        Returns:
            List of dictionaries with added 'embedding' field
        """
        texts = [summary.get('summary_text', '') for summary in well_summaries]
        embeddings = self.generate_embeddings_batch(texts)
        
        # Add embeddings to summaries
        for i, summary in enumerate(well_summaries):
            summary['embedding'] = embeddings[i].tolist()
            summary['embedding_id'] = f"well_{summary.get('well_name', 'unknown')}_{i}"
        
        return well_summaries
    
    def embed_interval_chunks(self, intervals: List[Dict]) -> List[Dict]:
        """
        Generate embeddings for depth interval chunks.
        
        Args:
            intervals: List of interval dictionaries
            
        Returns:
            List of dictionaries with added 'embedding' field
        """
        texts = [interval.get('description', '') for interval in intervals]
        embeddings = self.generate_embeddings_batch(texts)
        
        # Add embeddings to intervals
        for i, interval in enumerate(intervals):
            interval['embedding'] = embeddings[i].tolist()
            interval['embedding_id'] = (
                f"interval_{interval.get('well_name', 'unknown')}_"
                f"{interval.get('start_depth', 0):.1f}_{interval.get('end_depth', 0):.1f}"
            )
        
        return intervals
    
    def embed_formation_data(self, formation_tops_data: Dict[str, List[Dict]]) -> List[Dict]:
        """
        Generate embeddings for formation top information.
        
        Args:
            formation_tops_data: Dictionary mapping well names to formation tops
            
        Returns:
            List of dictionaries with formation data and embeddings
        """
        formation_embeddings = []
        
        for well_name, formations in formation_tops_data.items():
            for formation in formations:
                # Create more descriptive text for better searchability
                formation_name = formation.get('formation_name', 'Unknown')
                md = formation.get('md', 0)
                
                # Build comprehensive description with relevant keywords
                desc_parts = [
                    f"Well {well_name}",
                    f"Surface name: {formation_name}",
                    f"Formation top: {formation_name}",
                    f"Available surface: {formation_name}",
                    f"Formation at depth {md:.1f}m MD"
                ]
                
                if formation.get('tvd'):
                    desc_parts.append(f"TVD {formation.get('tvd'):.1f}m")
                if formation.get('tvdss'):
                    desc_parts.append(f"TVDSS {formation.get('tvdss'):.1f}m")
                
                # Add context about what this represents
                desc_parts.append(f"This is a formation top or surface marker for well {well_name}")
                
                text = ". ".join(desc_parts) + "."
                embedding = self.generate_embedding(text)
                
                formation_dict = {
                    **formation,
                    'embedding': embedding.tolist(),
                    'embedding_id': (
                        f"formation_{well_name}_{formation.get('formation_name', 'unknown')}_"
                        f"{formation.get('md', 0):.1f}"
                    ),
                    'description': text
                }
                
                formation_embeddings.append(formation_dict)
        
        return formation_embeddings
    
    def embed_query(self, query_text: str) -> np.ndarray:
        """
        Generate embedding for a query text.
        
        Args:
            query_text: Query text to embed
            
        Returns:
            Numpy array with query embedding
        """
        return self.generate_embedding(query_text)
    
    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for a list of texts (alias for generate_embeddings_batch).
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            
        Returns:
            Numpy array with embeddings (n_texts, embedding_dim)
        """
        return self.generate_embeddings_batch(texts, batch_size=batch_size)
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (0 to 1)
        """
        # Normalize embeddings
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Compute cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        return float(similarity)


if __name__ == "__main__":
    # Test the embedding generator
    generator = EmbeddingGenerator()
    
    # Test with sample text
    test_text = "Well 15/9-F-1 in VOLVE field with porosity 0.15 and permeability 50 mD"
    embedding = generator.generate_embedding(test_text)
    print(f"Generated embedding with dimension: {embedding.shape}")
    
    # Test batch processing
    test_texts = [
        "Well with high porosity",
        "Well with low permeability",
        "Formation Hugin at 3000m depth"
    ]
    embeddings = generator.generate_embeddings_batch(test_texts)
    print(f"Generated {len(embeddings)} embeddings with shape: {embeddings.shape}")

