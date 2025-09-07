"""
Embedding management module for generating and storing vector embeddings.
"""

import os
import numpy as np
from typing import List, Dict, Any, Optional
import openai
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from pathlib import Path


class EmbeddingManager:
    """Handles embedding generation and vector storage."""
    
    def __init__(self, model_name: str = "text-embedding-ada-002", use_openai: bool = True):
        """
        Initialize the embedding manager.
        
        Args:
            model_name: Name of the embedding model to use
            use_openai: Whether to use OpenAI embeddings or sentence-transformers
        """
        self.model_name = model_name
        self.use_openai = use_openai
        self.embedding_dim = None
        self.index = None
        self.chunks = []
        
        if use_openai:
            self._setup_openai()
        else:
            self._setup_sentence_transformers()
    
    def _setup_openai(self):
        """Setup OpenAI embedding model."""
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        # Set embedding dimension based on model
        if "ada-002" in self.model_name:
            self.embedding_dim = 1536
        elif "3-small" in self.model_name:
            self.embedding_dim = 1536
        elif "3-large" in self.model_name:
            self.embedding_dim = 3072
        else:
            self.embedding_dim = 1536  # Default
    
    def _setup_sentence_transformers(self):
        """Setup sentence-transformers model."""
        try:
            self.model = SentenceTransformer(self.model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
        except Exception as e:
            raise Exception(f"Failed to load sentence-transformers model {self.model_name}: {str(e)}")
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            Numpy array of embeddings
        """
        if self.use_openai:
            return self._generate_openai_embeddings(texts)
        else:
            return self._generate_sentence_transformer_embeddings(texts)
    
    def _generate_openai_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using OpenAI API."""
        try:
            # Process in batches to avoid rate limits
            batch_size = 100
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                response = openai.embeddings.create(
                    input=batch_texts,
                    model=self.model_name
                )
                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)
            
            return np.array(all_embeddings)
        except Exception as e:
            raise Exception(f"Error generating OpenAI embeddings: {str(e)}")
    
    def _generate_sentence_transformer_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using sentence-transformers."""
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return embeddings
        except Exception as e:
            raise Exception(f"Error generating sentence-transformer embeddings: {str(e)}")
    
    def create_index(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Create FAISS index from chunks.
        
        Args:
            chunks: List of chunk dictionaries with text and metadata
        """
        if not chunks:
            raise ValueError("No chunks provided")
        
        self.chunks = chunks
        texts = [chunk['text'] for chunk in chunks]
        
        # Generate embeddings
        print(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.generate_embeddings(texts)
        
        # Create FAISS index
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        
        print(f"Created FAISS index with {self.index.ntotal} vectors")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using the query.
        
        Args:
            query: Query text
            top_k: Number of top results to return
            
        Returns:
            List of similar chunks with scores
        """
        if self.index is None:
            raise ValueError("Index not created. Call create_index() first.")
        
        # Generate embedding for query
        query_embedding = self.generate_embeddings([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):
                chunk = self.chunks[idx].copy()
                chunk['similarity_score'] = float(score)
                results.append(chunk)
        
        return results
    
    def save_index(self, filepath: str) -> None:
        """Save the FAISS index and chunks to disk."""
        if self.index is None:
            raise ValueError("No index to save")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(filepath.with_suffix('.faiss')))
        
        # Save chunks metadata
        with open(filepath.with_suffix('.pkl'), 'wb') as f:
            pickle.dump({
                'chunks': self.chunks,
                'model_name': self.model_name,
                'use_openai': self.use_openai,
                'embedding_dim': self.embedding_dim
            }, f)
        
        print(f"Index saved to {filepath}")
    
    def load_index(self, filepath: str) -> None:
        """Load the FAISS index and chunks from disk."""
        filepath = Path(filepath)
        
        # Load FAISS index
        self.index = faiss.read_index(str(filepath.with_suffix('.faiss')))
        
        # Load chunks metadata
        with open(filepath.with_suffix('.pkl'), 'rb') as f:
            data = pickle.load(f)
            self.chunks = data['chunks']
            self.model_name = data['model_name']
            self.use_openai = data['use_openai']
            self.embedding_dim = data['embedding_dim']
        
        print(f"Index loaded from {filepath}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the current index."""
        if self.index is None:
            return {"status": "No index created"}
        
        return {
            "total_chunks": len(self.chunks),
            "index_size": self.index.ntotal,
            "embedding_dim": self.embedding_dim,
            "model_name": self.model_name,
            "use_openai": self.use_openai
        }


def test_embedding_manager():
    """Test function for the embedding manager."""
    # Create sample chunks
    sample_chunks = [
        {
            'text': 'Machine learning is a subset of artificial intelligence.',
            'metadata': {'chunk_id': 0, 'title': 'Test Doc'}
        },
        {
            'text': 'Supervised learning uses labeled training data.',
            'metadata': {'chunk_id': 1, 'title': 'Test Doc'}
        },
        {
            'text': 'Unsupervised learning finds patterns in unlabeled data.',
            'metadata': {'chunk_id': 2, 'title': 'Test Doc'}
        }
    ]
    
    # Test with sentence-transformers (no API key required)
    try:
        manager = EmbeddingManager(
            model_name="all-MiniLM-L6-v2", 
            use_openai=False
        )
        
        manager.create_index(sample_chunks)
        
        # Test search
        results = manager.search("What is machine learning?", top_k=2)
        
        print("Search results:")
        for i, result in enumerate(results):
            print(f"{i+1}. Score: {result['similarity_score']:.3f}")
            print(f"   Text: {result['text']}")
            print()
        
        print("Manager stats:", manager.get_stats())
        
    except Exception as e:
        print(f"Test failed: {e}")
        print("Note: This test requires sentence-transformers to be installed.")


if __name__ == "__main__":
    test_embedding_manager()
