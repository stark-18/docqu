"""
Text chunking module for splitting documents into manageable chunks for embedding.
"""

import re
from typing import List, Dict, Any
import tiktoken


class TextChunker:
    """Handles splitting text into chunks for embedding and retrieval."""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize the text chunker.
        
        Args:
            chunk_size: Maximum number of tokens per chunk
            chunk_overlap: Number of tokens to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding
    
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Split text into chunks with metadata.
        
        Args:
            text: The text to chunk
            metadata: Additional metadata to include with each chunk
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text.strip():
            return []
        
        # Split text into sentences first
        sentences = self._split_into_sentences(text)
        
        chunks = []
        current_chunk = ""
        current_tokens = 0
        chunk_id = 0
        
        for sentence in sentences:
            sentence_tokens = len(self.encoding.encode(sentence))
            
            # If adding this sentence would exceed chunk size, save current chunk
            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                chunks.append(self._create_chunk(
                    current_chunk.strip(), 
                    chunk_id, 
                    metadata
                ))
                chunk_id += 1
                
                # Start new chunk with overlap
                current_chunk = self._get_overlap_text(current_chunk, sentence)
                current_tokens = len(self.encoding.encode(current_chunk))
            else:
                current_chunk += " " + sentence if current_chunk else sentence
                current_tokens += sentence_tokens
        
        # Add the last chunk if it has content
        if current_chunk.strip():
            chunks.append(self._create_chunk(
                current_chunk.strip(), 
                chunk_id, 
                metadata
            ))
        
        return chunks
    
    def chunk_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk a parsed document.
        
        Args:
            document: Parsed document dictionary
            
        Returns:
            List of chunk dictionaries
        """
        metadata = {
            'title': document.get('title', 'Unknown'),
            'file_type': document.get('file_type', 'unknown'),
            'char_count': document.get('char_count', 0)
        }
        
        # Add page information if available
        if 'page_contents' in document:
            metadata['total_pages'] = document.get('total_pages', 0)
        
        return self.chunk_text(document['full_text'], metadata)
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex."""
        # Simple sentence splitting - can be improved with more sophisticated NLP
        sentence_endings = r'[.!?]+'
        sentences = re.split(sentence_endings, text)
        
        # Clean up sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:  # Filter out very short fragments
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def _get_overlap_text(self, previous_chunk: str, new_sentence: str) -> str:
        """Get overlap text from the end of previous chunk."""
        if not previous_chunk:
            return new_sentence
        
        # Get the last few sentences from previous chunk for overlap
        sentences = self._split_into_sentences(previous_chunk)
        overlap_sentences = sentences[-2:] if len(sentences) >= 2 else sentences[-1:]
        overlap_text = " ".join(overlap_sentences)
        
        # Ensure we don't exceed overlap size
        overlap_tokens = len(self.encoding.encode(overlap_text))
        if overlap_tokens > self.chunk_overlap:
            # Truncate to fit overlap size
            words = overlap_text.split()
            while len(self.encoding.encode(" ".join(words))) > self.chunk_overlap and words:
                words.pop(0)
            overlap_text = " ".join(words)
        
        return overlap_text + " " + new_sentence if overlap_text else new_sentence
    
    def _create_chunk(self, text: str, chunk_id: int, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Create a chunk dictionary with metadata."""
        chunk_metadata = metadata.copy() if metadata else {}
        chunk_metadata.update({
            'chunk_id': chunk_id,
            'char_count': len(text),
            'token_count': len(self.encoding.encode(text))
        })
        
        return {
            'text': text,
            'metadata': chunk_metadata
        }
    
    def get_chunk_stats(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about the chunks."""
        if not chunks:
            return {}
        
        token_counts = [chunk['metadata']['token_count'] for chunk in chunks]
        char_counts = [chunk['metadata']['char_count'] for chunk in chunks]
        
        return {
            'total_chunks': len(chunks),
            'avg_tokens_per_chunk': sum(token_counts) / len(token_counts),
            'min_tokens': min(token_counts),
            'max_tokens': max(token_counts),
            'avg_chars_per_chunk': sum(char_counts) / len(char_counts),
            'total_tokens': sum(token_counts),
            'total_chars': sum(char_counts)
        }


def test_chunker():
    """Test function for the text chunker."""
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    
    sample_text = """
    Machine learning is a subset of artificial intelligence that focuses on algorithms and statistical models.
    Supervised learning uses labeled training data to learn a mapping from inputs to outputs.
    Unsupervised learning finds hidden patterns in data without labeled examples.
    Reinforcement learning learns through interaction with an environment using rewards and penalties.
    Deep learning uses neural networks with multiple layers to model complex patterns.
    Natural language processing applies machine learning to understand and generate human language.
    Computer vision enables machines to interpret and understand visual information from images.
    """
    
    chunks = chunker.chunk_text(sample_text, {'source': 'test'})
    
    print(f"Created {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1}:")
        print(f"Text: {chunk['text'][:100]}...")
        print(f"Tokens: {chunk['metadata']['token_count']}")
        print(f"Chars: {chunk['metadata']['char_count']}")
    
    stats = chunker.get_chunk_stats(chunks)
    print(f"\nChunk Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    test_chunker()
