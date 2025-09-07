"""
Test script for the Document Q&A system.
"""

import os
from pathlib import Path
from document_parser import DocumentParser
from text_chunker import TextChunker
from embedding_manager import EmbeddingManager
from rag_system import RAGSystem


def test_document_parsing():
    """Test document parsing functionality."""
    print("Testing document parsing...")
    
    parser = DocumentParser()
    
    # Test with sample markdown file
    sample_file = Path("sample_documents/machine_learning_basics.md")
    if sample_file.exists():
        try:
            document = parser.parse_document(str(sample_file))
            print(f"✅ Successfully parsed: {document['title']}")
            print(f"   File type: {document['file_type']}")
            print(f"   Character count: {document['char_count']}")
            print(f"   Content preview: {document['full_text'][:100]}...")
            return document
        except Exception as e:
            print(f"❌ Error parsing document: {e}")
            return None
    else:
        print("❌ Sample document not found")
        return None


def test_text_chunking(document):
    """Test text chunking functionality."""
    print("\nTesting text chunking...")
    
    if not document:
        print("❌ No document to chunk")
        return None
    
    chunker = TextChunker(chunk_size=300, chunk_overlap=50)
    chunks = chunker.chunk_document(document)
    
    print(f"✅ Created {len(chunks)} chunks")
    
    # Show chunk statistics
    stats = chunker.get_chunk_stats(chunks)
    print(f"   Average tokens per chunk: {stats['avg_tokens_per_chunk']:.1f}")
    print(f"   Total tokens: {stats['total_tokens']}")
    
    # Show first chunk
    if chunks:
        print(f"   First chunk preview: {chunks[0]['text'][:100]}...")
    
    return chunks


def test_embedding_generation(chunks):
    """Test embedding generation."""
    print("\nTesting embedding generation...")
    
    if not chunks:
        print("❌ No chunks to embed")
        return None
    
    try:
        # Use sentence-transformers for testing (no API key required)
        embedding_manager = EmbeddingManager(
            model_name="all-MiniLM-L6-v2",
            use_openai=False
        )
        
        embedding_manager.create_index(chunks)
        print("✅ Successfully created embeddings and FAISS index")
        
        # Test search
        search_results = embedding_manager.search("What is machine learning?", top_k=3)
        print(f"   Search test returned {len(search_results)} results")
        
        if search_results:
            print(f"   Top result similarity: {search_results[0]['similarity_score']:.3f}")
        
        return embedding_manager
        
    except Exception as e:
        print(f"❌ Error creating embeddings: {e}")
        return None


def test_rag_system(embedding_manager):
    """Test RAG system."""
    print("\nTesting RAG system...")
    
    if not embedding_manager:
        print("❌ No embedding manager available")
        return None
    
    try:
        # Check if OpenAI API key is available
        use_openai = os.getenv("OPENAI_API_KEY") is not None
        
        if not use_openai:
            print("⚠️  OpenAI API key not found. RAG system requires API key for full functionality.")
            print("   Set OPENAI_API_KEY environment variable to test complete RAG functionality.")
            return None
        
        rag = RAGSystem(embedding_manager)
        
        # Test questions
        test_questions = [
            "What is machine learning?",
            "What are the types of machine learning?",
            "What is the difference between supervised and unsupervised learning?"
        ]
        
        for question in test_questions:
            print(f"\n   Question: {question}")
            response = rag.answer_question(question, top_k=3)
            print(f"   Answer: {response['answer'][:150]}...")
            print(f"   Confidence: {response['confidence']:.3f}")
            print(f"   Sources: {response['num_sources']}")
        
        return rag
        
    except Exception as e:
        print(f"❌ Error testing RAG system: {e}")
        return None


def main():
    """Run all tests."""
    print("🚀 Testing Document Q&A System")
    print("=" * 50)
    
    # Test document parsing
    document = test_document_parsing()
    
    # Test text chunking
    chunks = test_text_chunking(document)
    
    # Test embedding generation
    embedding_manager = test_embedding_generation(chunks)
    
    # Test RAG system
    rag_system = test_rag_system(embedding_manager)
    
    print("\n" + "=" * 50)
    if rag_system:
        print("🎉 All tests completed successfully!")
        print("   The system is ready to use.")
    else:
        print("⚠️  Some tests failed or require additional setup.")
        print("   Check the error messages above for details.")
    
    print("\nTo run the web interface:")
    print("   streamlit run app.py")


if __name__ == "__main__":
    main()
