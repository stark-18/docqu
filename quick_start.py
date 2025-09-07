"""
Quick start script for the Document Q&A system.
This version works with minimal dependencies for testing.
"""

import os
import sys
from pathlib import Path


def check_basic_imports():
    """Check if basic Python modules are available."""
    print("Checking basic Python modules...")
    
    try:
        import re
        import json
        import tempfile
        print("✅ Basic modules available")
        return True
    except ImportError as e:
        print(f"❌ Missing basic modules: {e}")
        return False


def create_minimal_test():
    """Create a minimal test that doesn't require external dependencies."""
    print("\nCreating minimal test...")
    
    # Test document parsing with a simple text file
    sample_text = """
    Machine Learning Basics
    
    Machine learning is a subset of artificial intelligence that focuses on algorithms and statistical models.
    
    Types of Machine Learning:
    1. Supervised Learning - Uses labeled training data
    2. Unsupervised Learning - Finds patterns in unlabeled data
    3. Reinforcement Learning - Learns through interaction with environment
    
    Deep learning uses neural networks with multiple layers to model complex patterns.
    """
    
    # Create a simple text file
    test_file = Path("test_document.txt")
    with open(test_file, 'w') as f:
        f.write(sample_text)
    
    print(f"✅ Created test document: {test_file}")
    return test_file


def test_simple_parsing(file_path):
    """Test simple text parsing without external dependencies."""
    print(f"\nTesting simple parsing of {file_path}...")
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Simple parsing
        lines = content.strip().split('\n')
        title = lines[0] if lines else "Unknown"
        
        # Simple chunking (split by paragraphs)
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        print(f"✅ Parsed document:")
        print(f"   Title: {title}")
        print(f"   Paragraphs: {len(paragraphs)}")
        print(f"   Characters: {len(content)}")
        
        # Show first paragraph
        if paragraphs:
            print(f"   First paragraph: {paragraphs[0][:100]}...")
        
        return {
            'title': title,
            'paragraphs': paragraphs,
            'content': content
        }
        
    except Exception as e:
        print(f"❌ Error parsing: {e}")
        return None


def test_simple_search(document, query):
    """Test simple text search without embeddings."""
    print(f"\nTesting simple search for: '{query}'...")
    
    if not document:
        print("❌ No document to search")
        return
    
    query_lower = query.lower()
    results = []
    
    for i, paragraph in enumerate(document['paragraphs']):
        if query_lower in paragraph.lower():
            results.append({
                'paragraph_id': i,
                'text': paragraph,
                'relevance': paragraph.lower().count(query_lower)
            })
    
    if results:
        print(f"✅ Found {len(results)} relevant paragraphs:")
        for i, result in enumerate(results[:3], 1):  # Show top 3
            print(f"   {i}. Paragraph {result['paragraph_id']} (relevance: {result['relevance']})")
            print(f"      {result['text'][:100]}...")
    else:
        print("❌ No relevant content found")


def main():
    """Main function for quick start testing."""
    print("🚀 Document Q&A System - Quick Start")
    print("=" * 50)
    
    # Check basic imports
    if not check_basic_imports():
        print("\n❌ Basic Python environment not ready")
        return False
    
    # Create test document
    test_file = create_minimal_test()
    
    # Test parsing
    document = test_simple_parsing(test_file)
    
    if document:
        # Test search
        test_queries = [
            "machine learning",
            "supervised learning",
            "neural networks"
        ]
        
        for query in test_queries:
            test_simple_search(document, query)
    
    # Clean up
    if test_file.exists():
        test_file.unlink()
        print(f"\n✅ Cleaned up test file")
    
    print("\n" + "=" * 50)
    print("🎉 Quick start test completed!")
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Set up OpenAI API key in .env file (optional)")
    print("3. Run full system: streamlit run app.py")
    print("4. Or run full test: python test_system.py")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
