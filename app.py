"""
Streamlit web interface for the Document Q&A system.
"""

import streamlit as st
import os
import tempfile
from pathlib import Path
from typing import List, Dict, Any
import time

# Import our custom modules
from document_parser import DocumentParser
from text_chunker import TextChunker
from embedding_manager import EmbeddingManager
from rag_system import RAGSystem


def initialize_session_state():
    """Initialize session state variables."""
    if 'documents_processed' not in st.session_state:
        st.session_state.documents_processed = False
    if 'embedding_manager' not in st.session_state:
        st.session_state.embedding_manager = None
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []


def process_documents(uploaded_files: List[Any], chunk_size: int, chunk_overlap: int) -> bool:
    """Process uploaded documents and create embeddings."""
    try:
        # Initialize components
        parser = DocumentParser()
        chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        # Check if OpenAI API key is available and not empty
        api_key = os.getenv("OPENAI_API_KEY")
        use_openai = api_key is not None and api_key.strip() != ""
        model_name = "text-embedding-ada-002" if use_openai else "all-MiniLM-L6-v2"
        
        embedding_manager = EmbeddingManager(
            model_name=model_name,
            use_openai=use_openai
        )
        
        all_chunks = []
        
        # Process each uploaded file
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            try:
                # Parse document
                document = parser.parse_document(tmp_file_path)
                
                # Chunk document
                chunks = chunker.chunk_document(document)
                all_chunks.extend(chunks)
                
                st.success(f"Processed {uploaded_file.name}: {len(chunks)} chunks")
                
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
            finally:
                # Clean up temporary file
                os.unlink(tmp_file_path)
        
        if not all_chunks:
            st.error("No chunks were created from the uploaded files.")
            return False
        
        # Create embeddings
        with st.spinner("Creating embeddings..."):
            embedding_manager.create_index(all_chunks)
        
        # Initialize RAG system
        rag_system = RAGSystem(embedding_manager)
        
        # Store in session state
        st.session_state.embedding_manager = embedding_manager
        st.session_state.rag_system = rag_system
        st.session_state.documents_processed = True
        
        # Show statistics
        stats = embedding_manager.get_stats()
        chunk_stats = chunker.get_chunk_stats(all_chunks)
        
        st.success(f"Successfully processed {len(uploaded_files)} documents!")
        st.info(f"Total chunks: {stats['total_chunks']}, Average tokens per chunk: {chunk_stats['avg_tokens_per_chunk']:.1f}")
        
        return True
        
    except Exception as e:
        st.error(f"Error processing documents: {str(e)}")
        return False


def display_chat_interface():
    """Display the chat interface for asking questions."""
    st.subheader("💬 Ask Questions About Your Documents")
    
    # Question input
    question = st.text_input(
        "Enter your question:",
        placeholder="e.g., What is the main topic of the document?",
        key="question_input"
    )
    
    col1, col2 = st.columns([1, 4])
    
    with col1:
        ask_button = st.button("Ask Question", type="primary")
    
    with col2:
        clear_chat = st.button("Clear Chat History")
    
    if clear_chat:
        st.session_state.chat_history = []
        st.rerun()
    
    # Process question
    if ask_button and question:
        if not st.session_state.documents_processed:
            st.warning("Please upload and process documents first.")
            return
        
        with st.spinner("Thinking..."):
            try:
                # Get answer from RAG system
                response = st.session_state.rag_system.answer_question(
                    question, 
                    top_k=5, 
                    include_sources=True
                )
                
                # Add to chat history
                st.session_state.chat_history.append({
                    'question': question,
                    'answer': response['answer'],
                    'sources': response.get('sources', []),
                    'confidence': response.get('confidence', 0),
                    'timestamp': time.time()
                })
                
            except Exception as e:
                st.error(f"Error getting answer: {str(e)}")
    
    # Display chat history
    if st.session_state.chat_history:
        st.subheader("Chat History")
        
        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            with st.expander(f"Q: {chat['question'][:50]}...", expanded=(i == 0)):
                st.write("**Question:**", chat['question'])
                st.write("**Answer:**", chat['answer'])
                
                if chat.get('confidence', 0) > 0:
                    st.write(f"**Confidence:** {chat['confidence']:.2f}")
                
                if chat.get('sources'):
                    st.write("**Sources:**")
                    for j, source in enumerate(chat['sources'][:3], 1):  # Show top 3 sources
                        with st.container():
                            st.write(f"{j}. **{source['title']}** (Score: {source['similarity_score']:.3f})")
                            st.write(f"   {source['text_preview']}")
                            if 'page_number' in source:
                                st.write(f"   *Page {source['page_number']}*")


def display_document_upload():
    """Display document upload interface."""
    st.subheader("📄 Upload Documents")
    
    # File upload
    uploaded_files = st.file_uploader(
        "Choose files to upload",
        type=['pdf', 'md', 'markdown', 'html', 'htm'],
        accept_multiple_files=True,
        help="Supported formats: PDF, Markdown (.md), HTML (.html, .htm)"
    )
    
    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files
        
        # Display uploaded files
        st.write("**Uploaded Files:**")
        for file in uploaded_files:
            st.write(f"• {file.name} ({file.size} bytes)")
    
    # Processing options
    st.subheader("⚙️ Processing Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        chunk_size = st.slider(
            "Chunk Size (tokens)",
            min_value=100,
            max_value=1000,
            value=500,
            help="Number of tokens per text chunk"
        )
    
    with col2:
        chunk_overlap = st.slider(
            "Chunk Overlap (tokens)",
            min_value=0,
            max_value=200,
            value=50,
            help="Number of tokens to overlap between chunks"
        )
    
    # Process button
    if st.button("Process Documents", type="primary", disabled=not uploaded_files):
        if not uploaded_files:
            st.warning("Please upload some documents first.")
        else:
            # Check for API key
            if not os.getenv("OPENAI_API_KEY"):
                st.warning("⚠️ OpenAI API key not found. Using sentence-transformers model (may be slower).")
            
            process_documents(uploaded_files, chunk_size, chunk_overlap)


def display_system_info():
    """Display system information and status."""
    st.sidebar.subheader("System Status")
    
    if st.session_state.documents_processed:
        st.sidebar.success("✅ Documents Processed")
        
        if st.session_state.embedding_manager:
            stats = st.session_state.embedding_manager.get_stats()
            st.sidebar.write(f"**Total Chunks:** {stats['total_chunks']}")
            st.sidebar.write(f"**Model:** {stats['model_name']}")
            st.sidebar.write(f"**Using OpenAI:** {'Yes' if stats['use_openai'] else 'No'}")
    else:
        st.sidebar.warning("⚠️ No Documents Processed")
    
    st.sidebar.subheader("About")
    st.sidebar.write("""
    This is a Document Q&A system that uses:
    - **RAG** (Retrieval-Augmented Generation)
    - **Vector embeddings** for semantic search
    - **LLM** for answer generation
    
    Upload documents and ask questions about their content!
    """)


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Document Q&A System",
        page_icon="📚",
        layout="wide"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.title("📚 Document Question & Answering System")
    st.markdown("Upload documents and ask questions about their content using AI-powered retrieval and generation.")
    
    # Sidebar
    display_system_info()
    
    # Main content
    tab1, tab2 = st.tabs(["📄 Upload Documents", "💬 Ask Questions"])
    
    with tab1:
        display_document_upload()
    
    with tab2:
        if st.session_state.documents_processed:
            display_chat_interface()
        else:
            st.info("Please upload and process documents first using the 'Upload Documents' tab.")
    
    # Footer
    st.markdown("---")
    st.markdown("Built with Streamlit, OpenAI, and FAISS")


if __name__ == "__main__":
    main()
