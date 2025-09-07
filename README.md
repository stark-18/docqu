# Document Question & Answering System

A powerful RAG (Retrieval-Augmented Generation) system that allows users to upload documents and ask natural language questions about their content.

## Features

- **Multi-format Support**: PDF, Markdown, and HTML documents
- **Intelligent Chunking**: Smart text segmentation with configurable overlap
- **Vector Search**: Semantic similarity search using embeddings
- **RAG Pipeline**: Combines retrieval with LLM generation for accurate answers
- **Web Interface**: User-friendly Streamlit interface
- **Source Citations**: Shows which parts of documents support answers
- **Chat History**: Maintains conversation context

## Tech Stack

- **Document Parsing**: PyPDF2, pdfplumber, BeautifulSoup, markdown
- **Embeddings**: OpenAI text-embedding-ada-002 or Sentence Transformers
- **Vector Database**: FAISS
- **LLM**: OpenAI GPT-3.5/GPT-4
- **Web Interface**: Streamlit
- **Language**: Python 3.8+

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd docqa
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   ```bash
   cp env_example.txt .env
   # Edit .env and add your OpenAI API key
   ```

4. **Run the application**:
   ```bash
   streamlit run app.py
   ```

## Usage

### 1. Upload Documents

- Navigate to the "Upload Documents" tab
- Upload PDF, Markdown, or HTML files
- Configure chunk size and overlap settings
- Click "Process Documents" to create embeddings

### 2. Ask Questions

- Go to the "Ask Questions" tab
- Type your question in natural language
- Get AI-powered answers with source citations
- View chat history and confidence scores

### 3. Example Questions

- "What is the main topic of this document?"
- "Summarize the key points from chapter 3"
- "What are the advantages and disadvantages mentioned?"
- "Explain the methodology used in this research"

## Configuration

### Environment Variables

Create a `.env` file with:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

### Chunking Parameters

- **Chunk Size**: Number of tokens per chunk (100-1000, default: 500)
- **Chunk Overlap**: Tokens to overlap between chunks (0-200, default: 50)

### Model Selection

The system automatically chooses between:
- **OpenAI embeddings** (if API key is available)
- **Sentence Transformers** (fallback, no API key required)

## API Usage

You can also use the system programmatically:

```python
from document_parser import DocumentParser
from text_chunker import TextChunker
from embedding_manager import EmbeddingManager
from rag_system import RAGSystem

# Parse document
parser = DocumentParser()
document = parser.parse_document("path/to/document.pdf")

# Chunk text
chunker = TextChunker(chunk_size=500, chunk_overlap=50)
chunks = chunker.chunk_document(document)

# Create embeddings
embedding_manager = EmbeddingManager(use_openai=True)
embedding_manager.create_index(chunks)

# Initialize RAG system
rag = RAGSystem(embedding_manager)

# Ask question
response = rag.answer_question("What is this document about?")
print(response['answer'])
```

## File Structure

```
docqa/
├── app.py                 # Streamlit web interface
├── document_parser.py     # Document parsing logic
├── text_chunker.py       # Text chunking utilities
├── embedding_manager.py  # Embedding generation and storage
├── rag_system.py         # RAG pipeline implementation
├── requirements.txt      # Python dependencies
├── env_example.txt       # Environment variables template
└── README.md            # This file
```

## Supported Document Formats

### PDF Files
- Extracts text using pdfplumber
- Handles multi-page documents
- Preserves page information for citations

### Markdown Files
- Supports standard Markdown syntax
- Extracts title from first heading
- Handles tables and code blocks

### HTML Files
- Parses HTML using BeautifulSoup
- Removes script and style elements
- Extracts clean text content

## Advanced Features

### Source Citations
- Shows which document sections support answers
- Displays similarity scores
- Includes page numbers for PDFs

### Confidence Scoring
- Calculates confidence based on similarity scores
- Helps assess answer reliability

### Chat History
- Maintains conversation context
- Allows follow-up questions
- Exportable chat sessions

## Troubleshooting

### Common Issues

1. **"No module named 'openai'"**
   - Run: `pip install -r requirements.txt`

2. **"OPENAI_API_KEY not set"**
   - Create `.env` file with your API key
   - Or use sentence-transformers (slower but free)

3. **"Error parsing PDF"**
   - Ensure PDF is not password-protected
   - Try a different PDF file

4. **"No relevant information found"**
   - Try rephrasing your question
   - Check if document was processed successfully
   - Increase chunk overlap or size

### Performance Tips

- Use smaller chunk sizes for more precise answers
- Increase chunk overlap for better context
- Use OpenAI embeddings for better quality (requires API key)
- Process documents in batches for large collections

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for embedding and language models
- Streamlit for the web interface
- FAISS for vector similarity search
- The open-source community for various libraries
