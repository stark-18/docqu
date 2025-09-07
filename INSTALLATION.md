# Installation Guide

## Quick Start (Minimal Dependencies)

The system includes a quick start script that works with basic Python modules:

```bash
python3 quick_start.py
```

This will test the basic functionality without requiring external dependencies.

## Full Installation

### 1. Install Python Dependencies

```bash
# Install all required packages
pip3 install -r requirements.txt

# Or install individually if you encounter issues:
pip3 install streamlit
pip3 install openai
pip3 install sentence-transformers
pip3 install faiss-cpu
pip3 install PyPDF2
pip3 install pdfplumber
pip3 install beautifulsoup4
pip3 install markdown
pip3 install html2text
pip3 install python-dotenv
pip3 install numpy
pip3 install pandas
pip3 install tiktoken
pip3 install langchain
pip3 install langchain-openai
```

### 2. Set Up Environment Variables

```bash
# Copy the example environment file
cp env_example.txt .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=your_openai_api_key_here
```

**Note**: The system can work without an OpenAI API key by using sentence-transformers, but OpenAI provides better quality embeddings and answers.

### 3. Test the System

```bash
# Run the full system test
python3 test_system.py

# Or run the web interface
streamlit run app.py
```

## Troubleshooting

### Common Issues

1. **Network Timeout During Installation**
   ```bash
   # Try installing with increased timeout
   pip3 install --timeout 1000 -r requirements.txt
   
   # Or install packages individually
   pip3 install streamlit
   pip3 install openai
   # ... etc
   ```

2. **Permission Errors**
   ```bash
   # Install for user only
   pip3 install --user -r requirements.txt
   ```

3. **Python Version Issues**
   - Ensure you're using Python 3.8 or higher
   - Check with: `python3 --version`

4. **Missing Dependencies**
   ```bash
   # Upgrade pip first
   python3 -m pip install --upgrade pip
   
   # Then install requirements
   pip3 install -r requirements.txt
   ```

### Alternative Installation Methods

#### Using Conda
```bash
# Create a new conda environment
conda create -n docqa python=3.9
conda activate docqa

# Install packages
pip install -r requirements.txt
```

#### Using Virtual Environment
```bash
# Create virtual environment
python3 -m venv docqa_env
source docqa_env/bin/activate  # On Windows: docqa_env\Scripts\activate

# Install packages
pip install -r requirements.txt
```

## System Requirements

- **Python**: 3.8 or higher
- **Memory**: At least 2GB RAM (4GB+ recommended)
- **Storage**: 1GB free space for dependencies
- **Internet**: Required for downloading models and API calls

## Optional: OpenAI API Setup

1. Create an account at [OpenAI](https://openai.com)
2. Generate an API key from the dashboard
3. Add it to your `.env` file:
   ```
   OPENAI_API_KEY=sk-your-api-key-here
   ```

## Running the System

### Web Interface
```bash
streamlit run app.py
```
Then open your browser to `http://localhost:8501`

### Command Line Testing
```bash
python3 test_system.py
```

### Programmatic Usage
```python
from document_parser import DocumentParser
from text_chunker import TextChunker
from embedding_manager import EmbeddingManager
from rag_system import RAGSystem

# Your code here...
```

## File Structure

```
docqa/
├── app.py                 # Streamlit web interface
├── document_parser.py     # Document parsing
├── text_chunker.py       # Text chunking
├── embedding_manager.py  # Embedding generation
├── rag_system.py         # RAG pipeline
├── test_system.py        # Full system test
├── quick_start.py        # Minimal test
├── setup.py              # Setup script
├── requirements.txt      # Dependencies
├── .env                  # Environment variables
├── sample_documents/     # Test documents
└── README.md            # Documentation
```

## Support

If you encounter issues:

1. Check the error messages carefully
2. Ensure all dependencies are installed
3. Verify Python version compatibility
4. Check internet connection for model downloads
5. Review the troubleshooting section above

For additional help, refer to the main README.md file.
