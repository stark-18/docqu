"""
Document parsing module for handling PDFs, Markdown, and HTML files.
"""

import os
import re
from typing import List, Dict, Any
import PyPDF2
import pdfplumber
import markdown
from bs4 import BeautifulSoup
import html2text
from pathlib import Path


class DocumentParser:
    """Handles parsing of various document formats."""
    
    def __init__(self):
        self.html_converter = html2text.HTML2Text()
        self.html_converter.ignore_links = False
        self.html_converter.ignore_images = True
    
    def parse_document(self, file_path: str) -> Dict[str, Any]:
        """
        Parse a document and return structured text content.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary containing parsed content and metadata
        """
        file_path = Path(file_path)
        file_extension = file_path.suffix.lower()
        
        if file_extension == '.pdf':
            return self._parse_pdf(file_path)
        elif file_extension in ['.md', '.markdown']:
            return self._parse_markdown(file_path)
        elif file_extension in ['.html', '.htm']:
            return self._parse_html(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    
    def _parse_pdf(self, file_path: Path) -> Dict[str, Any]:
        """Parse PDF file using pdfplumber for better text extraction."""
        try:
            with pdfplumber.open(file_path) as pdf:
                text_content = []
                page_contents = []
                
                for page_num, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text()
                    if page_text:
                        # Clean up the text
                        page_text = self._clean_text(page_text)
                        text_content.append(page_text)
                        page_contents.append({
                            'page_number': page_num,
                            'content': page_text,
                            'char_count': len(page_text)
                        })
                
                full_text = '\n\n'.join(text_content)
                
                return {
                    'title': file_path.stem,
                    'file_type': 'pdf',
                    'total_pages': len(pdf.pages),
                    'full_text': full_text,
                    'page_contents': page_contents,
                    'char_count': len(full_text)
                }
        except Exception as e:
            raise Exception(f"Error parsing PDF {file_path}: {str(e)}")
    
    def _parse_markdown(self, file_path: Path) -> Dict[str, Any]:
        """Parse Markdown file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Convert markdown to HTML first, then to text
            html = markdown.markdown(content, extensions=['tables', 'codehilite'])
            text_content = self.html_converter.handle(html)
            text_content = self._clean_text(text_content)
            
            # Extract title from first heading or filename
            title = self._extract_title_from_markdown(content) or file_path.stem
            
            return {
                'title': title,
                'file_type': 'markdown',
                'full_text': text_content,
                'char_count': len(text_content)
            }
        except Exception as e:
            raise Exception(f"Error parsing Markdown {file_path}: {str(e)}")
    
    def _parse_html(self, file_path: Path) -> Dict[str, Any]:
        """Parse HTML file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            soup = BeautifulSoup(content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Convert to text
            text_content = self.html_converter.handle(str(soup))
            text_content = self._clean_text(text_content)
            
            # Extract title
            title_tag = soup.find('title')
            title = title_tag.get_text().strip() if title_tag else file_path.stem
            
            return {
                'title': title,
                'file_type': 'html',
                'full_text': text_content,
                'char_count': len(text_content)
            }
        except Exception as e:
            raise Exception(f"Error parsing HTML {file_path}: {str(e)}")
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove excessive newlines
        text = re.sub(r'\n\s*\n', '\n\n', text)
        # Strip leading/trailing whitespace
        text = text.strip()
        return text
    
    def _extract_title_from_markdown(self, content: str) -> str:
        """Extract title from markdown content (first heading)."""
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('# '):
                return line[2:].strip()
        return None


def test_parser():
    """Test function for the document parser."""
    parser = DocumentParser()
    
    # Test with a sample markdown content
    test_md_content = """# Test Document

This is a test document for the Q&A system.

## Section 1
This section contains some sample content about machine learning.

### Subsection
Supervised learning uses labeled data, while unsupervised learning finds patterns in unlabeled data.
"""
    
    # Create a temporary test file
    test_file = Path("test_document.md")
    with open(test_file, 'w') as f:
        f.write(test_md_content)
    
    try:
        result = parser.parse_document(str(test_file))
        print("Parsing successful!")
        print(f"Title: {result['title']}")
        print(f"Type: {result['file_type']}")
        print(f"Character count: {result['char_count']}")
        print(f"Content preview: {result['full_text'][:200]}...")
    finally:
        # Clean up test file
        if test_file.exists():
            test_file.unlink()


if __name__ == "__main__":
    test_parser()
